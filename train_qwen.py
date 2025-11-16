from tqdm import tqdm
import argparse
import gc
import random

import numpy as np
from transformers import AutoTokenizer

import needle as ndl
import needle.nn as nn
from needle import Tensor


import pynvml

# --- 在脚本开头执行一次初始化 ---
try:
    pynvml.nvmlInit()
    # 假设您使用的是 GPU 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(3) 
    print("NVML 初始化成功。")
except Exception as e:
    print(f"无法初始化 NVML: {e}。内存监控将被禁用。")
    handle = None

def log_gpu_memory(point_in_code=""):
    """
    在代码的特定点记录当前 GPU 显存使用情况。
    """
    if handle is None:
        return # 如果 NVML 未初始化，则跳过
    try:
        # 获取关于显存的信息
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = mem_info.used / 1024**2
        total_mb = mem_info.total / 1024**2
        
        # 打印格式化的信息
        print(f"--- GPU 显存 [{point_in_code}]: {used_mb:.2f} MB / {total_mb:.2f} MB ---")
        
    except Exception as e:
        print(f"获取 GPU 显存信息时出错: {e}")


class Dataset:
    def __init__(self, doc_path: str, seq_len: int):
        self.doc_path = doc_path
        self.seq_len = seq_len
        
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        with open(doc_path, "r") as f:
            text = f.read()
            token_ids = self.tokenizer(text).input_ids
            # remove the last incomplete sequence
            token_ids = token_ids[:len(token_ids) - (len(token_ids) % seq_len)]
        self.token_ids = np.array(token_ids, dtype=np.int32)
        self.num_samples = len(self.token_ids) // seq_len
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(self.num_samples))
            return [self[i] for i in indices]

        start = idx * self.seq_len
        end = (idx + 1) * self.seq_len
        return ndl.Tensor(
            self.token_ids[start:end],
            device=ndl.cuda(),
            dtype="int32",
        )

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.num_batches = self.num_samples // batch_size
        self.idx = 0
        self.order = list(range(self.num_samples))  # sample index list

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            random.shuffle(self.order)  # shuffle once per epoch
        return self

    def __len__(self):
        return self.num_batches
    
    def restart(self):
        self.__iter__()

    def __next__(self):
        if self.idx < self.num_batches:
            start = self.idx * self.batch_size
            end = start + self.batch_size

            batch_indices = self.order[start:end]

            batch = ndl.stack([self.dataset[i] for i in batch_indices], axis=0)

            self.idx += 1
            return batch
        else:
            raise StopIteration

class QwenLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_attention_heads, device=ndl.cuda(), dtype="float32"):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        # assert embedding_size == num_head * dim_head
        head_dim = hidden_size // num_attention_heads
        self.encoding_layer = nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=hidden_size,
                                    device=device,
                                    dtype=dtype)

        # self.transformer_layers = nn.Sequential(*[nn.TransformerLayer(q_features=hidden_size,
        #                                        num_head=num_attention_heads,
        #                                        dim_head=head_dim,
        #                                        hidden_size=hidden_size,
        #                                        device=device,
        #                                        dtype=dtype) for _ in range(num_layers)])
        self.transformer_layers = nn.Transformer(hidden_size,
                                                 num_attention_heads,
                                                 num_layers,
                                                 num_head=num_attention_heads,
                                                 dim_head=head_dim,
                                                 device=device,
                                                 dtype=dtype)
        
        self.decoder_layer = nn.Linear(in_features=hidden_size,
                                       out_features=vocab_size,
                                       device=device,
                                       dtype=dtype)
    
    def forward(self, x):
        """x: [bs, seq_len], each element is token id within vocab_size"""
        x = self.encoding_layer(x.transpose(axes=(0, 1)))    # [seq_len, bs, hidden_size]
        x = self.transformer_layers(x.transpose(axes=(0, 1)))    # [bs, seq_len, hidden_size]
        x = self.decoder_layer(x)   # [bs, seq_len, vocab_size]
        return x

    def generate(self, input_ids: Tensor, max_new_tokens: int):
        """input_ids: Tensor [bs, seq_len]
        returns: Tensor [bs, seq_len + max_new_tokens]
        """
        self.eval()
        bs, seq_len = input_ids.shape
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self(generated)  # [bs, cur_len, vocab_size]
            # BUG: get item has bug: when set an integer, it still keeps the dim
            next_token_logits = logits[:, -1, :].reshape((bs, self.vocab_size))  # [bs, vocab_size]
            next_tokens_numpy = np.argmax(next_token_logits.numpy(), axis=-1)  # [bs,]
            generated = ndl.Tensor(
                np.concatenate(
                    [generated.numpy(), next_tokens_numpy[:, None]], axis=1
                ),
                device=self.device,
                dtype="int32"
            )
        return generated

    def save(self, path):
        """convert to numpy and save model parameters to path"""
        params = self.parameters()
        np_params = [param.data.numpy() for param in params]
        np.savez(path, *np_params)

    def load(self, path):
        """load model parameters from path and convert to Tensor"""

        np_data = np.load(path)
        params = self.parameters()
        assert len(np_data.files) == len(params), f"mismatch in number of parameters: {len(np_data.files)} vs {len(params)}"

        for param, key in zip(params, np_data.files):
            np_param_array = np_data[key]
            param.data = ndl.Tensor(np_param_array, device=self.device, dtype=self.dtype)

        print(f"Loaded model parameters from {path}")

def train_step(model: QwenLM, input_ids: Tensor):
    """input_ids: Tensor [bs, seq_len]
    }"""
    bs, seq_len = input_ids.shape
    vocab_size = model.vocab_size
    logits = model(input_ids)[:, :-1, :].reshape((bs*(seq_len-1), vocab_size))  # [bs*(seq_len-1), vocab_size]
    labels = input_ids[:, 1:].reshape((bs*(seq_len-1),))  # [bs*(seq_len-1)]
    loss = nn.SoftmaxLoss()(logits, labels)
    return loss

def train_qwen(model: QwenLM, data_loader: DataLoader, optimizer: ndl.optim.Optimizer, epochs: int):
    log_gpu_memory("Start of training")
    for epoch in range(epochs):
        avg_loss = 0.0
        data_loader.restart()
        log_gpu_memory(f"Start of epoch {epoch}")
        for input_ids in tqdm(data_loader):
            # log_gpu_memory(f"Before train step epoch {epoch}")
            optimizer.reset_grad()
            # log_gpu_memory(f"After reset_grad epoch {epoch}")
            loss = train_step(model, input_ids)
            # log_gpu_memory(f"After forward epoch {epoch}")
            loss.backward()
            # log_gpu_memory(f"After backward epoch {epoch}")
            optimizer.step()
            # log_gpu_memory(f"After optimizer step epoch {epoch}")
            avg_loss += loss.detach().numpy()
            del loss
            gc.collect()
            # log_gpu_memory(f"End of batch epoch {epoch}")
        avg_loss /= len(data_loader)
        print(f"Epoch {epoch}: Loss = {avg_loss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    vocab_size = tokenizer.vocab_size
    
    # Dataset
    dataset = Dataset(doc_path=args.doc_path, seq_len=args.seq_len)
    data_loader = DataLoader(dataset, batch_size=args.batch_size)

    # Model
    model = QwenLM(
        vocab_size=vocab_size,
        hidden_size=1024,
        num_layers=8,
        num_attention_heads=16, 
        device=ndl.cuda(),
        dtype="float32"
    )
    model.train()

    # Optimizer
    optimizer = ndl.optim.Adam(model.parameters(), lr=args.lr)
    print(f"training QwenLM with {sum(p.data.size for p in model.parameters())} parameters")

    # Train
    train_qwen(model, data_loader, optimizer, epochs=args.epochs)

    # Save model
    model.save(args.save_path)
