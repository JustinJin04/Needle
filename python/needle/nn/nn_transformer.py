from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential,
    Residual
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        out: (bs, num_head, seq_len, dim_head), attention: (bs, num_head, seq_len, seq_len)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        if self.causal:
            mask = self.create_causal_mask(queries_len, keys_values_len, device=q.device)
            attention = self.dropout(self.softmax(self.matmul(q, k) / (q_dim**0.5) + mask.broadcast_to((batch_size, num_head, queries_len, keys_values_len))))
        else:
            attention = self.dropout(self.softmax(self.matmul(q, k) / (q_dim**0.5)))
        
        result = self.matmul(attention, v.transpose())
        probs = attention
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):
        """
        q_features: q.shape = (bs, q_len, q_features)
        out_features: output.shape = (bs, q_len, out_features)
        """

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)
    
    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        q_proj = self.q_projection(self.prenorm_q(q).reshape((batch_size*queries_len, q_dim)))
        k_proj = self.k_projection(self.prenorm_k(k).reshape((batch_size*keys_values_len, k_dim)))
        v_proj = self.v_projection(self.prenorm_v(v).reshape((batch_size*keys_values_len, v_dim)))
        
        q_input = q_proj.reshape((batch_size, queries_len, self.num_head, self.dim_head)).transpose((1,2))
        k_input = k_proj.reshape((batch_size, keys_values_len, self.num_head, self.dim_head)).transpose((1,2))
        v_input = v_proj.reshape((batch_size, keys_values_len, self.num_head, self.dim_head)).transpose((1,2))

        out, attention = self.attn(q_input, k_input, v_input)
        out = out.transpose((1,2)).reshape((batch_size*queries_len, self.num_head * self.dim_head))
        result = self.out_projection(out).reshape((batch_size, queries_len, self.out_features))
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.attention =AttentionLayer(q_features=q_features, 
                                       num_head=num_head, 
                                       dim_head=dim_head, 
                                       dropout=dropout, 
                                       causal=causal,
                                       device=device, 
                                       dtype=dtype)
        self.dropout = Dropout(p=dropout)
        self.residual1 = Residual(Sequential(self.attention, self.dropout))

        self.layernorm = LayerNorm1d(dim=q_features, device=device, dtype=dtype)
        self.Linear1 = Linear(in_features=q_features, out_features=hidden_size, device=device, dtype=dtype)
        self.relu = ReLU()
        self.Linear2 = Linear(in_features=hidden_size, out_features=q_features, device=device, dtype=dtype)
        self.residual2 = Residual(Sequential(self.layernorm, self.Linear1, self.relu, self.dropout, self.Linear2, self.dropout))

        self.transformerlayer = Sequential(self.residual1, self.residual2)
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        x = self.transformerlayer(x)
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):
        """
        batch_first: true if input shape is (bs, seq_len, embedding_size) else (seq_len, bs, embedding_size)
        sequence_len: bptt window size from batchify function
        """

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        # assert embedding_size == num_head * dim_head
        self.encoding_Layer = Embedding(num_embeddings=sequence_len,
                                    embedding_dim=embedding_size,
                                    device=device,
                                    dtype=dtype)

        self.transformer_layers = Sequential(*[TransformerLayer(q_features=embedding_size,
                                               num_head=num_head,
                                               dim_head=dim_head,
                                               hidden_size=hidden_size,
                                               dropout=dropout,
                                               causal=causal,
                                               device=device,
                                               dtype=dtype) for _ in range(num_layers)])
        

        
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):
        """
        x: (bs, seq, embedding_size) or (seq, bs, embedding_size)
        """

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        bs, seq_len = x.shape[:2]
        positional_encoder = Tensor(np.broadcast_to(np.arange(seq_len), (bs, seq_len)).transpose(), device=x.device, dtype=x.dtype, requires_grad=False)
        x += self.encoding_Layer(positional_encoder).transpose(axes=(0, 1))
        x = self.transformer_layers(x)
        

        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)