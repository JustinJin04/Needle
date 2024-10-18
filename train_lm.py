import os
os.environ["PYTHONPATH"] = "./python"
os.environ["NEEDLE_BACKEND"] = "nd"
import sys
sys.path.append('./python')

import needle as ndl
sys.path.append('./apps')
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb

batch_size = 128
seq_len = 10
embedding_size = 20
hidden_size = 32

device = ndl.cuda()
dtype="float32"

corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, 
                               batch_size=batch_size, 
                               device=device, 
                               dtype=dtype)
model = LanguageModel(embedding_size, 
                      len(corpus.dictionary), 
                      hidden_size=hidden_size, 
                      num_layers=2, 
                      seq_model='rnn', 
                      seq_len=seq_len, 
                      device=device,
                      dtype=dtype)
train_ptb(model, 
          train_data, 
          seq_len=seq_len, 
          n_epochs=1, 
          device=device, 
          lr=0.003, 
          optimizer=ndl.optim.Adam)

evaluate_ptb(model, 
             train_data, 
             seq_len=seq_len, 
             device=device)