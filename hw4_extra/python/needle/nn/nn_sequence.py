"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
from .nn_basic import ReLU, Tanh


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x)) ** (-1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = 1 / hidden_size
        self.W_ih = Parameter(init.rand(input_size, hidden_size, 
                                              low=-(k**0.5), high=k**0.5, 
                                              device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, 
                                              low=-(k**0.5), high=k**0.5, 
                                              device=device, dtype=dtype, requires_grad=True))
        self.bias_ih = None
        self.bias_hh = None
        if bias is True:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-(k**0.5), high=(k**0.5),
                                               device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-(k**0.5), high=(k**0.5),
                                               device=device, dtype=dtype, requires_grad=True))
            
        if nonlinearity == 'tanh':
            self.nonlinearity = Tanh()
        elif nonlinearity == 'relu':
            self.nonlinearity = ReLU()
        else:
            raise ValueError
        ### END YOUR SOLUTION

    def forward(self, X: Tensor, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device = X.device, dtype=X.dtype, requires_grad=False)
        if self.bias_ih is not None:
            return self.nonlinearity(X @ self.W_ih + self.bias_ih.broadcast_to((X.shape[0], h.shape[1])) + h @ self.W_hh + self.bias_hh.broadcast_to((X.shape[0], h.shape[1])))
        else:
            return self.nonlinearity(X @ self.W_ih + h @ self.W_hh)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_cells = []
        for layer in range(num_layers):
            cell = RNNCell(input_size=input_size, 
                           hidden_size=hidden_size,
                           bias=bias,
                           nonlinearity=nonlinearity,
                           device=device,
                           dtype=dtype) if layer == 0 else\
                    RNNCell(input_size=hidden_size, 
                           hidden_size=hidden_size,
                           bias=bias,
                           nonlinearity=nonlinearity,
                           device=device,
                           dtype=dtype)
            self.rnn_cells.append(cell)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h0 is None:
            h0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=False)
        output = []
        input_list = list(ops.split(X, axis=0))
        h0_list = list(ops.split(h0, axis=0))
        
        for word in input_list:
            out = word
            for i in range(self.num_layers):
                out = self.rnn_cells[i](out, h0_list[i])
                h0_list[i] = out
            output.append(out)
        
        output = ops.stack(tuple(output), axis=0)
        hn = ops.stack(tuple(h0_list), axis=0)

        return output, hn

        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
       
        rand_low = - (1/hidden_size)**0.5
        rand_high = (1/hidden_size)**0.5
        self.W_ih = Parameter(init.rand(input_size, 4*hidden_size, 
                                        low=rand_low, high=rand_high,
                                        device=device, dtype=dtype,requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size,
                                        low=rand_low, high=rand_high,
                                        device=device, dtype=dtype,requires_grad=True))
        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size,
                                               low=rand_low, high=rand_high,
                                               device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(4*hidden_size,
                                               low=rand_low, high=rand_high,
                                               device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION


    def forward(self, X: Tensor, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = (init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype,requires_grad=False), 
                 init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype,requires_grad=False))
        else:
            assert h[0].shape == (X.shape[0], self.hidden_size)
            
        W_ii, W_if, W_ig, W_io = ops.split(self.W_ih, axis=1, split_size=self.hidden_size, keepdims=True)
        W_hi, W_hf, W_hg, W_ho = ops.split(self.W_hh, axis=1, split_size=self.hidden_size, keepdims=True)
        if self.bias_hh is not None:
            b_ii, b_if, b_ig, b_io = ops.split(self.bias_ih, axis=0, split_size=self.hidden_size, keepdims=True)
            b_hi, b_hf, b_hg, b_ho = ops.split(self.bias_hh, axis=0, split_size=self.hidden_size, keepdims=True)

            # print(f"X shape: {X.shape}, W_ii shape: {W_ii.shape}, b_ii shape: {b_ii.shape}, h[0] shape: {h[0].shape}, W_hi shape: {W_hi.shape}\
            #       x.shape[0]: {X.shape[0]}")
            input_gate = Sigmoid()(X @ W_ii + b_ii.broadcast_to((X.shape[0], self.hidden_size)) + 
                                   h[0] @ W_hi + b_hi.broadcast_to((X.shape[0], self.hidden_size)))
            forget_gate = Sigmoid()(X @ W_if + b_if.broadcast_to((X.shape[0], self.hidden_size)) + 
                                   h[0] @ W_hf + b_hf.broadcast_to((X.shape[0], self.hidden_size)))
            cell_gate = Tanh()(X @ W_ig + b_ig.broadcast_to((X.shape[0], self.hidden_size)) + 
                                   h[0] @ W_hg + b_hg.broadcast_to((X.shape[0], self.hidden_size)))
            output_gate = Sigmoid()(X @ W_io + b_io.broadcast_to((X.shape[0], self.hidden_size)) + 
                                   h[0] @ W_ho + b_ho.broadcast_to((X.shape[0], self.hidden_size)))
        
        else:
            input_gate = Sigmoid()(X @ W_ii + h[0] @ W_hi)
            forget_gate = Sigmoid()(X @ W_if + h[0] @ W_hf)
            cell_gate = Tanh()(X @ W_ig + h[0] @ W_hg)
            output_gate = Sigmoid()(X @ W_io + h[0] @ W_ho)
        
        c_out = forget_gate * h[1] + input_gate * cell_gate
        h_out = output_gate * Tanh()(c_out)

        assert h_out.shape == (X.shape[0], self.hidden_size)
        return h_out, c_out
        
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm_cells = []
        for i in range(num_layers):
            self.lstm_cells.append(LSTMCell(input_size=input_size, 
                                           hidden_size=hidden_size, 
                                           bias=bias, 
                                           device=device, 
                                           dtype=dtype) if i == 0 else\
                                  LSTMCell(input_size=hidden_size, 
                                           hidden_size=hidden_size, 
                                           bias=bias, 
                                           device=device, 
                                           dtype=dtype))
        
        ### END YOUR SOLUTION

    def forward(self, X: Tensor, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        X_list = list(ops.split(X, axis=0))
        if h is None:
            h0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, 
                            device=X.device, 
                            dtype=X.dtype, 
                            requires_grad=False)
            c0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, 
                            device=X.device, 
                            dtype=X.dtype, 
                            requires_grad=False)
        else:
            h0, c0 = h
        h_list = list(ops.split(h0, axis=0))
        c_list = list(ops.split(c0, axis=0))

        out_list = []

        for word in X_list:
            for i in range(self.num_layers):
                # print(i)
                # print(word.shape)
                # print(h_list[i].shape)
                ht, ct = self.lstm_cells[i](word, (h_list[i], c_list[i]))
                h_list[i], c_list[i] = ht, ct
                word = ht
            out_list.append(word)
        
        output = ops.stack(tuple(out_list), axis=0)
        hn = ops.stack(tuple(h_list), axis=0)
        cn = ops.stack(tuple(c_list), axis=0)

        return output, (hn, cn)
        

        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = init.randn(num_embeddings, embedding_dim, 
                                 mean=0, 
                                 std=1, 
                                 device=device, 
                                 dtype=dtype,
                                 requires_grad=True)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        x_one_hot = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype, requires_grad=False)
        seq_len, bs = x.shape
        return ((x_one_hot.reshape((seq_len * bs, self.num_embeddings))) @ self.weight).reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION