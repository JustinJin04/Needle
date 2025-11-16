"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

import needle as ndl

import needle.nn as nn
from apps.models import *
import time

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname,"rb") as imag_file:
      magic_num,num_images,num_rows,num_cols=struct.unpack('>4i',imag_file.read(16))
      assert(magic_num==2051)
      tot_pixels=num_rows*num_cols
      X=np.array(struct.unpack(f"{num_images*tot_pixels}B",imag_file.read(num_images*tot_pixels)),dtype=np.float32)
      X=X.reshape(num_images,tot_pixels)
      X-=np.min(X)
      X/=np.max(X)
      
    with gzip.open(label_filename,"rb") as label_file:
      magic_num,num_labels=struct.unpack('>2i',label_file.read(8))
      assert(magic_num==2049)
      y=np.array(struct.unpack(f"{num_labels}B",label_file.read(num_labels)),dtype=np.uint8)
  
    return (X,y)
      
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # answer = (ndl.log(ndl.exp(Z).sum((1,))).sum() - (y_one_hot * Z).sum()) / Z.shape[0]


    batch_size = Z.shape[0]
    exp_Z = ndl.exp(Z)
    softmax_prob = exp_Z/ndl.broadcast_to(ndl.reshape(ndl.summation(exp_Z,axes=1),(batch_size,1)),(batch_size,Z.shape[1]))
    loss = ndl.summation(ndl.negate(y_one_hot*ndl.log(softmax_prob)))/batch_size
    return loss
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples=X.shape[0] 
    num_classes=W2.shape[1] 
    y_one_hot=np.eye(num_classes)[y]
    for i in range(0,num_examples,batch):
       X_batch = ndl.Tensor(X[i:i+batch])
       y_batch = ndl.Tensor(y_one_hot[i:i+batch])
       
       # forward
       loss = softmax_loss(ndl.relu(X_batch @ W1) @ W2,y_batch)

       # backward (calculate gradients)
       loss.backward()

       # update parameters
       W1-=lr*W1.grad.realize_cached_data()
       W2-=lr*W2.grad.realize_cached_data()
    
    return (W1,W2)

    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss, device=ndl.cpu(), dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.train()
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(n_epochs):
        
        correct, total_loss = 0, 0
        nsteps = 0
        for batch in dataloader:
            nsteps += 1
            opt.reset_grad()
            X, y = batch
            X, y = ndl.Tensor(X, device=device, dtype=dtype), ndl.Tensor(y, device=device, dtype=dtype)
            out = model(X)
            correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
            loss = loss_fn()(out, y)
            total_loss += loss.data.numpy()
            loss.backward()
            opt.step()
            print(f"train: acc={np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())}/{y.shape[0]}, loss={loss.numpy()}")
        
        avg_acc = correct / len(dataloader.dataset)
        avg_loss = total_loss / nsteps

        print(f"train: avg_acc={avg_acc}, avg_loss={avg_loss}")
    
    return avg_acc, avg_loss

    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss, device=ndl.cpu(), dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.eval()
    correct, total_loss = 0, 0
    nsteps = 0
    for batch in dataloader:
        nsteps += 1
        X, y = batch
        X, y = ndl.Tensor(X, device=device, dtype=dtype), ndl.Tensor(y, device=device, dtype=dtype)
        out = model(X)
        loss = loss_fn()(out, y)
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        total_loss += loss.data.numpy()
        print(f"eval: acc={np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())}/{y.shape[0]}, loss={loss.numpy()}")
    avg_acc = correct / len(dataloader.dataset)
    avg_loss = total_loss / nsteps
    print(f"eval: avg_acc={avg_acc}, avg_loss={avg_loss}")

    return avg_acc, avg_loss
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()
    avg_acc, avg_loss = 0, 0
    nbatch = data.shape[0]
    for i in range (nbatch - 1):
        if opt is not None:
            opt.reset_grad()
        
        input_data, target = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        out, h = model(input_data)
        acc, loss = correct_loss(out, target)
        avg_acc += acc
        avg_loss += loss

        if opt is not None:
            opt.step()
            

    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=ndl.cpu(), dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.train()
    opt = optimizer(model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
    )
    nbatch, bs = data.shape
    assert nbatch > 1
    
    for _ in range(n_epochs):
        avg_acc, avg_loss = 0, 0
        for i in range(nbatch - 1):
            opt.reset_grad()
            input_data, target = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
            out, h = model(input_data)
            loss = loss_fn()(out, target)
            loss.backward()
            opt.step()

            avg_acc += np.sum(np.argmax(out.numpy(), axis=1) == target.numpy())
            avg_loss += loss.numpy()
            
            print(f"train: acc={np.sum(np.argmax(out.numpy(), axis=1) == target.numpy())}/{target.shape[0]}, loss={loss.numpy()}")
        
        # softmax loss function's output is total_loss / num_samples
        avg_acc /= (nbatch - 1) * bs * seq_len
        avg_loss /= (nbatch - 1)
        print(f"train: avg_acc={avg_acc}, avg_loss={avg_loss}")
    
    return avg_acc, avg_loss
            
            

    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=ndl.cpu(), dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.eval()
    avg_acc, avg_loss = 0, 0
    nbatch, bs = data.shape
    for i in range(nbatch - 1):
        input_data, target = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        out, h = model(input_data)
        loss = loss_fn()(out, target)
        
        avg_acc += np.sum(np.argmax(out.numpy(), axis=1) == target.numpy())
        avg_loss += loss.numpy()

        print(f"eval: acc={np.sum(np.argmax(out.numpy(), axis=1) == target.numpy())}/{target.shape[0]}, loss={loss.numpy()}")
    
    avg_acc /= (nbatch - 1) * bs * seq_len
    avg_loss /= (nbatch - 1)

    print(f"eval: avg_acc={avg_acc}, avg_loss={avg_loss}")
    
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """
    Helper function to compute both loss and error
        
    Args:
        h: (seq_len*bs, output_size)
        y: (seq_len*bs,)
    """
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)

def correct_loss(h, y):
    """
    Helper function to compute both total correctness and total loss
        
    Args:
        h: (seq_len*bs, output_size)
        y: (seq_len*bs,)
    """
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return np.sum(h.numpy().argmax(axis=1) == y), softmax_loss(h, y_).numpy()
