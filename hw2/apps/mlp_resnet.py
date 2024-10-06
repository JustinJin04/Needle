import sys
from needle.data import MNISTDataset, DataLoader
sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    Linear_1 = nn.Linear(dim,hidden_dim)
    Norm_1 = norm(hidden_dim)
    ReLU = nn.ReLU()
    Dropout = nn.Dropout(drop_prob)
    Linear_2 = nn.Linear(hidden_dim,dim)
    Norm_2 = norm(dim)
    
    return nn.Sequential(nn.Residual(nn.Sequential(Linear_1,Norm_1,ReLU,Dropout,Linear_2,Norm_2)),nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    model = []
    model.append(nn.Linear(dim,hidden_dim))
    model.append(nn.ReLU())
    for i in range(num_blocks):
        model.append(ResidualBlock(dim=hidden_dim,hidden_dim=hidden_dim//2,norm=norm,drop_prob=drop_prob))
    model.append(nn.Linear(hidden_dim,num_classes))
    return nn.Sequential(*model)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    
    total_error = 0
    total_loss = 0
    num_samples = len(dataloader.dataset)
    num_steps = 0
    
    loss_fn = nn.SoftmaxLoss()

    if opt is None:
        model.eval()
    else:
        model.train()
    for batch_x, batch_y in dataloader:
        num_steps += 1
        
        ## forward
        # print(batch_x.shape)
        y_pred = model(batch_x)
        loss = loss_fn(y_pred,batch_y)
        error = np.sum(np.argmax((y_pred.numpy()),axis=1)!=(batch_y.numpy()))

        ## training
        if opt is not None:
            
            ## backward
            opt.reset_grad()
            loss.backward()

            ## update
            opt.step()

        total_error += error
        total_loss += loss.numpy()
    
    return total_error/num_samples, total_loss/num_steps

    ### END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION

    ## load data
    train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz", 
        f"{data_dir}/train-labels-idx1-ubyte.gz"
    )
    train_dataloader = ndl.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle = True)
    test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz",
        f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    test_loader = ndl.data.DataLoader(test_dataset, batch_size=batch_size)

    ## initialize model
    resnet = MLPResNet(28*28,hidden_dim=hidden_dim)

    ## initialize optimizer
    opt = optimizer(resnet.parameters(),lr=lr,weight_decay=weight_decay)

    ## training!
    for i in range(epochs):
        train_err, train_loss = epoch(train_dataloader,resnet,opt)
        # print(f"train_err: %f, train_loss: %f",train_err,train_loss)
    
    ## evaluating
    test_err, test_loss = epoch(test_loader,resnet)
    print(train_err, train_loss, test_err, test_loss)
    return train_err, train_loss, test_err, test_loss

    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
