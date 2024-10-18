import os
os.environ["PYTHONPATH"] = "./python"
os.environ["NEEDLE_BACKEND"] = "nd"
import sys
sys.path.append('./python')
sys.path.append('./apps')
import needle as ndl
from models import ResNet9
from simple_ml import train_cifar10, evaluate_cifar10

device = ndl.cuda()
dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
dataloader = ndl.data.DataLoader(\
         dataset=dataset,
         batch_size=128,
         shuffle=True,)
model = ResNet9(device=device, dtype="float32")
train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
      lr=0.001, weight_decay=0.001, device=device, dtype="float32")
evaluate_cifar10(model, dataloader)