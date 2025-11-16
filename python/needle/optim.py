"""Optimization module"""
import needle as ndl
import numpy as np
from collections import defaultdict


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = defaultdict(float)
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for para in self.params:
            self.u[para]=self.momentum * self.u[para] + (1-self.momentum) * ndl.Tensor(para.grad.data + self.weight_decay * para.data,device=para.device,dtype="float32",requires_grad=False)
            para.data = para.data - self.lr * self.u[para]


        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = defaultdict(float)
        self.v = defaultdict(float)

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t+=1
        
        for w in self.params:
            grad = w.grad.data + self.weight_decay * w.data
            # print(f"Updating parameter with grad: {grad}")
            self.m[w] = self.beta1 * self.m[w] + (1-self.beta1) * grad
            self.v[w] = self.beta2 * self.v[w] + (1-self.beta2) * (grad**2)
            unbiased_m = self.m[w]/(1-self.beta1**self.t)
            unbiased_v = self.v[w]/(1-self.beta2**self.t)
            w.data = ndl.Tensor(w.data - self.lr * unbiased_m / (unbiased_v ** 0.5 + self.eps),device=w.device,dtype="float32")
            
        ### END YOUR SOLUTION
