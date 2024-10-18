"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        """
        supply input x.ndim > 2 by reshape before and after
        input x's last dim must equal in_features
        """
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(self.in_features,self.out_features,device=device,dtype=dtype,requires_grad=True))
        if bias == True:
            self.bias = Parameter(ops.reshape(init.kaiming_uniform(self.out_features,1,device=device,dtype=dtype,requires_grad=True),(1,self.out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        """
        X: (*, in_features)
        output: (*, out_features)
        """
        assert X.shape[-1] == self.in_features
        input_shape = X.shape

        if len(input_shape) > 2:
            compact_shape = (math.prod(input_shape) // input_shape[-1], input_shape[-1])
            X = X.reshape(compact_shape)

        if self.bias is None:
            output = ops.matmul(X,self.weight)
        else:
            shape=X.shape[:-1]+(self.out_features,)
            output = ops.matmul(X,self.weight) + ops.broadcast_to(self.bias,shape)
        
        if len(input_shape) > 2:
            output_shape = input_shape[:-1] + (self.out_features,)
            output = output.reshape(output_shape)
        
        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor):
        ### BEGIN YOUR SOLUTION
        size=1
        for i in X.shape[1:]:
            size*=i
        return X.reshape((X.shape[0],size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ret = x
        for module in self.modules:
            ret=module(ret)
        return ret
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        logsoftmax_logits=logits-ops.logsumexp(logits,axes=1).reshape((logits.shape[0],1)).broadcast_to(logits.shape)
        y_one_hot=init.one_hot(logits.shape[1],y, device=logits.device)
        # return Tensor((-logsoftmax_logits*y_one_hot).sum()/logits.shape[0],dtype="float32")
        ret = (-logsoftmax_logits*y_one_hot).sum()/logits.shape[0]
        return ret
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.constant(self.dim,c=1,device=device,dtype=dtype,requires_grad=True))
        self.bias=Parameter(init.constant(self.dim,c=0,device=device,dtype=dtype),requires_grad=True)
        self.running_mean=init.constant(self.dim,c=0,device=device,dtype=dtype)
        self.running_var=init.constant(self.dim,c=1,device=device,dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training is True:
            Ex = x.sum(axes=0).broadcast_to(x.shape)/x.shape[0]
            Varx=((x-Ex)**2).sum(axes=0).broadcast_to(x.shape)/x.shape[0]
            self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*(x.sum(axes=0)/x.shape[0]).data
            self.running_var=(1-self.momentum)*self.running_var+self.momentum*(((x-Ex)**2).sum(axes=0)/x.shape[0]).data
            norm = (x-Ex)/((Varx+self.eps)**0.5)
            return self.weight.broadcast_to(x.shape)*norm + self.bias.broadcast_to(x.shape)
        else:
            return self.weight.broadcast_to(x.shape)*(x-self.running_mean.broadcast_to(x.shape))/((self.running_var.broadcast_to(x.shape)+self.eps)**0.5)+self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.constant(self.dim,c=1,device=device,dtype=dtype,requires_grad=True))
        self.bias=Parameter(init.constant(self.dim,c=0,device=device,dtype=dtype,requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        TODO: BUG?????
        x: (batch_size, seq_len, dim)
        weight: (dim,)
        """
        ### BEGIN YOUR SOLUTION
        sum_axis = len(x.shape) - 1
        Ex=x.sum(axes=sum_axis).reshape((*x.shape[:-1],1)).broadcast_to(x.shape)/x.shape[sum_axis]
        Varx=((x-Ex)*(x-Ex)).sum(axes=sum_axis).reshape((*x.shape[:-1],1)).broadcast_to(x.shape)/x.shape[sum_axis]
        return self.weight.broadcast_to(x.shape)*(x-Ex)/((Varx+self.eps)**0.5)+self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training is True:
            ## bugging: probability of dropout is p, whereas probabiliy of one is 1-p!!!!!!!!!!!!!!!
            mask=init.randb(*x.shape,p=1-self.p, device=x.device, dtype=x.dtype, requires_grad=False)
            return x * mask / (1-self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION