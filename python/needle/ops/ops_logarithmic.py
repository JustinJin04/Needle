from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z:NDArray):
        ### BEGIN YOUR SOLUTION
        maxZ_keepdim=Z.max(axis=self.axes,keepdims=True)
        maxZ_sameshape = maxZ_keepdim.broadcast_to(Z.shape)
        maxZ=Z.max(axis=self.axes)
        return array_api.log(array_api.exp(Z-maxZ_sameshape).sum(axis=self.axes))+maxZ
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a=node.inputs[0]
        a_data=a.realize_cached_data()
        maxa=a_data.max(axis=self.axes,keepdims=True).broadcast_to(a_data.shape)
        exp_a=array_api.exp(a_data-maxa)
        partial_grad=exp_a/(array_api.sum(exp_a,axis=self.axes,keepdims=True).broadcast_to(exp_a.shape))
        
        expand_shape = list(a.shape)
        if self.axes is None:
            return out_grad.broadcast_to(a.shape)*partial_grad
        elif isinstance(self.axes,int):
            axes_shape = (self.axes,)
        else:
            axes_shape = self.axes
        for i in axes_shape:
            expand_shape[i]=1
        expand_shape=tuple(expand_shape)
        return out_grad.reshape(expand_shape).broadcast_to(a.shape)*partial_grad

        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

