"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

from ..backend_ndarray.ndarray import prod

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return self.scalar * out_grad * (a ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b = node.inputs
        return out_grad / b, -out_grad * a / (b**2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # return a/self.scalar
        return (a/self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


# class Transpose(TensorOp):
#     def __init__(self, axes: Optional[tuple] = None):
#         self.axes = axes

#     def compute(self, a):
#         ### BEGIN YOUR SOLUTION
#         if(self.axes is None):
#             return array_api.swapaxes(a,-2,-1)
#         else:
#             return array_api.swapaxes(a,self.axes[0],self.axes[1])
#         ### END YOUR SOLUTION

#     def gradient(self, out_grad, node):
#         ### BEGIN YOUR SOLUTION
#         return transpose(out_grad,self.axes)
#         ### END YOUR SOLUTION

class Transpose(TensorOp): 
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    # def compute(self, a):
    #     return array_api.transpose(a, self.axes)
    def compute(self, a):
        axes = list(range(a.ndim))
        if self.axes is None:
            self.axes = axes[-2:]  
        axes[self.axes[0]], axes[self.axes[1]] = axes[self.axes[1]], axes[self.axes[0]]

        return array_api.transpose(a, axes)
        
    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)

        

class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # TODO: COMPACT 设计????
        return array_api.reshape(a.compact(),self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad,node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape=node.inputs[0].shape
        out_shape=out_grad.shape
        len_bias=len(out_shape)-len(in_shape)
        broadcast_axes=()
        for i in range(len_bias):
            broadcast_axes+=(i,)
        out_shape=out_shape[len_bias:]
        for i in range(len(in_shape)):
            if(out_shape[i]>in_shape[i]):
                broadcast_axes+=(i+len_bias,)
        # print(in_shape,out_grad.shape,broadcast_axes)
        return reshape(summation(out_grad,broadcast_axes),in_shape)

        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # if self.axes is not None:
            # assert len(self.axes) == 1
        return array_api.sum(a,axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]        
        expand_shape = list(a.shape)
        if self.axes is None:
            return broadcast_to(out_grad,a.shape)
        if isinstance(self.axes, int):
            axes_tuple = (self.axes,)
        elif isinstance(self.axes, (tuple, list)):
            axes_tuple = self.axes
        else:
            assert 0 ,"axes must be int or list or tuple"
        for i in axes_tuple:
            expand_shape[i] = 1
        expand_shape = tuple(expand_shape)
        return broadcast_to(reshape(out_grad,expand_shape),a.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # return array_api.matmul(a,b)
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b=node.inputs
        a_ndim,b_ndim=len(a.shape),len(b.shape)
        lgrad,rgrad = matmul(out_grad,transpose(b)), matmul(transpose(a),out_grad)
        if(b_ndim<a_ndim):
            for i in range(a_ndim-b_ndim):
                rgrad = summation(rgrad,0)
        elif(a_ndim<b_ndim):
            for i in range(b_ndim-a_ndim):
                lgrad = summation(lgrad,0)
        return lgrad,rgrad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # return array_api.negative(a)
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a=node.inputs[0]
        return out_grad/a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a,array_api.zeros(a.shape, device = a.device))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        data = node.inputs[0].realize_cached_data()
        mask = data >= 0
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # TODO: 修改tensor __rsub__实现方式
        return out_grad * (1 - tanh(node.inputs[0])**2)

        ### END YOUR SOLUTION

def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = list(args[0].shape)
        shape.insert(self.axis, 1)
        shape[self.axis] = len(args)
        output = array_api.empty(shape, dtype = args[0].dtype,device = args[0].device)
        stack_slice_list = [slice(0, s, 1) for s in shape]
        for i, tensor in enumerate(args):
            stack_slice_list[self.axis] = i
            output[tuple(stack_slice_list)] = tensor
        return output
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))



class Cat(TensorOp):
    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        self.cat_size = args[0].shape[self.axis]
        shape = list(args[0].shape)
        shape[self.axis] *= len(args)
        output = array_api.empty(shape, dtype = args[0].dtype, device = args[0].device)
        stack_slice_list = [slice(None, None, 1) for _ in range(len(shape))]
        for i, tensor in enumerate(args):
            stack_slice_list[self.axis] = slice(i * self.cat_size, (i + 1) * self.cat_size, 1)
            output[tuple(stack_slice_list)] = tensor
        return output
    
    def gradient(self, out_grad, node):
        return split(out_grad, self.axis, split_size=self.cat_size)
    
def cat(args, axis):
    return Cat(axis)(make_tuple(*args))



class Split(TensorTupleOp):
    def __init__(self, axis: int, split_size=1, keepdims=False):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split

        update: if split_size=1 && keepdims=False, than the ndims will drop by 1, otherwise wont't drop
        """
        self.axis = axis
        self.split_size = split_size
        self.keepdims = keepdims

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        assert A.shape[self.axis] % self.split_size == 0, "must be divisible"
        split_slice_list = [slice(None, None, 1) for _ in range(len(A.shape))]
        ret = []
        shape = list(A.shape)
        if self.split_size == 1 and self.keepdims == False:
            shape.pop(self.axis)
        else:
            shape[self.axis] = self.split_size
        
        for i in range(0, A.shape[self.axis], self.split_size):
            split_slice_list[self.axis] = slice(i, i+self.split_size, 1)
            ret.append(A[tuple(split_slice_list)].compact().reshape(shape))
        return tuple(ret)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.split_size == 1 and self.keepdims == False:
            return stack(out_grad, self.axis)
        else:
            return cat(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis, split_size=1, keepdims=False):
    return Split(axis, split_size=split_size, keepdims=keepdims)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        
        return a.flip(self.axes)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        slice_list = [slice(None) for _ in a.shape]
        
        for axis in self.axes:
            new_shape[axis] *= (self.dilation + 1)
            slice_list[axis] = slice(0, None, self.dilation + 1)
        out = array_api.full(tuple(new_shape), 0, dtype=a.dtype, device=a.device)
        out[tuple(slice_list)] = a

        return out

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slice_list = [slice(None) for _ in a.shape]
        for axis in self.axes:
            slice_list[axis] = slice(0, None, self.dilation + 1)
        return a[tuple(slice_list)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)

# TODO: Implement Conv by myself
# class Conv(TensorOp):
#     def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
#         self.stride = stride
#         self.padding = padding

#     def compute(self, A, B):
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION

#     def gradient(self, out_grad, node):
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION



class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A = A.pad(((0,0), (self.padding, self.padding), (self.padding, self.padding),(0,0)))
        N, H, W, C_in = A.shape
        K, K_, C_in_, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        assert K == K_, "Conv kernel should be a square tensor"
        assert C_in == C_in_, "Conv kernel and input are not compatible"
        
        inner_dim = K * K * C_in
        out_H, out_W = (H-K+1)//self.stride, (W-K+1)//self.stride
        im2col = A.as_strided(shape=(N, out_H, out_W, K, K, C_in),
                              strides=(Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, Cs))\
                              .compact()\
                              .reshape((N*out_H*out_W, inner_dim))
        out = im2col @ B.compact().reshape((K*K_*C_in_, C_out))
        return out.compact().reshape((N, out_H, out_W, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        K, _, _, _ = W.shape

        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride - 1)
        W_permute = transpose(flip(W, (0, 1)), (2, 3)) # K * K * C_out * C_in
        # out_grad: # N * (H+2P-K+1) * (W+2P-K+1) * C_out
        X_grad = conv(out_grad, W_permute, padding=K-1-self.padding)

        X_permute = transpose(X, (0, 3)) # C_in * H * W * N
        grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2)) # (H+2P-K+1) * (W+2P-K+1) * N * C_out
        W_grad = conv(X_permute, grad_permute, padding=self.padding) # C_in * H * W * C_out
        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2)) # H * W * C_in * C_out

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
