"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
from .nn_basic import BatchNorm2d, ReLU


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        # TODO: WHY self.padding is this??????
        self.padding = (kernel_size - 1)//2
        fan_in = self.kernel_size * self.kernel_size * self.in_channels
        fan_out = self.out_channels
        self.weight = Parameter(init.kaiming_uniform(fan_in=fan_in, 
                                                     fan_out=fan_out, 
                                                     shape=(kernel_size, kernel_size, in_channels, out_channels),
                                                     dtype=dtype,
                                                     device=device),requires_grad=True)
        self.bias = None
        if bias:
            self.bias = Parameter(init.rand(out_channels, dtype=dtype, device=device), requires_grad=True)
        
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        input_x = x.transpose((1,2)).transpose((2,3))
        out = ops.conv(input_x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias:
            out += (self.bias).reshape((1,1,1,self.out_channels)).broadcast_to(out.shape)
        # BUG: OPS从没有在原地修改，都是建立新的计算图结点（tensor）
        return out.transpose((2,3)).transpose((1,2))
        
        ### END YOUR SOLUTION