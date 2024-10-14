import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


# TODO: implement by myself
# def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
#     assert nonlinearity == "relu", "Only relu supported currently"
#     ### BEGIN YOUR SOLUTION
#     raise NotImplementedError()
#     ### END YOUR SOLUTION
def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    if shape is not None:
        fan_in = math.prod(shape[:-1])
    else:
        shape = (fan_in, fan_out)
    gain = math.sqrt(2.0)
    bound = gain * math.sqrt(3.0 / fan_in)
    return rand(*shape, low=-bound, high=bound, **kwargs)

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION