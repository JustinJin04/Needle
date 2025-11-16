import operator
import math
from functools import reduce
import numpy as np
from . import ndarray_backend_numpy
from . import ndarray_backend_cpu


# math.prod not in Python 3.7
def prod(x):
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """A backend device, wrapps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cuda():
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy():
    """Return numpy device"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)


def default_device():
    return cpu_numpy()


def all_devices():
    """return a list of all available devices"""
    return [cpu(), cuda(), cpu_numpy()]


class NDArray:
    """A generic ND array class that may contain multipe different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.

    This class will only contains those functions that you need to implement
    to actually get the desired functionality for the programming examples
    in the homework, and no more.

    For now, for simplicity the class only supports float32 types, though
    this can be extended if desired.
    """

    def __init__(self, other, device=None):
        """
        Create by copying another NDArray, or from numpy
        currently only support dtype="float32"
        """
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other, dtype="float32"), device=device)
            self._init(array)

    def _init(self, other):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape):
        """Utility function to compute compact strides"""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """Create a new NDArray with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    ### Properies and string representations
    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides
    
    @property
    def offset(self):
        return self._offset

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # only support float32 for now
        return "float32"

    @property
    def ndim(self):
        """Return number of dimensions."""
        return len(self._shape)

    @property
    def size(self):
        return prod(self._shape)

    def __repr__(self):
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self):
        return self.numpy().__str__()

    ### Basic array manipulation
    def fill(self, value):
        """Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device):
        """Convert between devices, using to/from numpy calls as the unifying bridge."""
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)

    def numpy(self):
        """convert to a numpy array"""
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self):
        """Return true if array is compact in memory and internal size equals product
        of the shape dimensions"""
        return (
            self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size
        )

    def compact(self):
        """Convert a matrix to be compact"""
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device)
            # print(f"self._handle: {self._handle}, out._handle: {out._handle}, self.shape: {self.shape}, self.strides: {self.strides}, self._offset: {self._offset}")
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            assert out.is_compact()
            return out

    def as_strided(self, shape, strides):
        """Restride the matrix without copying memory."""
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle
        )

    @property
    def flat(self):
        return self.reshape((self.size,))

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.

        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.

        Args:
            new_shape (tuple): new shape of the array

        Returns:
            NDArray : reshaped array; this will point to thep
        """

        ### BEGIN YOUR SOLUTION
        if self.is_compact() is not True or prod(new_shape) != prod(self.shape):
            # raise ValueError
            if self.is_compact() is not True:
                raise ValueError
            else:
                raise ValueError
        new_strides = NDArray.compact_strides(new_shape)
        return NDArray.make(new_shape,new_strides,self.device,self._handle)
        ### END YOUR SOLUTION

    def permute(self, new_axes):
        """
        Permute order of the dimensions.  new_axes describes a permuation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memroy as the original array.

        Args:
            new_axes (tuple): permuation order of the dimensions

        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """

        ### BEGIN YOUR SOLUTION
        assert self._offset == 0,"TODO: consider offset != 0"
        def permute_func(input_tuple, permute_tuple):
            return tuple(input_tuple[i] for i in permute_tuple)
    
        new_shape = permute_func(self.shape, new_axes)
        new_strides = permute_func(self.strides, new_axes)
        return NDArray.make(new_shape,new_strides,self.device,self._handle)

        ### END YOUR SOLUTION

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.

        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1

        Args:
            new_shape (tuple): shape to broadcast to

        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """

        ### BEGIN YOUR SOLUTION
        assert self._offset == 0,"TODO: consider offset != 0"

        def reverse(input_tuple):
            return input_tuple[::-1]

        new_strides_reversed = []
        old_strides_reversed = reverse(self.strides)

        new_shape_reversed = reverse(new_shape)
        old_shape_reversed = reverse(self.shape)
        for i in range(len(new_shape_reversed)):
            if i < len(old_shape_reversed):
                if old_shape_reversed[i] != 1 and new_shape_reversed[i] != old_shape_reversed[i]:
                    raise ValueError
                elif old_shape_reversed[i] == 1:
                    new_strides_reversed.append(0)
                else:
                    new_strides_reversed.append(old_strides_reversed[i])
            else:
                new_strides_reversed.append(0)
        return NDArray.make(new_shape,reverse(tuple(new_strides_reversed)),self.device,self._handle)
            
        
        ### END YOUR SOLUTION

    ### Get and set elements

    def process_slice(self, sl, dim):
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs):
        """
        __getitem__ 运算符，用于访问数组元素。
        此版本已修正整数索引和负数索引的问题，但*不*支持 Ellipsis (...)。
        """
        assert self._offset == 0, "TODO: 考虑 offset != 0 的情况"

        # 1. 将索引标准化为元组
        if not isinstance(idxs, tuple):
            idxs = (idxs,)

        # 2. 初始化新视图的属性
        new_shape = []
        new_strides = []
        new_offset = self._offset  # 从当前数组的偏移量开始

        # 3. 处理索引并进行填充
        processed_idxs = list(idxs)
        
        # 显式地禁止 Ellipsis
        assert all(s is not Ellipsis for s in processed_idxs), \
            "此 __getitem__ 版本暂不支持 Ellipsis (...)"

        # 如果提供的索引数量少于数组维度，用完整的切片 (slice(None)) 填充末尾
        num_missing_dims = self.ndim - len(processed_idxs)
        if num_missing_dims > 0:
            processed_idxs.extend([slice(None)] * num_missing_dims)

        # 索引过多（即使填充后）也是一个错误
        assert len(processed_idxs) == self.ndim, \
            f"索引过多：数组有 {self.ndim} 维，但提供了 {len(idxs)} 个索引"

        # 4. 循环计算新的 shape, strides 和 offset
        for i, idx in enumerate(processed_idxs):
            dim_size = self.shape[i]
            dim_stride = self.strides[i]

            if isinstance(idx, int):
                ### 修正 Bug 1 (降维) 和 Bug 2 (负索引) ###
                
                # 处理负数索引
                if idx < 0:
                    idx += dim_size
                
                # 索引越界检查
                assert 0 <= idx < dim_size, \
                    f"索引 {idx} 超出第 {i} 维的范围 (大小为 {dim_size})"
                
                # 整数索引会增加偏移量
                new_offset += idx * dim_stride
                
                # **关键修正**：
                # 整数索引会“吃掉”这个维度，
                # 所以我们 *不* 将它添加到 new_shape 或 new_strides 中。
                
            elif isinstance(idx, slice):
                # 这是一个切片，使用 self.process_slice 进行标准化
                # (我们假设 process_slice 能正确处理 None 和负数)
                s = self.process_slice(idx, i) # i 是维度索引

                assert s.step > 0, "切片的步长 (step) 必须为正"
                
                # 计算这个维度的新大小
                if s.stop <= s.start:
                    current_dim_shape = 0
                else:
                    # (公式已在您的代码中提供)
                    current_dim_shape = (s.stop - s.start + s.step - 1) // s.step

                # 切片的起始位置会增加偏移量
                new_offset += s.start * dim_stride

                # **关键修正**：
                # 切片索引会 *保留* 这个维度。
                
                # 添加到 new shape 和 new strides
                new_shape.append(current_dim_shape)
                new_strides.append(dim_stride * s.step)

            else:
                raise TypeError(f"不支持的索引类型: {type(idx)}")

        # 5. 返回新的 NDArray 视图
        return NDArray.make(
            tuple(new_shape),
            tuple(new_strides),
            self.device,
            self._handle,
            new_offset
        )


    def __setitem__(self, idxs, other):
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape), f"Size mismatch in setitem {view.shape} vs {other.shape}"
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### Collection of elementwise and scalar function: add, multiply, boolean, etc

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        """Run either an elementwise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    ### Elementwise functions

    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other):
        """Matrix multplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.

        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will restride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size

        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """

        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # if the matrix is aligned, use tiled matrix multiplication
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )

        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Reductions, i.e., sum/max over all element or over given axis
    # def reduce_view_out(self, axis, keepdims=False):
    #     """ Return a view to the array set up for reduction functions and output array. """
    #     if isinstance(axis, tuple) and not axis:
    #         raise ValueError("Empty axis in reduce")

    #     if axis is None:
    #         view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
    #         #out = NDArray.make((1,) * self.ndim, device=self.device)
    #         out = NDArray.make((1,), device=self.device)

    #     else:
    #         if isinstance(axis, (tuple, list)):
    #             assert len(axis) == 1, "Only support reduction over a single axis"
    #             axis = axis[0]

    #         view = self.permute(
    #             tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
    #         )
    #         out = NDArray.make(
    #             tuple([1 if i == axis else s for i, s in enumerate(self.shape)])
    #             if keepdims else
    #             tuple([s for i, s in enumerate(self.shape) if i != axis]),
    #             device=self.device,
    #         )
    #     return view, out

    def reduce_view_out(self, axis, keepdims=False):
        """ Return a view to the array set up for reduction functions and output array. 
            edit: my implementation to support multiple axes
        """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            #out = NDArray.make((1,) * self.ndim, device=self.device)
            out = NDArray.make((1,), device=self.device)
            reduce_size = view.shape[-1]

        else:
            if isinstance(axis, int):
                axis = (axis, )
            elif isinstance(axis, list):
                axis = tuple(axis)
            
            view = self.permute(tuple([a for a in range(self.ndim) if a not in axis]) + axis)
            out = NDArray.make(
                tuple([1 if i in axis else s for i, s in enumerate(self.shape)])
                if keepdims else
                tuple([s for i, s in enumerate(self.shape) if i not in axis]),
                device=self.device,
            )
            reduce_size = 1
            for i in axis:
                reduce_size *= self.shape[i]

        return view, out, reduce_size

    

    def sum(self, axis=None, keepdims=False):
        # axis is 
        # an int: sum over it
        # a tuple without elements: do nothing and return self
        # a tuple with several elemetns: sum over these axes
        # None: sum over all axis
        
        if isinstance(axis, tuple) and not axis:
            return self
        view, out, reduce_size = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, reduce_size)
        return out

    # def sum(self, axes=None, keepdims=False):
    #     """modify axis to axes to support multiple axes"""
    #     if axes is None:
    #         # If no axes specified, reduce across all dimensions
    #         view, out = self.reduce_view_out(None, keepdims=keepdims)
    #         self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
    #     else:
    #         if isinstance(axes, int):
    #             axes = [axes]  # Convert single axis to list for uniform handling
            
    #         # Sort axes to ensure reduction happens from the highest dimension to lowest
    #         axes = reversed(sorted(axes))
    #         out = None

    #         # Reduce along each axis in the list
    #         for axis in axes:
    #             if out is None:
    #                 view, out = self.reduce_view_out(axis, keepdims=keepdims)
    #             else:
    #                 view, out = out.reduce_view_out(axis, keepdims=keepdims)
    #             self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])

    #         # Adjust shape for final result if keepdims is False
    #         if not keepdims:
    #             new_shape = [s for i, s in enumerate(out.shape) if i not in axes]
    #             out = out.reshape(tuple(new_shape))

    #     return out


    def max(self, axis=None, keepdims=False):
        assert axis is None or isinstance(axis, int) or len(axis) == 1, "only support one axis reduce"
        view, out, reduce_size = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def flip(self, axes):
        """
        Flip this ndarray along the specified axes.
        Note: compact() before returning.
        """
        ### BEGIN YOUR SOLUTION
        new_offset = self.offset
        new_strides = list(self.strides)

        if isinstance(axes, int):
            new_offset += self.strides[axes] * (self.shape[axes] - 1)
            new_strides[axes] = -self.strides[axes]
        else:
            for axis in axes:
                new_offset += self.strides[axis] * (self.shape[axis] - 1)
                new_strides[axis] = -self.strides[axis]
        
        out = NDArray.make(self.shape, new_strides, self.device, self._handle, new_offset)
        return out.compact()
        ### END YOUR SOLUTION

    def pad(self, axes):
        """
        Pad this ndarray by zeros by the specified amount in `axes`,
        which lists for _all_ axes the left and right padding amount, e.g.,
        axes = ( (0, 0), (1, 1), (0, 0)) pads the middle axis with a 0 on the left and right side.
        """
        ### BEGIN YOUR SOLUTION
        assert len(axes) == len(self.shape)
        new_shape = []
        slice_list = []
        for i, j in zip(self.shape, axes):
            new_shape.append(i + j[0] + j[1])
            slice_list.append(slice(j[0], j[0] + i, 1))
        new_shape = tuple(new_shape)
        slice_list = tuple(slice_list)
        out = full(new_shape, 0, dtype=self.dtype, device=self.device)
        # print(out[slice_list].shape)
        # print(self.shape)
        out[slice_list] = self
        return out


        ### END YOUR SOLUTION

def array(a, dtype="float32", device=None):
    """Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    # assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)


def reshape(array, new_shape):
    return array.reshape(new_shape)


def maximum(a, b):
    return a.maximum(b)


def log(a):
    return a.log()


def exp(a):
    return a.exp()


def tanh(a):
    return a.tanh()


def sum(a, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)

def zeros(shape, device):
    return full(shape, 0, device = device)

def swapaxes(a: NDArray, axis1, axis2):
    new_axes = list(range(len(a.shape)))
    new_axes[axis1], new_axes[axis2] = new_axes[axis2], new_axes[axis1]
    return a.permute(new_axes)

def transpose(a, axes=None):
    return a.permute(axes)

def power(a, b):
    return a ** b

def matmul(a,b):
    return a @ b

def max(a, axis=None, keepdims=False):
    return a.max(axis, keepdims)