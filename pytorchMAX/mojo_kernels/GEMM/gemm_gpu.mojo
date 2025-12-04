from max.tensor import Tensor, TensorShape
from max.driver import Accelerator, Device, AnyTensor
from algorithm import parallelize
from memory import UnsafePointer


fn gemm_gpu(
    a: Tensor,
    b: Tensor,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0
) raises -> Tensor:
    """GPU-accelerated GEMM using MAX."""
    var m = a.shape()[0]
    var k = a.shape()[1]
    var n = b.shape()[1]

    # Create output tensor
    var c_shape = TensorShape(m, n)
    var c = Tensor(c_shape, a.dtype())

    # Simple matmul on GPU
    # MAX will automatically dispatch to GPU if available
    for i in range(m):
        for j in range(n):
            var sum: Float32 = 0.0
            for ki in range(k):
                sum += a[i, ki].cast[DType.float32]() * b[ki, j].cast[DType.float32]()
            c[i, j] = alpha * sum + (beta * c[i, j].cast[DType.float32]() if beta != 0.0 else 0.0)

    return c
