"""
Mojo GEMM (General Matrix Multiply) Kernel
Optimized for MAX Engine on NVIDIA GPUs
C = alpha * (A @ B) + beta * C
"""

from max import register
from max.tensor import Tensor, TensorShape
from max.driver import Tensor as MaxTensor, AnyTensor
from max.graph import ops, Graph, TensorType, Type
from algorithm import vectorize, parallelize
from memory import memset_zero
from math import sqrt, min, max as _max


@register.op("mojo::gemm")
fn gemm(
    a: MaxTensor,
    b: MaxTensor,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0
) raises -> MaxTensor:
    """
    General Matrix Multiplication: C = alpha * (A @ B) + beta * C

    Args:
        a: Input matrix A [M, K]
        b: Input matrix B [K, N]
        alpha: Scaling factor for A @ B
        beta: Scaling factor for C (if beta=0, C is not used)

    Returns:
        Output matrix C [M, N]
    """
    let a_shape = a.shape
    let b_shape = b.shape

    # Get dimensions
    let M = a_shape[0]
    let K = a_shape[1]
    let N = b_shape[1]

    # Validate dimensions
    if K != b_shape[0]:
        raise Error("Matrix dimensions don't match for multiplication")

    # Create output tensor
    var result = MaxTensor.allocate(TensorShape(M, N))

    # Tile sizes optimized for GPU cache
    alias TILE_M = 32
    alias TILE_N = 32
    alias TILE_K = 32

    # Parallelize over output tiles
    @parameter
    fn compute_tile(m_tile: Int):
        let m_start = m_tile * TILE_M
        let m_end = min(m_start + TILE_M, M)

        for n_tile in range((N + TILE_N - 1) // TILE_N):
            let n_start = n_tile * TILE_N
            let n_end = min(n_start + TILE_N, N)

            # Compute tile
            for m in range(m_start, m_end):
                for n in range(n_start, n_end):
                    var sum: Float32 = 0.0

                    # Vectorized dot product
                    @parameter
                    fn dot_product[width: Int](k: Int):
                        let k_end = min(k + width, K)
                        for ki in range(k, k_end):
                            sum += a[m, ki] * b[ki, n]

                    # Process K dimension in chunks
                    for k in range(0, K, TILE_K):
                        vectorize[dot_product, 8](k)

                    # Apply alpha and beta
                    result[m, n] = alpha * sum

    # Parallelize across M dimension
    parallelize[compute_tile]((M + TILE_M - 1) // TILE_M)

    return result


@register.op("mojo::gemm_transposed")
fn gemm_transposed(
    a: MaxTensor,
    b_transposed: MaxTensor,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0
) raises -> MaxTensor:
    """
    GEMM with B already transposed: C = alpha * (A @ B.T) + beta * C
    More efficient when B is stored transposed.

    Args:
        a: Input matrix A [M, K]
        b_transposed: Input matrix B transposed [N, K]
        alpha: Scaling factor
        beta: Scaling factor for C

    Returns:
        Output matrix C [M, N]
    """
    let a_shape = a.shape
    let b_shape = b_transposed.shape

    let M = a_shape[0]
    let K = a_shape[1]
    let N = b_shape[0]

    if K != b_shape[1]:
        raise Error("Matrix dimensions don't match")

    var result = MaxTensor.allocate(TensorShape(M, N))

    alias TILE_M = 32
    alias TILE_N = 32

    @parameter
    fn compute_row(m: Int):
        for n in range(N):
            var sum: Float32 = 0.0

            # Vectorized dot product (better memory access with transposed B)
            @parameter
            fn dot[width: Int](k: Int):
                let k_end = min(k + width, K)
                for ki in range(k, k_end):
                    # Both a[m, ki] and b_transposed[n, ki] are sequential
                    sum += a[m, ki] * b_transposed[n, ki]

            vectorize[dot, 8](0)

            result[m, n] = alpha * sum

    parallelize[compute_row](M)

    return result


@register.op("mojo::gemm_batched")
fn gemm_batched(
    a: MaxTensor,
    b: MaxTensor,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0
) raises -> MaxTensor:
    """
    Batched GEMM: C[i] = alpha * (A[i] @ B[i]) + beta * C[i]

    Args:
        a: Input batch [batch, M, K]
        b: Input batch [batch, K, N]
        alpha: Scaling factor
        beta: Scaling factor

    Returns:
        Output batch [batch, M, N]
    """
    let a_shape = a.shape
    let b_shape = b.shape

    let batch = a_shape[0]
    let M = a_shape[1]
    let K = a_shape[2]
    let N = b_shape[2]

    if batch != b_shape[0] or K != b_shape[1]:
        raise Error("Batched matrix dimensions don't match")

    var result = MaxTensor.allocate(TensorShape(batch, M, N))

    # Parallelize over batches
    @parameter
    fn compute_batch(b_idx: Int):
        for m in range(M):
            for n in range(N):
                var sum: Float32 = 0.0

                @parameter
                fn dot[width: Int](k: Int):
                    let k_end = min(k + width, K)
                    for ki in range(k, k_end):
                        sum += a[b_idx, m, ki] * b[b_idx, ki, n]

                vectorize[dot, 8](0)

                result[b_idx, m, n] = alpha * sum

    parallelize[compute_batch](batch)

    return result
