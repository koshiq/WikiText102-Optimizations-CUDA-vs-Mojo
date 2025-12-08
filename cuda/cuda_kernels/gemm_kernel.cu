/**
 * Optimized GEMM (General Matrix Multiply) Kernels for RTX 4070
 * C = alpha * (A @ B) + beta * C
 *
 * Optimizations:
 * - Shared memory tiling to reduce global memory access
 * - Coalesced memory access patterns
 * - Bank conflict avoidance (+1 padding)
 * - Loop unrolling for better ILP
 * - Support for FP32 and FP16
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 32
#define BLOCK_SIZE 32

// ============================================================================
// FP32 GEMM Kernel
// ============================================================================

__global__ void gemm_fp32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    // Shared memory with +1 padding to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // Tile across K dimension
    #pragma unroll 4
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory with coalesced access
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product with loop unrolling
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result with alpha and beta scaling
    if (row < M && col < N) {
        if (beta == 0.0f) {
            C[row * N + col] = alpha * sum;
        } else {
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        }
    }
}

// ============================================================================
// FP16 GEMM Kernel
// ============================================================================

__global__ void gemm_fp16_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    __shared__ __half As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ __half Bs[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    #pragma unroll 4
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = __float2half(0.0f);
        }

        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = __float2half(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float result = alpha * sum;
        if (beta != 0.0f) {
            result += beta * __half2float(C[row * N + col]);
        }
        C[row * N + col] = __float2half(result);
    }
}

// ============================================================================
// Host Functions (PyTorch Interface)
// ============================================================================

torch::Tensor gemm_cuda_fp32(
    torch::Tensor A,
    torch::Tensor B,
    float alpha,
    float beta
) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_fp32_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K, alpha, beta
    );

    return C;
}

torch::Tensor gemm_cuda_fp16(
    torch::Tensor A,
    torch::Tensor B,
    float alpha,
    float beta
) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_fp16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K, alpha, beta
    );

    return C;
}
