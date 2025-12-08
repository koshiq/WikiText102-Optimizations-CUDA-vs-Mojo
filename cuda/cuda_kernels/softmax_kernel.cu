/**
 * Optimized Softmax Kernels
 *
 * Optimizations:
 * - Online (single-pass where possible) algorithm
 * - Numerically stable with max subtraction
 * - Warp-level shuffle reductions
 * - Fused operations (max, exp, sum, normalize)
 * - Special attention softmax with masking and scaling
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// ============================================================================
// Standard Softmax Forward Kernel
// ============================================================================

template<typename T>
__global__ void softmax_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int N, int D
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const T* x = input + row * D;
    T* y = output + row * D;

    __shared__ float shared_data[WARP_SIZE * 2];
    float* shared_max = shared_data;
    float* shared_sum = shared_data + WARP_SIZE;

    // ========== PASS 1: Find maximum value for numerical stability ==========
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = static_cast<float>(x[i]);
        thread_max = fmaxf(thread_max, val);
    }

    // Warp-level reduction for max
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(mask, thread_max, offset);
        thread_max = fmaxf(thread_max, other_max);
    }

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        shared_max[warp_id] = thread_max;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float max_val = shared_max[0];
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 1; i < num_warps; i++) {
            max_val = fmaxf(max_val, shared_max[i]);
        }
        shared_max[0] = max_val;
    }
    __syncthreads();

    float max_val = shared_max[0];

    // ========== PASS 2: Compute exp and sum ==========
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = expf(static_cast<float>(x[i]) - max_val);
        thread_sum += val;
        y[i] = static_cast<T>(val); // Store exp temporarily
    }

    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    if (lane == 0) {
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total_sum = 0.0f;
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 0; i < num_warps; i++) {
            total_sum += shared_sum[i];
        }
        shared_sum[0] = 1.0f / total_sum;  // Store reciprocal
    }
    __syncthreads();

    float inv_sum = shared_sum[0];

    // ========== PASS 3: Normalize ==========
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        y[i] = static_cast<T>(static_cast<float>(y[i]) * inv_sum);
    }
}

// ============================================================================
// Softmax Backward Kernel
// ============================================================================

template<typename T>
__global__ void softmax_backward_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ output,
    T* __restrict__ grad_input,
    int N, int D
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const T* dy = grad_output + row * D;
    const T* y = output + row * D;
    T* dx = grad_input + row * D;

    __shared__ float shared_sum[WARP_SIZE];

    // ========== PASS 1: Compute sum(dy * y) ==========
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        thread_sum += static_cast<float>(dy[i]) * static_cast<float>(y[i]);
    }

    // Warp-level reduction
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 0; i < num_warps; i++) {
            total += shared_sum[i];
        }
        shared_sum[0] = total;
    }
    __syncthreads();

    float sum_dy_y = shared_sum[0];

    // ========== PASS 2: Compute gradient: dx = y * (dy - sum) ==========
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float y_i = static_cast<float>(y[i]);
        float dy_i = static_cast<float>(dy[i]);
        dx[i] = static_cast<T>(y_i * (dy_i - sum_dy_y));
    }
}

// ============================================================================
// Attention Softmax (Fused with scaling and masking)
// ============================================================================

template<typename T>
__global__ void attention_softmax_kernel(
    const T* __restrict__ scores,
    T* __restrict__ probs,
    const bool* __restrict__ mask,
    int batch_size, int num_heads, int seq_len,
    float scale
) {
    int batch_head = blockIdx.y;
    int query_idx = blockIdx.x;

    if (query_idx >= seq_len) return;

    int batch = batch_head / num_heads;
    int head = batch_head % num_heads;
    int offset = ((batch * num_heads + head) * seq_len + query_idx) * seq_len;

    const T* x = scores + offset;
    T* y = probs + offset;
    const bool* m = mask ? mask + query_idx * seq_len : nullptr;

    __shared__ float shared_data[WARP_SIZE * 2];
    float* shared_max = shared_data;
    float* shared_sum = shared_data + WARP_SIZE;

    // ========== PASS 1: Find max with masking and scaling ==========
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        if (!m || m[i]) {
            float val = static_cast<float>(x[i]) * scale;
            thread_max = fmaxf(thread_max, val);
        }
    }

    unsigned mask_val = 0xffffffff;
    #pragma unroll
    for (int offset_val = 16; offset_val > 0; offset_val /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(mask_val, thread_max, offset_val));
    }

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) shared_max[warp_id] = thread_max;
    __syncthreads();

    if (threadIdx.x == 0) {
        float max_val = shared_max[0];
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 1; i < num_warps; i++) {
            max_val = fmaxf(max_val, shared_max[i]);
        }
        shared_max[0] = max_val;
    }
    __syncthreads();

    float max_val = shared_max[0];

    // ========== PASS 2: Compute exp and sum with masking ==========
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val;
        if (!m || m[i]) {
            val = expf(static_cast<float>(x[i]) * scale - max_val);
            thread_sum += val;
        } else {
            val = 0.0f;  // Masked positions get 0 probability
        }
        y[i] = static_cast<T>(val);
    }

    #pragma unroll
    for (int offset_val = 16; offset_val > 0; offset_val /= 2) {
        thread_sum += __shfl_down_sync(mask_val, thread_sum, offset_val);
    }

    if (lane == 0) shared_sum[warp_id] = thread_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 0; i < num_warps; i++) {
            sum += shared_sum[i];
        }
        shared_sum[0] = 1.0f / sum;
    }
    __syncthreads();

    float inv_sum = shared_sum[0];

    // ========== PASS 3: Normalize ==========
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        y[i] = static_cast<T>(static_cast<float>(y[i]) * inv_sum);
    }
}

// ============================================================================
// Host Functions (PyTorch Interface)
// ============================================================================

torch::Tensor softmax_cuda_forward(torch::Tensor input, int dim) {
    // Flatten all dims except the softmax dim
    auto sizes = input.sizes().vec();
    int N = 1;
    for (int i = 0; i < dim; i++) N *= sizes[i];
    for (int i = dim + 1; i < sizes.size(); i++) N *= sizes[i];
    int D = sizes[dim];

    auto input_2d = input.view({N, D});
    auto output = torch::empty_like(input_2d);

    const int threads = BLOCK_SIZE;
    const int blocks = N;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "softmax_forward_cuda", ([&] {
        softmax_forward_kernel<scalar_t><<<blocks, threads>>>(
            input_2d.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, D
        );
    }));

    return output.view(sizes);
}

torch::Tensor softmax_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor output
) {
    auto sizes = output.sizes().vec();
    int total = output.numel();
    int D = sizes.back();
    int N = total / D;

    auto grad_output_2d = grad_output.view({N, D});
    auto output_2d = output.view({N, D});
    auto grad_input = torch::empty_like(output_2d);

    const int threads = BLOCK_SIZE;
    const int blocks = N;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.scalar_type(), "softmax_backward_cuda", ([&] {
        softmax_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output_2d.data_ptr<scalar_t>(),
            output_2d.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            N, D
        );
    }));

    return grad_input.view(sizes);
}

torch::Tensor attention_softmax_cuda(
    torch::Tensor scores,
    torch::Tensor mask,
    float scale
) {
    const int batch_size = scores.size(0);
    const int num_heads = scores.size(1);
    const int seq_len = scores.size(2);

    auto probs = torch::empty_like(scores);

    const int threads = BLOCK_SIZE;
    dim3 blocks(seq_len, batch_size * num_heads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(scores.scalar_type(), "attention_softmax_cuda", ([&] {
        attention_softmax_kernel<scalar_t><<<blocks, threads>>>(
            scores.data_ptr<scalar_t>(),
            probs.data_ptr<scalar_t>(),
            mask.defined() ? mask.data_ptr<bool>() : nullptr,
            batch_size, num_heads, seq_len, scale
        );
    }));

    return probs;
}
