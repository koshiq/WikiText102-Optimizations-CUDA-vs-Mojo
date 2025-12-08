"""
Mojo LayerNorm Kernels
Numerically stable layer normalization for MAX Engine
"""

from max import register
from max.tensor import Tensor, TensorShape
from max.driver import Tensor as MaxTensor
from algorithm import vectorize, parallelize
from math import sqrt, rsqrt


@register.op("mojo::layer_norm")
fn layer_norm(
    input: MaxTensor,
    normalized_shape: Int,
    weight: MaxTensor,
    bias: MaxTensor,
    eps: Float32 = 1e-5
) raises -> MaxTensor:
    """
    Layer normalization forward pass.

    Normalizes over the last dimension(s) with shape normalized_shape.

    Args:
        input: Input tensor [..., normalized_shape]
        normalized_shape: Size of the dimension to normalize over
        weight: Learnable weight [normalized_shape]
        bias: Learnable bias [normalized_shape]
        eps: Small constant for numerical stability

    Returns:
        Normalized output tensor
    """
    let input_shape = input.shape
    let total_size = _get_total_size(input_shape)
    let num_instances = total_size // normalized_shape

    var output = MaxTensor.allocate(input_shape)

    @parameter
    fn normalize_instance(instance_idx: Int):
        let base_offset = instance_idx * normalized_shape

        # Compute mean using Welford's online algorithm
        var mean: Float32 = 0.0
        for i in range(normalized_shape):
            let val = input[base_offset + i]
            mean += val
        mean = mean / normalized_shape

        # Compute variance
        var variance: Float32 = 0.0
        for i in range(normalized_shape):
            let val = input[base_offset + i]
            let diff = val - mean
            variance += diff * diff
        variance = variance / normalized_shape

        # Compute normalization: (x - mean) / sqrt(variance + eps)
        let inv_std = rsqrt(variance + eps)

        for i in range(normalized_shape):
            let idx = base_offset + i
            let normalized = (input[idx] - mean) * inv_std
            let w = weight[i]
            let b = bias[i]
            output[idx] = normalized * w + b

    parallelize[normalize_instance](num_instances)

    return output


@register.op("mojo::layer_norm_with_stats")
fn layer_norm_with_stats(
    input: MaxTensor,
    normalized_shape: Int,
    weight: MaxTensor,
    bias: MaxTensor,
    eps: Float32 = 1e-5
) raises -> (MaxTensor, MaxTensor, MaxTensor):
    """
    Layer normalization forward pass that also returns mean and inverse std.

    Used for training where we need these stats for the backward pass.

    Args:
        input: Input tensor [..., normalized_shape]
        normalized_shape: Size of the dimension to normalize over
        weight: Learnable weight [normalized_shape]
        bias: Learnable bias [normalized_shape]
        eps: Small constant for numerical stability

    Returns:
        Tuple of (output, mean, inv_std)
    """
    let input_shape = input.shape
    let total_size = _get_total_size(input_shape)
    let num_instances = total_size // normalized_shape

    var output = MaxTensor.allocate(input_shape)

    # Allocate tensors for mean and inv_std (one value per instance)
    var mean_shape = TensorShape(num_instances)
    var mean_tensor = MaxTensor.allocate(mean_shape)
    var inv_std_tensor = MaxTensor.allocate(mean_shape)

    @parameter
    fn normalize_instance(instance_idx: Int):
        let base_offset = instance_idx * normalized_shape

        # Compute mean
        var mean: Float32 = 0.0
        for i in range(normalized_shape):
            mean += input[base_offset + i]
        mean = mean / normalized_shape
        mean_tensor[instance_idx] = mean

        # Compute variance
        var variance: Float32 = 0.0
        for i in range(normalized_shape):
            let diff = input[base_offset + i] - mean
            variance += diff * diff
        variance = variance / normalized_shape

        # Compute inverse std
        let inv_std = rsqrt(variance + eps)
        inv_std_tensor[instance_idx] = inv_std

        # Normalize and apply affine transform
        for i in range(normalized_shape):
            let idx = base_offset + i
            let normalized = (input[idx] - mean) * inv_std
            output[idx] = normalized * weight[i] + bias[i]

    parallelize[normalize_instance](num_instances)

    return (output, mean_tensor, inv_std_tensor)


@register.op("mojo::layer_norm_backward")
fn layer_norm_backward(
    grad_output: MaxTensor,
    input: MaxTensor,
    normalized_shape: Int,
    weight: MaxTensor,
    mean: MaxTensor,
    inv_std: MaxTensor,
    eps: Float32 = 1e-5
) raises -> (MaxTensor, MaxTensor, MaxTensor):
    """
    Layer normalization backward pass.

    Args:
        grad_output: Gradient from upstream [..., normalized_shape]
        input: Original input tensor [..., normalized_shape]
        normalized_shape: Size of normalized dimension
        weight: Weight parameter [normalized_shape]
        mean: Mean computed in forward pass [num_instances]
        inv_std: Inverse std computed in forward pass [num_instances]
        eps: Epsilon used in forward pass

    Returns:
        Tuple of (grad_input, grad_weight, grad_bias)
    """
    let input_shape = input.shape
    let total_size = _get_total_size(input_shape)
    let num_instances = total_size // normalized_shape

    var grad_input = MaxTensor.allocate(input_shape)

    # Gradient accumulators for weight and bias
    var weight_shape = TensorShape(normalized_shape)
    var grad_weight = MaxTensor.allocate(weight_shape)
    var grad_bias = MaxTensor.allocate(weight_shape)

    # Initialize gradient accumulators to zero
    for i in range(normalized_shape):
        grad_weight[i] = 0.0
        grad_bias[i] = 0.0

    @parameter
    fn backward_instance(instance_idx: Int):
        let base_offset = instance_idx * normalized_shape
        let mean_val = mean[instance_idx]
        let inv_std_val = inv_std[instance_idx]

        # Compute intermediate gradients
        var sum_grad_output: Float32 = 0.0
        var sum_grad_output_centered: Float32 = 0.0

        for i in range(normalized_shape):
            let idx = base_offset + i
            let grad_out = grad_output[idx]
            let w = weight[i]
            let grad_normalized = grad_out * w

            sum_grad_output += grad_normalized

            let centered = input[idx] - mean_val
            sum_grad_output_centered += grad_normalized * centered

        # Compute gradients with respect to input
        let scale1 = inv_std_val / normalized_shape
        let scale2 = sum_grad_output * scale1
        let scale3 = sum_grad_output_centered * inv_std_val * inv_std_val * scale1

        for i in range(normalized_shape):
            let idx = base_offset + i
            let grad_out = grad_output[idx]
            let w = weight[i]
            let centered = input[idx] - mean_val

            let grad_normalized = grad_out * w
            grad_input[idx] = (grad_normalized - scale2 - centered * scale3) * inv_std_val

            # Accumulate weight and bias gradients
            let normalized = centered * inv_std_val
            let grad_w = grad_out * normalized
            let grad_b = grad_out

            # Note: This is a simplified version. In production, we'd use atomics
            # For now, we'll need to accumulate these serially after parallelization
            grad_weight[i] = grad_weight[i] + grad_w
            grad_bias[i] = grad_bias[i] + grad_b

    # Process instances (gradient accumulation done within, could use atomics)
    for instance_idx in range(num_instances):
        backward_instance(instance_idx)

    return (grad_input, grad_weight, grad_bias)


@register.op("mojo::rms_norm")
fn rms_norm(
    input: MaxTensor,
    normalized_shape: Int,
    weight: MaxTensor,
    eps: Float32 = 1e-5
) raises -> MaxTensor:
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Simpler variant that only normalizes by RMS, no mean subtraction.
    Used in LLaMA and other modern architectures.

    Args:
        input: Input tensor [..., normalized_shape]
        normalized_shape: Size of the dimension to normalize over
        weight: Learnable weight [normalized_shape]
        eps: Small constant for numerical stability

    Returns:
        Normalized output tensor
    """
    let input_shape = input.shape
    let total_size = _get_total_size(input_shape)
    let num_instances = total_size // normalized_shape

    var output = MaxTensor.allocate(input_shape)

    @parameter
    fn normalize_instance(instance_idx: Int):
        let base_offset = instance_idx * normalized_shape

        # Compute mean square
        var mean_square: Float32 = 0.0
        for i in range(normalized_shape):
            let val = input[base_offset + i]
            mean_square += val * val
        mean_square = mean_square / normalized_shape

        # Compute normalization: x / sqrt(mean_square + eps)
        let inv_rms = rsqrt(mean_square + eps)

        for i in range(normalized_shape):
            let idx = base_offset + i
            output[idx] = input[idx] * inv_rms * weight[i]

    parallelize[normalize_instance](num_instances)

    return output


# Helper functions

fn _get_total_size(shape: TensorShape) -> Int:
    """Compute total number of elements in tensor"""
    var result = 1
    for i in range(len(shape)):
        result *= shape[i]
    return result
