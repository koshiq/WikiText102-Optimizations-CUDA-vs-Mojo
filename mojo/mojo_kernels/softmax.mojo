"""
Mojo Softmax Kernels
Numerically stable softmax and log_softmax for MAX Engine
"""

from max import register
from max.tensor import Tensor, TensorShape
from max.driver import Tensor as MaxTensor
from algorithm import vectorize, parallelize
from math import exp, log, max as math_max


@register.op("mojo::softmax")
fn softmax(
    input: MaxTensor,
    dim: Int = -1
) raises -> MaxTensor:
    """
    Numerically stable softmax operation.

    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Args:
        input: Input tensor
        dim: Dimension along which to compute softmax (default: -1)

    Returns:
        Softmax output tensor
    """
    let input_shape = input.shape
    let ndim = len(input_shape)

    # Handle negative dim
    var actual_dim = dim
    if dim < 0:
        actual_dim = ndim + dim

    # Get dimensions
    let outer_size = _product_before(input_shape, actual_dim)
    let dim_size = input_shape[actual_dim]
    let inner_size = _product_after(input_shape, actual_dim)

    var result = MaxTensor.allocate(input_shape)

    # Process each row (outer x inner)
    @parameter
    fn compute_softmax(row: Int):
        let outer = row // inner_size
        let inner = row % inner_size

        # Find max for numerical stability
        var max_val: Float32 = -3.4028235e38  # -inf for float32

        for d in range(dim_size):
            let idx = outer * dim_size * inner_size + d * inner_size + inner
            let val = input[idx]
            if val > max_val:
                max_val = val

        # Compute exp(x - max) and sum
        var sum: Float32 = 0.0
        var exp_values = List[Float32](dim_size)

        for d in range(dim_size):
            let idx = outer * dim_size * inner_size + d * inner_size + inner
            let exp_val = exp(input[idx] - max_val)
            exp_values.append(exp_val)
            sum += exp_val

        # Normalize
        let inv_sum = 1.0 / sum

        for d in range(dim_size):
            let idx = outer * dim_size * inner_size + d * inner_size + inner
            result[idx] = exp_values[d] * inv_sum

    parallelize[compute_softmax](outer_size * inner_size)

    return result


@register.op("mojo::log_softmax")
fn log_softmax(
    input: MaxTensor,
    dim: Int = -1
) raises -> MaxTensor:
    """
    Numerically stable log-softmax operation.

    log_softmax(x_i) = x_i - max(x) - log(sum(exp(x_j - max(x))))

    More numerically stable than log(softmax(x)).

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax

    Returns:
        Log-softmax output tensor
    """
    let input_shape = input.shape
    let ndim = len(input_shape)

    var actual_dim = dim
    if dim < 0:
        actual_dim = ndim + dim

    let outer_size = _product_before(input_shape, actual_dim)
    let dim_size = input_shape[actual_dim]
    let inner_size = _product_after(input_shape, actual_dim)

    var result = MaxTensor.allocate(input_shape)

    @parameter
    fn compute_log_softmax(row: Int):
        let outer = row // inner_size
        let inner = row % inner_size

        # Find max for numerical stability
        var max_val: Float32 = -3.4028235e38

        for d in range(dim_size):
            let idx = outer * dim_size * inner_size + d * inner_size + inner
            let val = input[idx]
            if val > max_val:
                max_val = val

        # Compute log(sum(exp(x - max)))
        var sum_exp: Float32 = 0.0

        for d in range(dim_size):
            let idx = outer * dim_size * inner_size + d * inner_size + inner
            sum_exp += exp(input[idx] - max_val)

        let log_sum_exp = log(sum_exp)

        # Compute log_softmax = x - max - log(sum(exp))
        for d in range(dim_size):
            let idx = outer * dim_size * inner_size + d * inner_size + inner
            result[idx] = input[idx] - max_val - log_sum_exp

    parallelize[compute_log_softmax](outer_size * inner_size)

    return result


@register.op("mojo::softmax_backward")
fn softmax_backward(
    grad_output: MaxTensor,
    output: MaxTensor,
    dim: Int = -1
) raises -> MaxTensor:
    """
    Backward pass for softmax.

    grad_input[i] = output[i] * (grad_output[i] - sum(grad_output * output))

    Args:
        grad_output: Gradient from upstream
        output: Output from forward softmax pass
        dim: Dimension along which softmax was computed

    Returns:
        Gradient with respect to input
    """
    let shape = output.shape
    let ndim = len(shape)

    var actual_dim = dim
    if dim < 0:
        actual_dim = ndim + dim

    let outer_size = _product_before(shape, actual_dim)
    let dim_size = shape[actual_dim]
    let inner_size = _product_after(shape, actual_dim)

    var grad_input = MaxTensor.allocate(shape)

    @parameter
    fn compute_gradient(row: Int):
        let outer = row // inner_size
        let inner = row % inner_size

        # Compute sum(grad_output * output)
        var sum_grad_output: Float32 = 0.0

        for d in range(dim_size):
            let idx = outer * dim_size * inner_size + d * inner_size + inner
            sum_grad_output += grad_output[idx] * output[idx]

        # Compute gradient
        for d in range(dim_size):
            let idx = outer * dim_size * inner_size + d * inner_size + inner
            grad_input[idx] = output[idx] * (grad_output[idx] - sum_grad_output)

    parallelize[compute_gradient](outer_size * inner_size)

    return grad_input


# Helper functions

fn _product_before(shape: TensorShape, dim: Int) -> Int:
    """Compute product of dimensions before specified dimension"""
    var result = 1
    for i in range(dim):
        result *= shape[i]
    return result


fn _product_after(shape: TensorShape, dim: Int) -> Int:
    """Compute product of dimensions after specified dimension"""
    var result = 1
    for i in range(dim + 1, len(shape)):
        result *= shape[i]
    return result
