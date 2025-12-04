from memory import UnsafePointer
from math import sqrt


fn layernorm_kernel(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    gamma: UnsafePointer[Float32],
    beta: UnsafePointer[Float32],
    B: Int,
    S: Int,
    H: Int,
    eps: Float32
):
    var total_rows = B * S

    for row in range(total_rows):
        var row_start = row * H
        var output_mut = output.mut_cast[True]()

        # Compute mean
        var mean: Float32 = 0.0
        for h in range(H):
            mean += (input + (row_start + h))[]
        mean = mean / Float32(H)

        # Compute variance
        var variance: Float32 = 0.0
        for h in range(H):
            var diff = (input + (row_start + h))[] - mean
            variance += diff * diff
        variance = variance / Float32(H)

        var inv_std = 1.0 / sqrt(variance + eps)

        # Normalize, scale, shift
        for h in range(H):
            var x = (input + (row_start + h))[]
            var normed = (x - mean) * inv_std
            var g = (gamma + h)[]
            var b = (beta + h)[]
            (output_mut + (row_start + h))[] = normed * g + b
