from memory import UnsafePointer
from math import exp


fn softmax_kernel(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    B: Int,
    S: Int
):
    for b in range(B):
        var row_start = b * S
        var output_mut = output.mut_cast[True]()

        # Find max for numerical stability
        var max_val: Float32 = -1e30
        for i in range(S):
            var v = (input + (row_start + i))[]
            if v > max_val:
                max_val = v

        # Compute exp(x - max) sum
        var sum_exp: Float32 = 0.0
        for i in range(S):
            var e = exp((input + (row_start + i))[] - max_val)
            sum_exp += e
            (output_mut + (row_start + i))[] = e

        # Normalize
        for i in range(S):
            var e = (output_mut + (row_start + i))[]
            (output_mut + (row_start + i))[] = e / sum_exp
