from memory import UnsafePointer, alloc, memset_zero
from algorithm import parallelize
from math import rsqrt

alias TILE_H = 256

struct LayerNormKernel:

    @staticmethod
    fn launch_kernel(
        input: UnsafePointer[Float32],
        output: UnsafePointer[Float32],
        gamma: UnsafePointer[Float32],
        beta: UnsafePointer[Float32],
        B: Int, S: Int, H: Int,
        eps: Float32
    ) raises:

        var total_rows = B * S

        @parameter
        fn process_row(row: Int):
            LayerNormKernel._layernorm_row(
                input, output, gamma, beta,
                row, H, eps
            )

        parallelize[process_row](total_rows, total_rows)

    # ============================================================
    # Per-row LayerNorm
    # ============================================================
    @staticmethod
    fn _layernorm_row(
        input: UnsafePointer[Float32],
        output: UnsafePointer[Float32],
        gamma: UnsafePointer[Float32],
        beta: UnsafePointer[Float32],
        row: Int,
        H: Int,
        eps: Float32
    ):
        var row_start = row * H

        # ---- Make output mutable (required by Mojo) ----
        var output_mut = output.mut_cast[True]()

        # ---- 1) Compute mean ----
        var mean: Float32 = 0.0
        for h in range(H):
            mean += (input + (row_start + h))[]
        mean /= Float32(H)

        # ---- 2) Variance ----
        var var_acc: Float32 = 0.0
        for h in range(H):
            var diff = (input + (row_start + h))[] - mean
            var_acc += diff * diff
        var variance = var_acc / Float32(H)
        var inv_std = rsqrt(variance + eps)

        # ---- 3) Normalize, scale, shift ----
        for h in range(H):
            var x = (input + (row_start + h))[]
            var normed = (x - mean) * inv_std
            var g = (gamma + h)[]
            var b = (beta + h)[]

            (output_mut + (row_start + h))[] = normed * g + b
