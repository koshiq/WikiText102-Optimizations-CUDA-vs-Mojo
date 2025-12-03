from memory import UnsafePointer, alloc, memset_zero
from algorithm import parallelize
from math import exp

struct SoftmaxKernel:

    @staticmethod
    fn launch_kernel(
        input: UnsafePointer[Float32],
        output: UnsafePointer[Float32],
        B: Int,   # batch size
        S: Int    # sequence length (softmax dimension)
    ) raises:

        var total_rows = B

        @parameter
        fn process_row(b: Int):
            SoftmaxKernel._softmax_row(input, output, b, S)

        parallelize[process_row](total_rows, total_rows)

    # ------------------------------------------
    # Per-row softmax
    # ------------------------------------------
    @staticmethod
    fn _softmax_row(
        input: UnsafePointer[Float32],
        output: UnsafePointer[Float32],
        b: Int,
        S: Int
    ):

        var row_start = b * S
        var output_mut = output.mut_cast[True]()

        # ---- 1) Max for numerical stability ----
        var max_val: Float32 = -1e30 
        for i in range(S):
            var v = (input + (row_start + i))[]
            if v > max_val:
                max_val = v

        # ---- 2) Compute exp(x - max) sum ----
        var sum_exp: Float32 = 0.0
        for i in range(S):
            var e = exp((input + (row_start + i))[] - max_val)
            sum_exp += e
            (output_mut + (row_start + i))[] = e  # store temporary

        # ---- 3) Normalize ----
        for i in range(S):
            var e = (output_mut + (row_start + i))[]
            (output_mut + (row_start + i))[] = e / sum_exp
