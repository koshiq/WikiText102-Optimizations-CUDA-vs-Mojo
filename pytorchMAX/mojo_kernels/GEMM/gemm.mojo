from memory import UnsafePointer, alloc, memset_zero
from algorithm import parallelize

alias TILE_M = 128
alias TILE_N = 128
alias TILE_K = 32
alias THREAD_M = 8
alias THREAD_N = 8
alias SMEM_PAD = 8


struct GEMMKernel:
    @staticmethod
    fn launch_kernel(
        A: UnsafePointer[Float32],
        B: UnsafePointer[Float32],
        mut C: UnsafePointer[Float32],
        M: Int, N: Int, K: Int,
        alpha: Float32, beta: Float32,
        lda: Int, ldb: Int, ldc: Int
    ) raises:
        var grid_m = (M + TILE_M - 1) // TILE_M
        var grid_n = (N + TILE_N - 1) // TILE_N
        var num_blocks = grid_m * grid_n

        @parameter
        fn process_block(block_idx: Int):
            var block_m = (block_idx // grid_n) * TILE_M
            var block_n = (block_idx % grid_n) * TILE_N
            Self._gemm_block(
                A, B, C, M, N, K,
                alpha, beta,
                lda, ldb, ldc,
                block_m, block_n
            )

        parallelize[process_block](num_blocks, num_blocks)

    @staticmethod
    fn _gemm_block(
        A: UnsafePointer[Float32],
        B: UnsafePointer[Float32],
        mut C: UnsafePointer[Float32],
        M: Int, N: Int, K: Int,
        alpha: Float32, beta: Float32,
        lda: Int, ldb: Int, ldc: Int,
        block_m: Int, block_n: Int
    ):
        var smem_A = alloc[Float32](TILE_M * (TILE_K + SMEM_PAD))
        var smem_B = alloc[Float32](TILE_K * (TILE_N + SMEM_PAD))
        var acc = alloc[Float32](THREAD_M * THREAD_N)

        memset_zero(acc, THREAD_M * THREAD_N)

        for k_tile in range(0, K, TILE_K):
            var k_end = min(k_tile + TILE_K, K)
            var k_size = k_end - k_tile

            Self._load_tile_A(A, smem_A, M, K, lda, block_m, k_tile, k_size)
            Self._load_tile_B(B, smem_B, K, N, ldb, k_tile, block_n, k_size)
            Self._compute_tile(smem_A, smem_B, acc, k_size)

        Self._store_tile_C(C, acc, M, N, ldc, block_m, block_n, alpha, beta)

        smem_A.free()
        smem_B.free()
        acc.free()

    @staticmethod
    fn _load_tile_A(
        A: UnsafePointer[Float32],
        smem_A: UnsafePointer[Float32],
        M: Int, K: Int, lda: Int,
        block_m: Int, k_tile: Int, k_size: Int
    ):
        var smem_A_mut = smem_A.mut_cast[True]()

        for m in range(TILE_M):
            var g_m = block_m + m
            for k in range(k_size):
                var dest = smem_A_mut + (m * (TILE_K + SMEM_PAD) + k)
                if g_m >= M:
                    dest[] = 0.0
                else:
                    var g_k = k_tile + k
                    var val = (A + (g_m * lda + g_k))[]
                    dest[] = val

    @staticmethod
    fn _load_tile_B(
        B: UnsafePointer[Float32],
        smem_B: UnsafePointer[Float32],
        K: Int, N: Int, ldb: Int,
        k_tile: Int, block_n: Int, k_size: Int
    ):
        var smem_B_mut = smem_B.mut_cast[True]()

        for k in range(k_size):
            for n in range(TILE_N):
                var dest = smem_B_mut + (k * (TILE_N + SMEM_PAD) + n)
                var g_k = k_tile + k
                var g_n = block_n + n

                if g_k >= K or g_n >= N:
                    dest[] = 0.0
                else:
                    dest[] = (B + (g_k * ldb + g_n))[]

    @staticmethod
    fn _compute_tile(
        smem_A: UnsafePointer[Float32],
        smem_B: UnsafePointer[Float32],
        acc: UnsafePointer[Float32],
        k_size: Int
    ):
        var acc_mut = acc.mut_cast[True]()

        for k in range(k_size):
            for m in range(THREAD_M):
                for n in range(THREAD_N):
                    var a_offset = m * (TILE_K + SMEM_PAD) + k
                    var a_val = (smem_A + a_offset)[]

                    var b_offset = k * (TILE_N + SMEM_PAD) + n
                    var b_val = (smem_B + b_offset)[]

                    var idx = m * THREAD_N + n
                    var old = (acc_mut + idx)[]
                    (acc_mut + idx)[] = old + a_val * b_val

    @staticmethod
    fn _store_tile_C(
        C: UnsafePointer[Float32],
        acc: UnsafePointer[Float32],
        M: Int, N: Int, ldc: Int,
        block_m: Int, block_n: Int,
        alpha: Float32, beta: Float32
    ):
        var C_mut = C.mut_cast[True]()

        for m in range(THREAD_M):
            var g_m = block_m + m
            if g_m >= M:
                continue

            for n in range(THREAD_N):
                var g_n = block_n + n
                if g_n >= N:
                    continue

                var c_idx = g_m * ldc + g_n
                var acc_val = (acc + (m * THREAD_N + n))[]
                var dst = C_mut + c_idx

                if beta == 0.0:
                    dst[] = alpha * acc_val
                else:
                    dst[] = alpha * acc_val + beta * dst[]
