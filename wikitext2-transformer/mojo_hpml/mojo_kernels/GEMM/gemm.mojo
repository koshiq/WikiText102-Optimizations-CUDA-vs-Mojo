from memory import UnsafePointer, alloc, memset_zero
from algorithm import parallelize
from sys.info import simd_width_of

alias TILE_M = 128
alias TILE_N = 128
alias TILE_K = 32
alias THREAD_M = 8
alias THREAD_N = 8
alias SMEM_PAD = 8


struct GEMMKernel:

    @staticmethod
    fn launch_kernel(
        A: UnsafePointer[Float16],
        B: UnsafePointer[Float16],
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


    # ============================================================
    # GEMM block
    # ============================================================

    @staticmethod
    fn _gemm_block(
        A: UnsafePointer[Float16],
        B: UnsafePointer[Float16],
        mut C: UnsafePointer[Float32],
        M: Int, N: Int, K: Int,
        alpha: Float32, beta: Float32,
        lda: Int, ldb: Int, ldc: Int,
        block_m: Int, block_n: Int
    ):

        var smem_A = alloc[Float16](TILE_M * (TILE_K + SMEM_PAD))
        var smem_B = alloc[Float16](TILE_K * (TILE_N + SMEM_PAD))
        var acc = alloc[Float32](THREAD_M * THREAD_N)

        memset_zero(acc, THREAD_M * THREAD_N)

        for k_tile in range(0, K, TILE_K):
            var k_end = min(k_tile + TILE_K, K)
            var k_size = k_end - k_tile

            Self._load_tile_A(A, smem_A, M, K, lda, block_m, k_tile, k_size)
            Self._load_tile_B(B, smem_B, K, N, ldb, k_tile, block_n, k_size)
            Self._compute_tile(smem_A, smem_B, acc, k_size)

        Self._store_tile_C(
            C, acc, M, N, ldc,
            block_m, block_n,
            alpha, beta
        )

        smem_A.free()
        smem_B.free()
        acc.free()


    # ============================================================
    # LOAD TILE A
    # ============================================================

    @staticmethod
    fn _load_tile_A(
        A: UnsafePointer[Float16],
        smem_A: UnsafePointer[Float16],
        M: Int, K: Int, lda: Int,
        block_m: Int, k_tile: Int, k_size: Int
    ):
        # Changed 'let' to 'var' and fixed indentation
        var smem_A_mut = smem_A.mut_cast[True]()

        for m in range(TILE_M):
            var g_m = block_m + m

            for k in range(k_size):

                # Changed 'let' to 'var' to make the local 'dest' pointer binding mutable
                var dest = smem_A_mut + (m * (TILE_K + SMEM_PAD) + k)

                if g_m >= M:
                    dest[] = 0.0          
                else:
                    # Changed 'let' to 'var' for consistency/compatibility
                    var g_k = k_tile + k
                    # Changed 'let' to 'var' for consistency/compatibility
                    var val = (A + (g_m * lda + g_k))[]
                    dest[] = val



    # ============================================================
    # LOAD TILE B
    # ============================================================

    @staticmethod
    fn _load_tile_B(
        B: UnsafePointer[Float16],
        smem_B: UnsafePointer[Float16],
        K: Int, N: Int, ldb: Int,
        k_tile: Int, block_n: Int, k_size: Int
    ):
        # Convert smem_B into a mutable pointer
        var smem_B_mut = smem_B.mut_cast[True]()

        for k in range(k_size):

            for n in range(TILE_N):

                # Must use the mutable pointer, not smem_B
                var dest = smem_B_mut + (k * (TILE_N + SMEM_PAD) + n)

                var g_k = k_tile + k
                var g_n = block_n + n

                if g_k >= K or g_n >= N:
                    dest[] = 0.0
                else:
                    dest[] = (B + (g_k * ldb + g_n))[]



    # ============================================================
    # COMPUTE TILE (tiny inner GEMM)
    # ============================================================

    @staticmethod
    fn _compute_tile(
        smem_A: UnsafePointer[Float16],
        smem_B: UnsafePointer[Float16],
        mut acc: UnsafePointer[Float32],
        k_size: Int
    ):

        for k in range(k_size):
            for m in range(THREAD_M):
                for n in range(THREAD_N):

                    var a_offset = m * (TILE_K + SMEM_PAD) + k
                    var a_val: Float32 = (smem_A + a_offset)[].cast[DType.float32]()

                    var b_offset = k * (TILE_N + SMEM_PAD) + n
                    var b_val: Float32 = (smem_B + b_offset)[].cast[DType.float32]()



                    var idx = m * THREAD_N + n
                    var old = (acc + idx)[]
                    (acc + idx)[] = old + a_val * b_val


    # ============================================================
    # STORE TILE C
    # ============================================================

    @staticmethod
    fn _store_tile_C(
        mut C: UnsafePointer[Float32],
        acc: UnsafePointer[Float32],
        M: Int, N: Int, ldc: Int,
        block_m: Int, block_n: Int,
        alpha: Float32, beta: Float32
    ):

        for m in range(THREAD_M):
            var g_m = block_m + m
            if g_m >= M: continue

            for n in range(THREAD_N):

                var g_n = block_n + n
                if g_n >= N: continue

                var c_idx = g_m * ldc + g_n
                var acc_val = (acc + (m * THREAD_N + n))[]

                var dst = C + c_idx

                if beta == 0.0:
                    dst[] = alpha * acc_val
                else:
                    dst[] = alpha * acc_val + beta * dst[]




# koshiq
# """
# Optimized GEMM kernel for NVIDIA A100 SXM-4 40GB
# Computes C = alpha * A @ B + beta * C where A is MxK, B is KxN, C is MxN
# """

# from memory import memset_zero, memcpy
# from algorithm import vectorize, parallelize
# from sys import simdwidthof
# from math import min, max

# # A100 specific constants
# alias WARP_SIZE = 32
# alias WARPS_PER_BLOCK = 8
# alias BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK  # 256 threads per block

# # Tile dimensions optimized for A100 tensor cores (16x16x16 WMMA)
# alias TILE_M = 128
# alias TILE_N = 128
# alias TILE_K = 32

# # Warp tile dimensions
# alias WARP_M = 32
# alias WARP_N = 64

# # Thread block tile dimensions for register blocking
# alias THREAD_M = 8
# alias THREAD_N = 8

# # Shared memory padding to avoid bank conflicts
# alias SMEM_PAD = 8


# struct GEMMKernel:
    
#     @staticmethod
#     fn launch_kernel(
#         A: DTypePointer[DType.float16],
#         B: DTypePointer[DType.float16],
#         C: DTypePointer[DType.float32],
#         M: Int,
#         N: Int,
#         K: Int,
#         alpha: Float32,
#         beta: Float32,
#         lda: Int,
#         ldb: Int,
#         ldc: Int
#     ) raises:
#         let grid_m = (M + TILE_M - 1) // TILE_M
#         let grid_n = (N + TILE_N - 1) // TILE_N
#         let num_blocks = grid_m * grid_n
        
#         @parameter
#         fn process_block(block_idx: Int):
#             let block_m = (block_idx // grid_n) * TILE_M
#             let block_n = (block_idx % grid_n) * TILE_N
            
#             Self._gemm_block(
#                 A, B, C, M, N, K, alpha, beta,
#                 lda, ldb, ldc, block_m, block_n
#             )
        
#         parallelize[process_block](num_blocks, num_blocks)
    
#     @staticmethod
#     fn _gemm_block(
#         A: DTypePointer[DType.float16],
#         B: DTypePointer[DType.float16],
#         C: DTypePointer[DType.float32],
#         M: Int, N: Int, K: Int,
#         alpha: Float32, beta: Float32,
#         lda: Int, ldb: Int, ldc: Int,
#         block_m: Int, block_n: Int
#     ):
#         var smem_A = DTypePointer[DType.float16].alloc(TILE_M * (TILE_K + SMEM_PAD))
#         var smem_B = DTypePointer[DType.float16].alloc(TILE_K * (TILE_N + SMEM_PAD))
#         var acc = DTypePointer[DType.float32].alloc(THREAD_M * THREAD_N)
#         memset_zero(acc, THREAD_M * THREAD_N)
        
#         for k_tile in range(0, K, TILE_K):
#             let k_end = min(k_tile + TILE_K, K)
#             let k_size = k_end - k_tile
            
#             Self._load_tile_A(A, smem_A, M, K, lda, block_m, k_tile, k_size)
#             Self._load_tile_B(B, smem_B, K, N, ldb, k_tile, block_n, k_size)
#             Self._compute_tile_multiply(smem_A, smem_B, acc, k_size)
        
#         Self._store_tile_C(C, acc, M, N, ldc, block_m, block_n, alpha, beta)
        
#         smem_A.free()
#         smem_B.free()
#         acc.free()
    
#     @staticmethod
#     fn _load_tile_A(
#         A: DTypePointer[DType.float16],
#         smem_A: DTypePointer[DType.float16],
#         M: Int, K: Int, lda: Int,
#         block_m: Int, k_tile: Int, k_size: Int
#     ):
#         alias VECTOR_WIDTH = 8
        
#         for m_idx in range(TILE_M):
#             let global_m = block_m + m_idx
#             if global_m >= M:
#                 for k_idx in range(k_size):
#                     smem_A[m_idx * (TILE_K + SMEM_PAD) + k_idx] = 0.0
#             else:
#                 @parameter
#                 fn load_vector(k_vec: Int):
#                     let k_start = k_vec * VECTOR_WIDTH
#                     if k_start < k_size:
#                         for i in range(VECTOR_WIDTH):
#                             let k_idx = k_start + i
#                             if k_idx < k_size:
#                                 let global_k = k_tile + k_idx
#                                 let val = A[global_m * lda + global_k]
#                                 smem_A[m_idx * (TILE_K + SMEM_PAD) + k_idx] = val
                
#                 let num_vectors = (k_size + VECTOR_WIDTH - 1) // VECTOR_WIDTH
#                 for v in range(num_vectors):
#                     load_vector(v)
    
#     @staticmethod
#     fn _load_tile_B(
#         B: DTypePointer[DType.float16],
#         smem_B: DTypePointer[DType.float16],
#         K: Int, N: Int, ldb: Int,
#         k_tile: Int, block_n: Int, k_size: Int
#     ):
#         alias VECTOR_WIDTH = 8
        
#         for k_idx in range(k_size):
#             let global_k = k_tile + k_idx
#             if global_k >= K:
#                 for n_idx in range(TILE_N):
#                     smem_B[k_idx * (TILE_N + SMEM_PAD) + n_idx] = 0.0
#             else:
#                 @parameter
#                 fn load_vector(n_vec: Int):
#                     let n_start = n_vec * VECTOR_WIDTH
#                     if n_start < TILE_N:
#                         for i in range(VECTOR_WIDTH):
#                             let n_idx = n_start + i
#                             let global_n = block_n + n_idx
#                             if n_idx < TILE_N and global_n < N:
#                                 let val = B[global_k * ldb + global_n]
#                                 smem_B[k_idx * (TILE_N + SMEM_PAD) + n_idx] = val
                
#                 let num_vectors = (TILE_N + VECTOR_WIDTH - 1) // VECTOR_WIDTH
#                 for v in range(num_vectors):
#                     load_vector(v)
    
#     @staticmethod
#     fn _compute_tile_multiply(
#         smem_A: DTypePointer[DType.float16],
#         smem_B: DTypePointer[DType.float16],
#         acc: DTypePointer[DType.float32],
#         k_size: Int
#     ):
#         for k_idx in range(k_size):
#             for m_idx in range(THREAD_M):
#                 for n_idx in range(THREAD_N):
#                     let a_val = smem_A[m_idx * (TILE_K + SMEM_PAD) + k_idx].cast[DType.float32]()
#                     let b_val = smem_B[k_idx * (TILE_N + SMEM_PAD) + n_idx].cast[DType.float32]()
#                     acc[m_idx * THREAD_N + n_idx] += a_val * b_val
    
#     @staticmethod
#     fn _store_tile_C(
#         C: DTypePointer[DType.float32],
#         acc: DTypePointer[DType.float32],
#         M: Int, N: Int, ldc: Int,
#         block_m: Int, block_n: Int,
#         alpha: Float32, beta: Float32
#     ):
#         for m_idx in range(THREAD_M):
#             let global_m = block_m + m_idx
#             if global_m >= M:
#                 continue
            
#             for n_idx in range(THREAD_N):
#                 let global_n = block_n + n_idx
#                 if global_n >= N:
#                     continue
                
#                 let c_idx = global_m * ldc + global_n
#                 let acc_val = acc[m_idx * THREAD_N + n_idx]
                
#                 if beta == 0.0:
#                     C[c_idx] = alpha * acc_val
#                 else:
#                     C[c_idx] = alpha * acc_val + beta * C[c_idx]