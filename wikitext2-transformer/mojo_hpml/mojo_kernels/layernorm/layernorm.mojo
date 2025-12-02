import math
import compiler
from max.tensor import Tensor, InputTensor, OutputTensor
from runtime.asyncrt import DeviceContextPtr

# Register the kernel name "layer_norm" so Python side can call it
@compiler.register("layer_norm")
struct LayerNorm:

    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.float32, rank=3],
        input: InputTensor[dtype=DType.float32, rank=3],
        weight: InputTensor[dtype=DType.float32, rank=1],
        bias: InputTensor[dtype=DType.float32, rank=1],
        eps: Float32,
        ctx: DeviceContextPtr,
    ) raises:

        # Shapes: [batch, seq, hidden]
        let B = input.dim(0)
        let S = input.dim(1)
        let H = input.dim(2)

        # Reference LayerNorm implementation:
        # for each (b, s) row, normalize over the hidden dimension H
        for b in range(B):
            for s in range(S):

                # 1) Compute mean over hidden dim
                var mean: Float32 = 0.0
                for h in range(H):
                    mean += input[b, s, h]
                mean /= Float32(H)

                # 2) Compute variance over hidden dim
                var var_acc: Float32 = 0.0
                for h in range(H):
                    let diff = input[b, s, h] - mean
                    var_acc += diff * diff
                let variance = var_acc / Float32(H)
                let inv_std = math.rsqrt(variance + eps)

                # 3) Normalize, then apply affine transform (gamma, beta)
                for h in range(H):
                    let x = input[b, s, h]
                    let normalized = (x - mean) * inv_std
                    let gamma = weight[h]
                    let beta = bias[h]
                    output[b, s, h] = normalized * gamma + beta



# import math
# import compiler
# from max.tensor import Tensor, InputTensor, OutputTensor, foreach
# from runtime.asyncrt import DeviceContextPtr
# from utils.index import IndexList

# # Register the kernel name "layer_norm"
# @compiler.register("layer_norm")
# struct LayerNorm:

#     @staticmethod
#     fn execute[
#         target: StaticString,
#     ](
#         output: OutputTensor[dtype=DType.float32, rank=3],
#         input: InputTensor[dtype=DType.float32, rank=3],
#         weight: InputTensor[dtype=DType.float32, rank=1],
#         bias: InputTensor[dtype=DType.float32, rank=1],
#         eps: Float32,
#         ctx: DeviceContextPtr,
#     ) raises:
#         var batch = input.dim(0)
#         var seq = input.dim(1)
#         var hidden = input.dim(2)

#         @parameter
#         fn compute_row[idx: IndexList[2]]:
#             let b = idx[0]
#             let s = idx[1]

#             # Mean
#             var sum: Float32 = 0.0
#             for i in range(hidden):
#                 sum += input[b, s, i]
#             var mean = sum / Float32(hidden)

#             # Variance
#             var sum_sq_diff: Float32 = 0.0
#             for i in range(hidden):
#                 var diff = input[b, s, i] - mean
#                 sum_sq_diff += diff * diff
#             var variance = sum_sq_diff / Float32(hidden)
#             var inv_std = math.rsqrt(variance + eps)

#             # Normalize + Affine
#             for i in range(hidden):
#                 var val = input[b, s, i]
#                 var w = weight[i]
#                 var beta = bias[i]
#                 var normalized = (val - mean) * inv_std
#                 output[b, s, i] = normalized * w + beta

#         # foreach[compute_row, target=target](IndexList[2](batch, seq), ctx)
#         foreach[compute_row, target=target, runtime=true](batch, seq, ctx)









# import math
# import compiler

# # from max.tensor import Tensor, InputTensor, OutputTensor, foreach
# from runtime.asyncrt import DeviceContextPtr
# from utils.index import IndexList

# # Register the kernel so MAX can find it
# @compiler.register("layer_norm")
# struct LayerNorm:

#     @staticmethod
#     fn execute[
#         target: StaticString,
#     ](
#         output: OutputTensor[dtype=DType.float32, rank=3],
#         input: InputTensor[dtype=DType.float32, rank=3],
#         weight: InputTensor[dtype=DType.float32, rank=1],
#         bias: InputTensor[dtype=DType.float32, rank=1],
#         eps: Float32,
#         ctx: DeviceContextPtr,
#     ) raises:

#         var batch = input.dim(0)
#         var seq = input.dim(1)
#         var hidden = input.dim(2)

#         @parameter
#         fn compute_row[idx: IndexList[2]]:
#             let b = idx[0]
#             let s = idx[1]

#             # Compute mean
#             var sum: Float32 = 0.0
#             for i in range(hidden):
#                 sum += input[b, s, i]

#             var mean = sum / Float32(hidden)

#             # Compute variance
#             var sum_sq_diff: Float32 = 0.0
#             for i in range(hidden):
#                 var diff = input[b, s, i] - mean
#                 sum_sq_diff += diff * diff

#             var variance = sum_sq_diff / Float32(hidden)
#             var inv_std = math.rsqrt(variance + eps)

#             # Normalize + affine transform
#             for i in range(hidden):
#                 var val = input[b, s, i]
#                 var w = weight[i]
#                 var beta = bias[i]

#                 var normalized = (val - mean) * inv_std
#                 output[b, s, i] = normalized * w + beta

#         foreach[compute_row, target=target](IndexList[2](batch, seq), ctx)
