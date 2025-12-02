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
