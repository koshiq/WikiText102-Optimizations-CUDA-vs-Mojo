"""
MAX Graph implementation of Transformer inference.
Uses MAX's built-in optimized ops (matmul, softmax, layer_norm).

This is Phase 2 of your project: MAX Graph Baseline.
"""

import numpy as np
import torch
from max.graph import Graph, TensorType, ops
from max.dtype import DType
from max.graph.ops import DeviceRef
import math


def build_attention_layer_graph(d_model, nhead, device_type="gpu"):
    """
    Build a single attention layer using MAX Graph API.
    Uses MAX's optimized matmul and softmax operations.
    """
    device = DeviceRef.GPU() if device_type == "gpu" else DeviceRef.CPU()
    d_k = d_model // nhead

    # Input: (seq_len, batch_size, d_model)
    # For simplicity, using static shapes. You can make these symbolic later.
    input_type = TensorType(DType.float32, ("seq_len", "batch", d_model), device=device)

    with Graph("attention_layer", input_types=[input_type]) as graph:
        x = graph.inputs[0]

        # Load weight matrices from external constants (loaded from PyTorch weights)
        # Q, K, V projection weights: (d_model, d_model)
        W_q = ops.constant_external("attn.in_proj_weight_q",
                                    TensorType(DType.float32, (d_model, d_model), device=device))
        W_k = ops.constant_external("attn.in_proj_weight_k",
                                    TensorType(DType.float32, (d_model, d_model), device=device))
        W_v = ops.constant_external("attn.in_proj_weight_v",
                                    TensorType(DType.float32, (d_model, d_model), device=device))

        # Project to Q, K, V using MAX's optimized matmul
        Q = ops.matmul(x, W_q)  # (seq_len, batch, d_model)
        K = ops.matmul(x, W_k)
        V = ops.matmul(x, W_v)

        # Reshape for multi-head attention: (seq_len, batch, nhead, d_k)
        # Then permute to (batch, nhead, seq_len, d_k)
        Q = ops.reshape(Q, ("seq_len", "batch", nhead, d_k))
        K = ops.reshape(K, ("seq_len", "batch", nhead, d_k))
        V = ops.reshape(V, ("seq_len", "batch", nhead, d_k))

        # Transpose: (batch, nhead, seq_len, d_k)
        Q = ops.transpose(Q, (1, 2, 0, 3))
        K = ops.transpose(K, (1, 2, 0, 3))
        V = ops.transpose(V, (1, 2, 0, 3))

        # Attention scores: Q @ K^T
        # K^T: (batch, nhead, d_k, seq_len)
        K_T = ops.transpose(K, (0, 1, 3, 2))
        scores = ops.matmul(Q, K_T)  # (batch, nhead, seq_len, seq_len)

        # Scale
        scale = ops.constant(1.0 / math.sqrt(d_k), DType.float32, device=device)
        scores = ops.mul(scores, scale)

        # Softmax over last dimension - MAX's optimized implementation
        attn_weights = ops.softmax(scores, axis=-1)

        # Apply attention to values
        attn_output = ops.matmul(attn_weights, V)  # (batch, nhead, seq_len, d_k)

        # Reshape back: (seq_len, batch, d_model)
        attn_output = ops.transpose(attn_output, (2, 0, 1, 3))
        attn_output = ops.reshape(attn_output, ("seq_len", "batch", d_model))

        # Output projection
        W_out = ops.constant_external("attn.out_proj_weight",
                                     TensorType(DType.float32, (d_model, d_model), device=device))
        output = ops.matmul(attn_output, W_out)

        graph.output(output)

    return graph


def build_ffn_graph(d_model, d_ff, device_type="gpu"):
    """
    Build feed-forward network using MAX Graph.
    Uses MAX's optimized matmul and gelu.
    """
    device = DeviceRef.GPU() if device_type == "gpu" else DeviceRef.CPU()

    input_type = TensorType(DType.float32, ("seq_len", "batch", d_model), device=device)

    with Graph("ffn", input_types=[input_type]) as graph:
        x = graph.inputs[0]

        # First linear layer
        W1 = ops.constant_external("ffn.linear1_weight",
                                   TensorType(DType.float32, (d_model, d_ff), device=device))
        b1 = ops.constant_external("ffn.linear1_bias",
                                   TensorType(DType.float32, (d_ff,), device=device))

        hidden = ops.matmul(x, W1)  # MAX's optimized GEMM
        hidden = ops.add(hidden, b1)
        hidden = ops.gelu(hidden)  # MAX's optimized GELU

        # Second linear layer
        W2 = ops.constant_external("ffn.linear2_weight",
                                   TensorType(DType.float32, (d_ff, d_model), device=device))
        b2 = ops.constant_external("ffn.linear2_bias",
                                   TensorType(DType.float32, (d_model,), device=device))

        output = ops.matmul(hidden, W2)
        output = ops.add(output, b2)

        graph.output(output)

    return graph


def build_layer_norm_graph(d_model, eps=1e-5, device_type="gpu"):
    """
    Build LayerNorm using MAX's optimized layer_norm op.
    """
    device = DeviceRef.GPU() if device_type == "gpu" else DeviceRef.CPU()

    input_type = TensorType(DType.float32, ("seq_len", "batch", d_model), device=device)

    with Graph("layer_norm", input_types=[input_type]) as graph:
        x = graph.inputs[0]

        gamma = ops.constant_external("ln.weight",
                                     TensorType(DType.float32, (d_model,), device=device))
        beta = ops.constant_external("ln.bias",
                                    TensorType(DType.float32, (d_model,), device=device))

        # MAX's optimized LayerNorm
        output = ops.layer_norm(x, gamma, beta, epsilon=eps)

        graph.output(output)

    return graph


def compile_and_benchmark_attention(d_model=200, nhead=2, seq_len=35, batch_size=20):
    """
    Example: Compile and benchmark a single attention layer.
    This demonstrates Phase 2 of your project.
    """
    print("Building MAX Graph for Attention Layer...")
    graph = build_attention_layer_graph(d_model, nhead, device_type="gpu")

    print("Compiling graph...")
    # The graph compilation will optimize matmul and softmax operations
    session = graph.compile()

    print("Creating dummy input...")
    # Create input tensor
    input_data = np.random.randn(seq_len, batch_size, d_model).astype(np.float32)

    print("Running inference...")
    output = session.run(input_data)

    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ MAX Graph execution successful!")

    return session


if __name__ == "__main__":
    print("="*70)
    print("MAX Graph Transformer Baseline (Phase 2)")
    print("="*70)
    print()
    print("This uses MAX's built-in optimized operations:")
    print("  - ops.matmul()    -> Optimized GEMM")
    print("  - ops.softmax()   -> Optimized Softmax")
    print("  - ops.layer_norm() -> Optimized LayerNorm")
    print()
    print("No custom .mojopkg kernels needed!")
    print("="*70)
    print()

    try:
        compile_and_benchmark_attention()
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: You need to provide actual PyTorch weights")
        print("Run the weight extraction script first!")
