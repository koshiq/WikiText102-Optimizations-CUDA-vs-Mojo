"""
Mojo MAX Operations using pure Python Graph API.
This uses MAX's Graph API to compile and execute kernels without .mojopkg files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =====================================================================================
# MAX Graph API CHECK
# =====================================================================================
try:
    from max import engine
    from max.graph import Graph, TensorType, ops, Dim, DeviceRef
    from max.dtype import DType as MAXDType
    GRAPH_AVAILABLE = True
    print("[mojo_ops_graph] MAX Graph API detected ✓")
except Exception as e:
    GRAPH_AVAILABLE = False
    print(f"[mojo_ops_graph] MAX Graph API unavailable → fallback mode: {e}")


# =====================================================================================
# GEMM Graph
# =====================================================================================
def create_gemm_graph(m, k, n):
    """Create a GEMM graph using MAX Graph API."""
    if not GRAPH_AVAILABLE:
        return None

    try:
        with Graph("gemm", input_types=[
            TensorType(MAXDType.float32, shape=[m, k], device=DeviceRef.GPU()),  # A
            TensorType(MAXDType.float32, shape=[k, n], device=DeviceRef.GPU()),  # B
        ]) as graph:
            a, b = graph.inputs
            # C = A @ B
            result = ops.matmul(a, b)
            graph.output(result)

        from max.driver import Device, cpu, gpu
        # Try to get GPU device, fallback to CPU
        try:
            device = gpu()
        except:
            device = cpu()
        session = engine.InferenceSession(devices=[device])
        model = session.load(graph)
        print(f"[GEMM Graph] Compiled successfully for [{m}x{k}] @ [{k}x{n}]")
        return model
    except Exception as e:
        print(f"[GEMM Graph] Compilation failed: {e}")
        return None


# =====================================================================================
# Softmax Graph
# =====================================================================================
def create_softmax_graph(batch_size, seq_len):
    """Create a Softmax graph using MAX Graph API."""
    if not GRAPH_AVAILABLE:
        return None

    try:
        with Graph("softmax", input_types=[
            TensorType(MAXDType.float32, shape=[batch_size, seq_len], device=DeviceRef.GPU())
        ]) as graph:
            x = graph.inputs[0]
            # Softmax along last dimension
            result = ops.softmax(x, axis=-1)
            graph.output(result)

        from max.driver import cpu, gpu
        # Try to get GPU device, fallback to CPU
        try:
            device = gpu()
        except:
            device = cpu()
        session = engine.InferenceSession(devices=[device])
        model = session.load(graph)
        print(f"[Softmax Graph] Compiled successfully for [{batch_size}x{seq_len}]")
        return model
    except Exception as e:
        print(f"[Softmax Graph] Compilation failed: {e}")
        return None


# =====================================================================================
# LayerNorm Graph
# =====================================================================================
def create_layernorm_graph(batch_size, seq_len, hidden_size, eps=1e-5):
    """Create a LayerNorm graph using MAX Graph API."""
    if not GRAPH_AVAILABLE:
        return None

    try:
        with Graph("layernorm", input_types=[
            TensorType(MAXDType.float32, shape=[batch_size, seq_len, hidden_size], device=DeviceRef.GPU()),  # input
            TensorType(MAXDType.float32, shape=[hidden_size], device=DeviceRef.GPU()),  # gamma
            TensorType(MAXDType.float32, shape=[hidden_size], device=DeviceRef.GPU()),  # beta
        ]) as graph:
            x, gamma, beta = graph.inputs

            # Use built-in layer_norm if available, otherwise use basic ops
            try:
                # Try using layer_norm directly
                result = ops.layer_norm(x, gamma, beta, eps=eps)
            except AttributeError:
                # Fallback: use manual computation with reduce_sum
                # mean = sum(x) / hidden_size
                sum_x = ops.reduce_sum(x, axis=-1, keepdims=True)
                mean = ops.div(sum_x, float(hidden_size))

                # variance = sum((x - mean)^2) / hidden_size
                x_centered = ops.sub(x, mean)
                x_squared = ops.mul(x_centered, x_centered)
                sum_squared = ops.reduce_sum(x_squared, axis=-1, keepdims=True)
                variance = ops.div(sum_squared, float(hidden_size))

                # normalized = (x - mean) / sqrt(variance + eps)
                denom = ops.sqrt(ops.add(variance, eps))
                normalized = ops.div(x_centered, denom)

                # result = normalized * gamma + beta
                result = ops.add(ops.mul(normalized, gamma), beta)

            graph.output(result)

        from max.driver import cpu, gpu
        # Try to get GPU device, fallback to CPU
        try:
            device = gpu()
        except:
            device = cpu()
        session = engine.InferenceSession(devices=[device])
        model = session.load(graph)
        print(f"[LayerNorm Graph] Compiled successfully for [{batch_size}x{seq_len}x{hidden_size}]")
        return model
    except Exception as e:
        print(f"[LayerNorm Graph] Compilation failed: {e}")
        return None


# =====================================================================================
# Mojo GEMM using Graph API
# =====================================================================================
class MojoGEMM(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.graph_model = None
        self.use_graph = False

        if GRAPH_AVAILABLE:
            try:
                # We'll compile the graph on first forward pass with actual shapes
                self.use_graph = True
                print("[MojoGEMM] Will use Graph API ✓")
            except Exception as e:
                print(f"[MojoGEMM] Graph setup failed → fallback: {e}")

    def forward(self, x):
        if not self.use_graph or not GRAPH_AVAILABLE:
            return F.linear(x, self.weight, self.bias)

        # Reshape input to 2D for matmul
        original_shape = x.shape
        x_2d = x.view(-1, self.in_features)

        try:
            # Lazy compilation on first run
            if self.graph_model is None:
                batch_size = x_2d.shape[0]
                self.graph_model = create_gemm_graph(batch_size, self.in_features, self.out_features)
                if self.graph_model is None:
                    self.use_graph = False
                    return F.linear(x, self.weight, self.bias)

            # Convert to numpy
            x_np = x_2d.detach().cpu().numpy().astype(np.float32)
            w_np = self.weight.t().detach().cpu().numpy().astype(np.float32)

            # Execute graph
            result_np = self.graph_model.execute(x_np, w_np)[0]

            # Convert back to torch
            result = torch.from_numpy(result_np).to(x.device)

            # Add bias if present
            if self.bias is not None:
                result = result + self.bias

            # Reshape back to original
            return result.view(*original_shape[:-1], self.out_features)

        except Exception as e:
            print(f"[MojoGEMM] Graph execution failed, using fallback: {e}")
            self.use_graph = False
            return F.linear(x, self.weight, self.bias)


# =====================================================================================
# Mojo Softmax using Graph API
# =====================================================================================
class MojoSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.graph_model = None
        self.use_graph = False

        if GRAPH_AVAILABLE:
            try:
                self.use_graph = True
                print("[MojoSoftmax] Will use Graph API ✓")
            except Exception as e:
                print(f"[MojoSoftmax] Graph setup failed → fallback: {e}")

    def forward(self, x):
        if not self.use_graph or not GRAPH_AVAILABLE:
            return F.softmax(x, dim=self.dim)

        try:
            # Reshape to 2D for graph execution
            original_shape = x.shape
            if len(x.shape) == 3:
                # [seq_len, batch_size, hidden_size] -> [seq_len * batch_size, hidden_size]
                x_2d = x.reshape(-1, x.shape[-1])
            elif len(x.shape) == 2:
                x_2d = x
            else:
                return F.softmax(x, dim=self.dim)

            # Lazy compilation
            if self.graph_model is None:
                batch_size, seq_len = x_2d.shape
                self.graph_model = create_softmax_graph(batch_size, seq_len)
                if self.graph_model is None:
                    self.use_graph = False
                    return F.softmax(x, dim=self.dim)

            # Convert to numpy
            x_np = x_2d.detach().cpu().numpy().astype(np.float32)

            # Execute graph
            result_np = self.graph_model.execute(x_np)[0]

            # Convert back to torch and reshape
            result = torch.from_numpy(result_np).to(x.device)
            return result.reshape(original_shape)

        except Exception as e:
            print(f"[MojoSoftmax] Graph execution failed, using fallback: {e}")
            self.use_graph = False
            return F.softmax(x, dim=self.dim)


class MojoLogSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Use PyTorch fallback for now
        return F.log_softmax(x, dim=self.dim)


# =====================================================================================
# Mojo LayerNorm using Graph API
# =====================================================================================
class MojoLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.graph_model = None
        self.use_graph = False

        if GRAPH_AVAILABLE:
            try:
                self.use_graph = True
                print("[MojoLayerNorm] Will use Graph API ✓")
            except Exception as e:
                print(f"[MojoLayerNorm] Graph setup failed → fallback: {e}")

    def forward(self, x):
        if not self.use_graph or not GRAPH_AVAILABLE:
            return F.layer_norm(x, (self.normalized_shape,), self.weight, self.bias, self.eps)

        try:
            # Ensure 3D shape
            if len(x.shape) != 3:
                return F.layer_norm(x, (self.normalized_shape,), self.weight, self.bias, self.eps)

            # Lazy compilation with actual shape
            if self.graph_model is None:
                seq_len, batch_size, hidden_size = x.shape  # [35, 20, 200]
                self.graph_model = create_layernorm_graph(seq_len, batch_size, hidden_size, self.eps)
                if self.graph_model is None:
                    self.use_graph = False
                    return F.layer_norm(x, (self.normalized_shape,), self.weight, self.bias, self.eps)

            # Convert to numpy
            x_np = x.detach().cpu().numpy().astype(np.float32)
            gamma_np = self.weight.detach().cpu().numpy().astype(np.float32)
            beta_np = self.bias.detach().cpu().numpy().astype(np.float32)

            # Execute graph
            result_np = self.graph_model.execute(x_np, gamma_np, beta_np)[0]

            # Convert back to torch
            return torch.from_numpy(result_np).to(x.device)

        except Exception as e:
            print(f"[MojoLayerNorm] Graph execution failed, using fallback: {e}")
            self.use_graph = False
            return F.layer_norm(x, (self.normalized_shape,), self.weight, self.bias, self.eps)


# =====================================================================================
# Benchmark Helper
# =====================================================================================
def benchmark_op(op_name, mojo_op, pytorch_op, input_tensor, warmup=10, iterations=100):
    import time
    device = input_tensor.device

    for _ in range(warmup):
        mojo_op(input_tensor)
        pytorch_op(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        mojo_op(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    mojo_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        pytorch_op(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    torch_time = time.perf_counter() - start

    speed = torch_time / mojo_time if mojo_time > 0 else 0

    return {
        "operation": op_name,
        "mojo_time_ms": mojo_time * 1000 / iterations,
        "pytorch_time_ms": torch_time * 1000 / iterations,
        "speedup": speed,
        "faster": "Mojo" if speed > 1.0 else "PyTorch",
    }


__all__ = [
    "MojoLayerNorm",
    "MojoGEMM",
    "MojoSoftmax",
    "MojoLogSoftmax",
    "benchmark_op",
    "GRAPH_AVAILABLE",
]
