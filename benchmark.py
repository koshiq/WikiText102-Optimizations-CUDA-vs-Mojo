"""
Comprehensive Benchmark: PyTorch vs CUDA-Optimized vs MAX Graph vs Mojo Kernels
Compares all implementations on RTX 4070
"""

import torch
import torch.nn as nn
import time
import numpy as np
import sys
import os

# Import models
from transformer import TransformerModel as PyTorchModel
from customCudaKernel_transformer import TransformerModel as CUDAModel, get_model_info

# Try to import MAX Graph API model
MAX_GRAPH_AVAILABLE = False
try:
    from maxGraph_transformer import MaxLinear, MaxLogSoftmax, MaxLayerNorm
    from maxGraph_transformer import TransformerModel as MaxGraphModel, get_model_info as get_max_info
    MAX_GRAPH_AVAILABLE = True
    print("[OK] MAX Graph API model loaded")
except ImportError as e:
    print(f"MAX Graph API model not available: {e}")


def benchmark_forward(model, data, num_iter=100, warmup=10):
    """Benchmark forward pass only"""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iter):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return np.mean(times), np.std(times), np.min(times), np.max(times)


def benchmark_training(model, data, targets, criterion, num_iter=50, warmup=5):
    """Benchmark full training step (forward + backward)"""
    model.train()

    # Warmup
    for _ in range(warmup):
        model.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, output.size(-1)), targets)
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iter):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        model.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, output.size(-1)), targets)
        loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return np.mean(times), np.std(times), np.min(times), np.max(times)


def print_header(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_section(title):
    print(f"\n{'-' * 80}")
    print(f" {title}")
    print(f"{'-' * 80}")

def benchmark_op(op_name: str, pytorch_op, custom_op, input_tensor: torch.Tensor,
                 warmup: int = 10, iterations: int = 100):
    """
    Benchmark a custom operation against its PyTorch CUDA baseline.
    """
    device = input_tensor.device

    # Warmup
    for _ in range(warmup):
        _ = pytorch_op(input_tensor)
        _ = custom_op(input_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(iterations):
        _ = pytorch_op(input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / iterations

    # Benchmark Custom
    start = time.perf_counter()
    for _ in range(iterations):
        _ = custom_op(input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    custom_time = (time.perf_counter() - start) / iterations

    speedup = pytorch_time / custom_time
    faster = "Custom" if speedup > 1.0 else "PyTorch"

    print(f"  Benchmarking {op_name}:")
    print(f"    PyTorch: {pytorch_time * 1000:.4f} ms | Custom: {custom_time * 1000:.4f} ms | Speedup: {speedup:.2f}x ({faster} faster)")

    return pytorch_time, custom_time


def main():
    print_header("WikiText-2 Transformer: Complete Performance Comparison")
    print("Comparing: PyTorch Standard | CUDA-Optimized | MAX Graph API ")

    # Check CUDA
    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA not available! This benchmark requires a GPU.")
        return

    device = torch.device('cuda')
    print(f"\nSystem Information:")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Version: {torch.version.cuda}")

    # Check what's available
    cuda_ops_enabled = get_model_info()['cuda_ops_available']

    print(f"\nAvailable Implementations:")
    print(f"   [OK] PyTorch Standard (baseline)")
    print(f"   [{'OK' if cuda_ops_enabled else 'FAIL'}] CUDA-Optimized Kernels {'(ENABLED)' if cuda_ops_enabled else '(disabled)'}")
    print(f"   [{'OK' if MAX_GRAPH_AVAILABLE else 'FAIL'}] MAX Graph API {'(ENABLED)' if MAX_GRAPH_AVAILABLE else '(not installed)'}")

    if MAX_GRAPH_AVAILABLE:
        max_info = get_max_info()
        print(f"       Backend: {max_info['backend']}, Expected speedup: {max_info['expected_speedup']}")

    # Model configuration
    print_section("Model Configuration (WikiText-2)")

    ntoken = 33278      # WikiText-2 vocabulary
    ninp = 200          # Embedding dimension
    nhead = 2           # Attention heads
    nhid = 200          # Feedforward dimension
    nlayers = 4         # Encoder layers
    dropout = 0.2
    batch_size = 20
    seq_len = 35

    print(f"   Vocabulary Size: {ntoken:,}")
    print(f"   Embedding Dim:   {ninp}")
    print(f"   Attention Heads: {nhead}")
    print(f"   Hidden Dim:      {nhid}")
    print(f"   Num Layers:      {nlayers}")
    print(f"   Batch Size:      {batch_size}")
    print(f"   Sequence Length: {seq_len}")
    print(f"   Tokens/Batch:    {batch_size * seq_len:,}")

    # Create test data
    input_data = torch.randint(0, ntoken, (seq_len, batch_size), device=device)
    target_data = torch.randint(0, ntoken, (seq_len * batch_size,), device=device)
    criterion = nn.NLLLoss()

    # Build models
    print_section("Building Models")

    models = {}
    results = {}
    model_num = 1
    total_models = 1 + (1 if cuda_ops_enabled else 0) + (1 if MAX_GRAPH_AVAILABLE else 0)

    # 1. PyTorch Standard
    print(f"\n   [{model_num}/{total_models}] PyTorch Standard Model...")
    pytorch_model = PyTorchModel(ntoken, ninp, nhead, nhid, nlayers, dropout).to(device)
    pytorch_params = sum(p.numel() for p in pytorch_model.parameters())
    models['PyTorch'] = pytorch_model
    print(f"         Parameters: {pytorch_params:,}")
    model_num += 1

    # 2. CUDA-Optimized
    if cuda_ops_enabled:
        print(f"\n   [{model_num}/{total_models}] CUDA-Optimized Model...")
        cuda_model = CUDAModel(ntoken, ninp, nhead, nhid, nlayers, dropout).to(device)
        # Ensure internal tensors of custom pybind modules are moved to device
        try:
            import customCudaKernel_transformer as cct
            if hasattr(cct, 'move_custom_modules_to_device'):
                cct.move_custom_modules_to_device(cuda_model, device)
        except Exception:
            pass
        cuda_params = sum(p.numel() for p in cuda_model.parameters())
        models['CUDA-Kernels'] = cuda_model
        print(f"         Parameters: {cuda_params:,}")
        model_num += 1
    else:
        print(f"\n   [SKIP] CUDA-Optimized Model... (ops not built)")

    # 3. MAX Graph API
    if MAX_GRAPH_AVAILABLE:
        print(f"\n   [{model_num}/{total_models}] MAX Graph API Model...")
        try:
            max_graph_model = MaxGraphModel(ntoken, ninp, nhead, nhid, nlayers, dropout).to(device)
            max_graph_params = sum(p.numel() for p in max_graph_model.parameters())
            models['MAX-Graph'] = max_graph_model
            print(f"         Parameters: {max_graph_params:,}")
            model_num += 1
        except Exception as e:
            print(f"         [ERROR] Error loading MAX Graph model: {e}")
    else:
        print(f"\n   [SKIP] MAX Graph API Model... (not installed)")

    # Individual Operation Benchmarks
    print_header("BENCHMARK 1: INDIVIDUAL OPERATION BENCHMARKS")
    
    # GEMM Benchmark
    if MAX_GRAPH_AVAILABLE:
        gemm_input = torch.randn(batch_size * seq_len, ninp, device=device)
        pytorch_gemm = nn.Linear(ninp, nhid).to(device)
        if 'MAX-Graph' in models:
            max_gemm = MaxLinear(ninp, nhid).to(device)
            benchmark_op("GEMM (MAX vs PyTorch)", pytorch_gemm, max_gemm, gemm_input)

    # LogSoftmax Benchmark
    if MAX_GRAPH_AVAILABLE:
        softmax_input = torch.randn(batch_size * seq_len, ntoken, device=device)
        pytorch_softmax = nn.LogSoftmax(dim=-1).to(device)
        if 'MAX-Graph' in models:
            max_softmax = MaxLogSoftmax(dim=-1).to(device)
            benchmark_op("LogSoftmax (MAX vs PyTorch)", pytorch_softmax, max_softmax, softmax_input)

    # Run benchmarks
    print_header("BENCHMARK 2: INFERENCE (Forward Pass Only)")

    for name, model in models.items():
        print(f"\n   Testing {name} Model...")
        try:
            mean, std, min_t, max_t = benchmark_forward(model, input_data, num_iter=100)
            throughput = (batch_size * seq_len * 1000) / mean
            results[f'{name}_inference'] = {
                'mean': mean,
                'std': std,
                'min': min_t,
                'max': max_t,
                'throughput': throughput
            }
            print(f"   Time: {mean:.4f} ± {std:.4f} ms (min: {min_t:.4f}, max: {max_t:.4f})")
            print(f"   Throughput: {throughput:,.0f} tokens/sec")
        except Exception as e:
            print(f"   [ERROR] Error: {e}")

    # Training benchmark
    print_header("BENCHMARK 3: TRAINING (Forward + Backward)")

    # We can't train the MAX model, so we filter it out here.
    trainable_models = {k: v for k, v in models.items() if "MAX" not in k}

    for name, model in trainable_models.items():
        print(f"\n   Testing {name} Model...")
        try:
            mean, std, min_t, max_t = benchmark_training(
                model, input_data, target_data, criterion, num_iter=50
            )
            throughput = 1000 / mean
            results[f'{name}_training'] = {
                'mean': mean,
                'std': std,
                'min': min_t,
                'max': max_t,
                'throughput': throughput
            }
            print(f"   Time: {mean:.4f} ± {std:.4f} ms (min: {min_t:.4f}, max: {max_t:.4f})")
            print(f"   Throughput: {throughput:.2f} batches/sec")
        except Exception as e:
            print(f"   [ERROR] Error: {e}")

    # Memory comparison
    print_header("MEMORY USAGE")

    for name, model in models.items():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(input_data)
        mem = torch.cuda.max_memory_allocated() / 1024**3
        results[f'{name}_memory'] = mem
        print(f"   {name:15s}: {mem:.3f} GB")

    # Summary comparison
    print_header("PERFORMANCE SUMMARY")

    baseline_name = 'PyTorch'
    if f'{baseline_name}_inference' in results and f'{baseline_name}_training' in results:
        baseline_inf = results[f'{baseline_name}_inference']['mean']
        baseline_train = results[f'{baseline_name}_training']['mean']

        print(f"\n{'Implementation':<15} {'Inference':<20} {'Training':<20} {'Memory':<15}")
        print("-" * 80)

        for name in models.keys():
            if f'{name}_inference' in results:
                inf_time = results[f'{name}_inference']['mean']
                train_time = results.get(f'{name}_training', {}).get('mean', float('nan'))
                memory = results[f'{name}_memory']

                # Calculate speedup
                inf_speedup = baseline_inf / inf_time if name != baseline_name else 1.0
                train_speedup = baseline_train / train_time if name != baseline_name else 1.0

                inf_str = f"{inf_time:.3f}ms ({inf_speedup:.2f}x)"
                if not np.isnan(train_time):
                    train_str = f"{train_time:.3f}ms ({train_speedup:.2f}x)"
                else:
                    train_str = "N/A (Inference Only)"
                mem_str = f"{memory:.3f} GB"

                print(f"{name:<15} {inf_str:<20} {train_str:<20} {mem_str:<15}")

    # Winner analysis
    print_header("ANALYSIS")

    if len(results) > 2:
        # Find fastest for inference
        inf_times = {name: results[f'{name}_inference']['mean']
                    for name in models.keys() if f'{name}_inference' in results}
        fastest_inf = min(inf_times, key=inf_times.get)
        fastest_inf_time = inf_times[fastest_inf]
        baseline_inf_time = inf_times.get(baseline_name, fastest_inf_time)
        inf_improvement = ((baseline_inf_time / fastest_inf_time) - 1) * 100

        # Find fastest for training
        train_times = {name: results[f'{name}_training']['mean']
                      for name in trainable_models.keys() if f'{name}_training' in results}
        fastest_train = min(train_times, key=train_times.get)
        fastest_train_time = train_times[fastest_train]
        baseline_train_time = train_times.get(baseline_name, fastest_train_time)
        train_improvement = ((baseline_train_time / fastest_train_time) - 1) * 100

        print(f"\nFastest Inference:  {fastest_inf} ({inf_improvement:+.1f}% vs PyTorch)")
        print(f"Fastest Training:   {fastest_train} ({train_improvement:+.1f}% vs PyTorch)")

        # Time savings estimate
        print(f"\nTime Savings for 10,000 Iterations:")
        for name in models.keys():
            if name != baseline_name and f'{name}_training' in results:
                saved = (baseline_train_time - train_times[name]) * 10000 / 1000 / 60
                print(f"   {name}: {saved:.1f} minutes")

if __name__ == '__main__':
    main()
