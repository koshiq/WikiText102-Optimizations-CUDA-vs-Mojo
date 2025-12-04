"""
Benchmark PyTorch CUDA vs MAX Graph using @max.torch.graph_op decorator.
This is the CORRECT and SIMPLE approach for your project!
"""

import torch
import time
import json
import max.torch
from max.graph import ops
from max.dtype import DType


# Define MAX Graph operations using the decorator
@max.torch.graph_op
def max_matmul(A, B):
    """MAX Graph matrix multiplication"""
    return ops.matmul(A, B)


@max.torch.graph_op
def max_softmax(x):
    """MAX Graph softmax"""
    return ops.softmax(x, axis=-1)


@max.torch.graph_op
def max_layer_norm(x, gamma, beta):
    """MAX Graph layer normalization"""
    return ops.layer_norm(x, gamma, beta, epsilon=1e-5)


def benchmark_operation(name, max_op, pytorch_op, *inputs, iterations=100, warmup=10, device='cuda'):
    """Generic benchmark function"""

    # Move inputs to device
    inputs_cuda = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs]

    # Warmup MAX
    max_output = torch.empty_like(inputs_cuda[0] if len(inputs_cuda) == 1 else inputs_cuda[0])
    for _ in range(warmup):
        max_op(max_output, *inputs_cuda)

    torch.cuda.synchronize()

    # Benchmark MAX
    start = time.perf_counter()
    for _ in range(iterations):
        max_op(max_output, *inputs_cuda)
    torch.cuda.synchronize()
    max_time = time.perf_counter() - start

    # Warmup PyTorch
    for _ in range(warmup):
        _ = pytorch_op(*inputs_cuda)

    torch.cuda.synchronize()

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(iterations):
        _ = pytorch_op(*inputs_cuda)
    torch.cuda.synchronize()
    pytorch_time = time.perf_counter() - start

    max_ms = max_time * 1000 / iterations
    pytorch_ms = pytorch_time * 1000 / iterations
    speedup = pytorch_ms / max_ms

    return {
        'name': name,
        'max_ms': max_ms,
        'pytorch_ms': pytorch_ms,
        'speedup': speedup,
        'faster': 'MAX' if speedup > 1.0 else 'PyTorch'
    }


def run_benchmark(config):
    """Run full benchmark suite"""
    print("="*70)
    print("PyTorch CUDA vs MAX Graph Benchmark")
    print("Using @max.torch.graph_op decorator")
    print("="*70)
    print()
    print("Configuration:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Sequence length: {config['seq_len']}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Iterations: {config['iterations']}")
    print(f"  Warmup: {config['warmup']}")
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠ Warning: CUDA not available, using CPU")
    else:
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    print()

    results = {'config': config, 'operations': []}

    # GEMM Benchmark
    print("Benchmarking GEMM (Matrix Multiplication)...")
    M = config['seq_len'] * config['batch_size']
    K = config['hidden_size']
    N = config['hidden_size']

    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)

    gemm_result = benchmark_operation(
        'GEMM',
        max_matmul,
        lambda a, b: torch.matmul(a, b),
        A, B,
        iterations=config['iterations'],
        warmup=config['warmup'],
        device=device
    )
    results['operations'].append(gemm_result)

    print(f"  MAX:     {gemm_result['max_ms']:.4f} ms")
    print(f"  PyTorch: {gemm_result['pytorch_ms']:.4f} ms")
    print(f"  Speedup: {gemm_result['speedup']:.2f}x ({gemm_result['faster']} faster)")
    print()

    # Softmax Benchmark
    print("Benchmarking Softmax...")
    x = torch.randn(config['batch_size'], config['seq_len'], config['hidden_size'], device=device)

    softmax_result = benchmark_operation(
        'Softmax',
        max_softmax,
        lambda x: torch.softmax(x, dim=-1),
        x,
        iterations=config['iterations'],
        warmup=config['warmup'],
        device=device
    )
    results['operations'].append(softmax_result)

    print(f"  MAX:     {softmax_result['max_ms']:.4f} ms")
    print(f"  PyTorch: {softmax_result['pytorch_ms']:.4f} ms")
    print(f"  Speedup: {softmax_result['speedup']:.2f}x ({softmax_result['faster']} faster)")
    print()

    # LayerNorm Benchmark
    print("Benchmarking LayerNorm...")
    x = torch.randn(config['batch_size'], config['seq_len'], config['hidden_size'], device=device)
    gamma = torch.ones(config['hidden_size'], device=device)
    beta = torch.zeros(config['hidden_size'], device=device)
    ln = torch.nn.LayerNorm(config['hidden_size']).to(device)

    ln_result = benchmark_operation(
        'LayerNorm',
        max_layer_norm,
        lambda x, g, b: ln(x),
        x, gamma, beta,
        iterations=config['iterations'],
        warmup=config['warmup'],
        device=device
    )
    results['operations'].append(ln_result)

    print(f"  MAX:     {ln_result['max_ms']:.4f} ms")
    print(f"  PyTorch: {ln_result['pytorch_ms']:.4f} ms")
    print(f"  Speedup: {ln_result['speedup']:.2f}x ({ln_result['faster']} faster)")
    print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)

    total_max = sum(r['max_ms'] for r in results['operations'])
    total_pytorch = sum(r['pytorch_ms'] for r in results['operations'])
    overall_speedup = total_pytorch / total_max

    print(f"Total time:")
    print(f"  MAX:     {total_max:.4f} ms")
    print(f"  PyTorch: {total_pytorch:.4f} ms")
    print(f"  Overall Speedup: {overall_speedup:.2f}x")
    print()

    results['summary'] = {
        'total_max_ms': total_max,
        'total_pytorch_ms': total_pytorch,
        'overall_speedup': overall_speedup
    }

    # Save results
    with open('max_vs_pytorch_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: max_vs_pytorch_results.json")
    print("="*70)

    return results


if __name__ == "__main__":
    config = {
        'batch_size': 20,
        'seq_len': 35,
        'hidden_size': 200,
        'iterations': 100,
        'warmup': 10
    }

    try:
        results = run_benchmark(config)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
