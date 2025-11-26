"""
Benchmark script comparing Mojo kernels vs PyTorch CUDA.
Tests GEMM, Softmax, and LayerNorm operations.
"""

import argparse
import time
import json
import torch
import torch.nn as nn
from mojo_ops import (
    MojoGEMM, MojoSoftmax, MojoLogSoftmax, MojoLayerNorm,
    benchmark_op, MOJO_AVAILABLE
)

parser = argparse.ArgumentParser(description='Benchmark Mojo vs CUDA kernels')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use (cuda or cpu)')
parser.add_argument('--batch-size', type=int, default=20,
                    help='Batch size for testing')
parser.add_argument('--seq-len', type=int, default=35,
                    help='Sequence length')
parser.add_argument('--hidden-size', type=int, default=200,
                    help='Hidden dimension size')
parser.add_argument('--warmup', type=int, default=10,
                    help='Number of warmup iterations')
parser.add_argument('--iterations', type=int, default=100,
                    help='Number of benchmark iterations')
parser.add_argument('--output', type=str, default='mojo_benchmark.json',
                    help='Output JSON file')
args = parser.parse_args()


def benchmark_gemm(device, batch_size, seq_len, hidden_size, warmup, iterations):
    """Benchmark GEMM operation."""
    print("\n" + "="*80)
    print("Benchmarking GEMM (Linear Layer)")
    print("="*80)

    # Create test input
    x = torch.randn(seq_len, batch_size, hidden_size, device=device)

    # Create layers
    mojo_layer = MojoGEMM(hidden_size, hidden_size).to(device)
    pytorch_layer = nn.Linear(hidden_size, hidden_size).to(device)

    # Copy weights for fair comparison
    pytorch_layer.weight.data = mojo_layer.weight.data.clone()
    pytorch_layer.bias.data = mojo_layer.bias.data.clone()

    # Benchmark
    results = benchmark_op(
        "GEMM",
        mojo_layer,
        pytorch_layer,
        x,
        warmup,
        iterations
    )

    print(f"  Mojo:    {results['mojo_time_ms']:.3f} ms/iter")
    print(f"  PyTorch: {results['pytorch_time_ms']:.3f} ms/iter")
    print(f"  Speedup: {results['speedup']:.2f}x ({results['faster']} is faster)")

    return results


def benchmark_softmax(device, batch_size, seq_len, hidden_size, warmup, iterations):
    """Benchmark Softmax operation."""
    print("\n" + "="*80)
    print("Benchmarking Softmax")
    print("="*80)

    # Create test input
    x = torch.randn(seq_len, batch_size, hidden_size, device=device)

    # Create layers
    mojo_layer = MojoSoftmax(dim=-1)
    pytorch_layer = nn.Softmax(dim=-1)

    # Benchmark
    results = benchmark_op(
        "Softmax",
        mojo_layer,
        pytorch_layer,
        x,
        warmup,
        iterations
    )

    print(f"  Mojo:    {results['mojo_time_ms']:.3f} ms/iter")
    print(f"  PyTorch: {results['pytorch_time_ms']:.3f} ms/iter")
    print(f"  Speedup: {results['speedup']:.2f}x ({results['faster']} is faster)")

    return results


def benchmark_log_softmax(device, batch_size, seq_len, hidden_size, warmup, iterations):
    """Benchmark LogSoftmax operation."""
    print("\n" + "="*80)
    print("Benchmarking LogSoftmax")
    print("="*80)

    # Create test input
    x = torch.randn(seq_len, batch_size, hidden_size, device=device)

    # Create layers
    mojo_layer = MojoLogSoftmax(dim=-1)
    pytorch_layer = nn.LogSoftmax(dim=-1)

    # Benchmark
    results = benchmark_op(
        "LogSoftmax",
        mojo_layer,
        pytorch_layer,
        x,
        warmup,
        iterations
    )

    print(f"  Mojo:    {results['mojo_time_ms']:.3f} ms/iter")
    print(f"  PyTorch: {results['pytorch_time_ms']:.3f} ms/iter")
    print(f"  Speedup: {results['speedup']:.2f}x ({results['faster']} is faster)")

    return results


def benchmark_layernorm(device, batch_size, seq_len, hidden_size, warmup, iterations):
    """Benchmark LayerNorm operation."""
    print("\n" + "="*80)
    print("Benchmarking LayerNorm")
    print("="*80)

    # Create test input
    x = torch.randn(seq_len, batch_size, hidden_size, device=device)

    # Create layers
    mojo_layer = MojoLayerNorm(hidden_size).to(device)
    pytorch_layer = nn.LayerNorm(hidden_size).to(device)

    # Copy weights for fair comparison
    pytorch_layer.weight.data = mojo_layer.weight.data.clone()
    pytorch_layer.bias.data = mojo_layer.bias.data.clone()

    # Benchmark
    results = benchmark_op(
        "LayerNorm",
        mojo_layer,
        pytorch_layer,
        x,
        warmup,
        iterations
    )

    print(f"  Mojo:    {results['mojo_time_ms']:.3f} ms/iter")
    print(f"  PyTorch: {results['pytorch_time_ms']:.3f} ms/iter")
    print(f"  Speedup: {results['speedup']:.2f}x ({results['faster']} is faster)")

    return results


if __name__ == '__main__':
    print("\n" + "="*80)
    print("MOJO vs CUDA KERNEL BENCHMARK")
    print("="*80)

    if not MOJO_AVAILABLE:
        print("\nERROR: MAX Engine not available!")
        print("Please install MAX Engine to run Mojo benchmarks.")
        print("Falling back to PyTorch CUDA for all operations.")
        exit(1)

    # Check device
    device = torch.device(args.device)
    print(f"\nDevice: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    print(f"\nTest Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Benchmark iterations: {args.iterations}")

    # Run benchmarks
    results = {
        'config': {
            'device': str(device),
            'batch_size': args.batch_size,
            'seq_len': args.seq_len,
            'hidden_size': args.hidden_size,
            'warmup': args.warmup,
            'iterations': args.iterations
        },
        'benchmarks': []
    }

    # GEMM benchmark
    gemm_results = benchmark_gemm(
        device, args.batch_size, args.seq_len, args.hidden_size,
        args.warmup, args.iterations
    )
    results['benchmarks'].append(gemm_results)

    # Softmax benchmark
    softmax_results = benchmark_softmax(
        device, args.batch_size, args.seq_len, args.hidden_size,
        args.warmup, args.iterations
    )
    results['benchmarks'].append(softmax_results)

    # LogSoftmax benchmark
    log_softmax_results = benchmark_log_softmax(
        device, args.batch_size, args.seq_len, args.hidden_size,
        args.warmup, args.iterations
    )
    results['benchmarks'].append(log_softmax_results)

    # LayerNorm benchmark
    layernorm_results = benchmark_layernorm(
        device, args.batch_size, args.seq_len, args.hidden_size,
        args.warmup, args.iterations
    )
    results['benchmarks'].append(layernorm_results)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total_mojo_time = sum(r['mojo_time_ms'] for r in results['benchmarks'])
    total_pytorch_time = sum(r['pytorch_time_ms'] for r in results['benchmarks'])
    overall_speedup = total_pytorch_time / total_mojo_time

    print(f"\nTotal Time:")
    print(f"  Mojo:    {total_mojo_time:.3f} ms")
    print(f"  PyTorch: {total_pytorch_time:.3f} ms")
    print(f"  Overall Speedup: {overall_speedup:.2f}x")

    results['summary'] = {
        'total_mojo_time_ms': total_mojo_time,
        'total_pytorch_time_ms': total_pytorch_time,
        'overall_speedup': overall_speedup
    }

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")
