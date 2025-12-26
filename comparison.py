"""
Benchmark PyTorch vs TorchScript vs MAX-accelerated WikiText2 Transformer model.
Compares inference performance on the actual trained model.
"""

import torch
import torch.nn as nn
import time
import json
import argparse
from transformer import TransformerModel
from transformer import create_scripted_model
from maxGraph_transformer import TransformerModel as TransformerModelMAX, max_matmul, max_log_softmax, max_layer_norm


def benchmark_operation(name, max_op, pytorch_op, *inputs, iterations=100, warmup=10, device='cuda', output_shape=None):
    """Benchmark individual operation"""
    inputs_cuda = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs]

    # Create output tensor
    if output_shape is not None:
        max_output = torch.empty(output_shape, dtype=inputs_cuda[0].dtype, device=device)
    else:
        max_output = torch.empty_like(inputs_cuda[0] if len(inputs_cuda) == 1 else inputs_cuda[0])

    # Warmup MAX
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


def benchmark_model(model, data, iterations=100, warmup=10, device='cuda'):
    """Benchmark a model's inference time"""
    model.eval()
    model.to(device)
    data = data.to(device)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(data)

        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(data)
        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    return elapsed * 1000 / iterations  # ms per iteration


def run_transformer_benchmark(config):
    """Run full Transformer model benchmark"""
    print("="*70)
    print("WikiText2 Transformer: PyTorch vs TorchScript vs MAX Benchmark")
    print("="*70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠ Warning: CUDA not available, using CPU")
        return

    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    print()
    print("Model Configuration:")
    print(f"  Vocabulary size: {config['ntoken']}")
    print(f"  Embedding dim: {config['ninp']}")
    print(f"  Number of heads: {config['nhead']}")
    print(f"  Hidden dim (FFN): {config['nhid']}")
    print(f"  Number of layers: {config['nlayers']}")
    print()
    print("Benchmark Configuration:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Sequence length: {config['seq_len']}")
    print(f"  Iterations: {config['iterations']}")
    print(f"  Warmup: {config['warmup']}")
    print()

    # Create test data
    test_data = torch.randint(0, config['ntoken'], (config['seq_len'], config['batch_size']), device=device)

    # Benchmark PyTorch Eager model
    print("Building PyTorch-Eager model...")
    pytorch_model = TransformerModel(
        config['ntoken'],
        config['ninp'],
        config['nhead'],
        config['nhid'],
        config['nlayers'],
        config['dropout']
    )

    print("Benchmarking PyTorch-Eager model...")
    pytorch_time = benchmark_model(
        pytorch_model,
        test_data,
        config['iterations'],
        config['warmup'],
        device
    )
    print(f"  PyTorch-Eager: {pytorch_time:.4f} ms/iter")
    print()

    # Save PyTorch model weights to disk
    print("Saving PyTorch model weights...")
    weights_path = 'temp_pytorch_weights.pth'
    torch.save(pytorch_model.state_dict(), weights_path)

    # Delete PyTorch model and free GPU memory
    print("Freeing PyTorch-Eager model memory...")
    del pytorch_model
    torch.cuda.empty_cache()
    print(f"  GPU Memory freed. Available: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print()

    # Benchmark PyTorch TorchScript model
    print("Building PyTorch-Script model (TorchScript with fusion)...")
    try:
        scripted_model = create_scripted_model(
            config['ntoken'],
            config['ninp'],
            config['nhead'],
            config['nhid'],
            config['nlayers'],
            config['dropout'],
            device
        )

        print("Benchmarking PyTorch-Script model...")
        scripted_time = benchmark_model(
            scripted_model,
            test_data,
            config['iterations'],
            config['warmup'],
            device
        )
        print(f"  PyTorch-Script: {scripted_time:.4f} ms/iter")
        script_speedup = pytorch_time / scripted_time
        print(f"  Speedup vs Eager: {script_speedup:.2f}x ({(script_speedup-1)*100:+.1f}%)")
        print()

        # Free TorchScript model memory
        print("Freeing PyTorch-Script model memory...")
        del scripted_model
        torch.cuda.empty_cache()
        print(f"  GPU Memory freed. Available: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print()
    except Exception as e:
        print(f"  [ERROR] TorchScript benchmark failed: {e}")
        scripted_time = None
        print()

    # Benchmark MAX model
    print("Building MAX-accelerated model...")
    max_model = TransformerModelMAX(
        config['ntoken'],
        config['ninp'],
        config['nhead'],
        config['nhid'],
        config['nlayers'],
        config['dropout']
    )

    # Load weights from saved PyTorch model
    print("Loading weights from saved PyTorch model...")
    pytorch_weights = torch.load(weights_path)
    max_model.load_state_dict(pytorch_weights, strict=False)

    # Clean up weights file
    import os
    os.remove(weights_path)

    print("Benchmarking MAX model...")
    max_time = benchmark_model(
        max_model,
        test_data,
        config['iterations'],
        config['warmup'],
        device
    )
    print(f"  MAX: {max_time:.4f} ms/iter")
    print()

    # Free MAX model memory before individual operation benchmarks
    print("Freeing MAX model memory...")
    del max_model
    torch.cuda.empty_cache()
    print()

    # Benchmark individual operations
    print("="*70)
    print("INDIVIDUAL OPERATION BENCHMARKS (3-way comparison)")
    print("="*70)
    print()

    # Try to import custom CUDA ops
    try:
        import custom_ops
        cuda_available = True
        print("✓ Custom CUDA kernels available")
    except ImportError:
        # Try JIT compilation
        print("⏳ Custom CUDA kernels not pre-installed, attempting JIT compilation...")
        try:
            import os
            import sys
            from torch.utils.cpp_extension import load

            # Monkey-patch to disable CUDA version check
            from torch.utils import cpp_extension
            original_check = cpp_extension._check_cuda_version
            def patched_check(*args, **kwargs):
                pass
            cpp_extension._check_cuda_version = patched_check

            # Get cuda directory path
            cuda_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda')

            custom_ops = load(
                name='custom_ops_jit',
                sources=[
                    os.path.join(cuda_dir, 'custom_ops.cpp'),
                    os.path.join(cuda_dir, 'cuda_kernels', 'gemm_kernel.cu'),
                    os.path.join(cuda_dir, 'cuda_kernels', 'layernorm_kernel.cu'),
                    os.path.join(cuda_dir, 'cuda_kernels', 'softmax_kernel.cu'),
                ],
                verbose=False,
            )
            cuda_available = True
            print("✓ Custom CUDA kernels compiled via JIT")
        except Exception as e:
            cuda_available = False
            print(f"✗ Custom CUDA kernels compilation failed: {e}")
            print("  Note: On Windows, Microsoft Visual C++ Build Tools are required:")
            print("  https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print()

    operation_results = []

    # GEMM (Matrix Multiplication) - use model dimensions
    print("Benchmarking GEMM (Matrix Multiplication)...")
    batch_tokens = config['batch_size'] * config['seq_len']
    A = torch.randn(batch_tokens, config['ninp'], device=device)
    B = torch.randn(config['ninp'], config['nhid'], device=device)
    # For custom_ops.gemm_forward, weights are [out_features, in_features]
    B_weights = torch.randn(config['nhid'], config['ninp'], device=device)

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(config['warmup']):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(config['iterations']):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) * 1000 / config['iterations']

    # Benchmark MAX Graph
    max_output = torch.empty(batch_tokens, config['nhid'], dtype=A.dtype, device=device)
    torch.cuda.synchronize()
    for _ in range(config['warmup']):
        max_matmul(max_output, A, B)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(config['iterations']):
        max_matmul(max_output, A, B)
    torch.cuda.synchronize()
    max_time = (time.perf_counter() - start) * 1000 / config['iterations']

    # Benchmark Custom CUDA Tensor Cores
    if cuda_available:
        torch.cuda.synchronize()
        for _ in range(config['warmup']):
            _ = custom_ops.gemm_forward(A, B_weights)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(config['iterations']):
            _ = custom_ops.gemm_forward(A, B_weights)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) * 1000 / config['iterations']

        print(f"  PyTorch cuBLAS: {pytorch_time:.4f} ms")
        print(f"  MAX Graph:      {max_time:.4f} ms  ({pytorch_time/max_time:.2f}x vs cuBLAS)")
        print(f"  Custom CUDA:    {cuda_time:.4f} ms  ({pytorch_time/cuda_time:.2f}x vs cuBLAS)")
        print(f"  🏆 Winner: {'Custom CUDA' if cuda_time < min(pytorch_time, max_time) else ('MAX' if max_time < pytorch_time else 'cuBLAS')}")
    else:
        print(f"  PyTorch cuBLAS: {pytorch_time:.4f} ms")
        print(f"  MAX Graph:      {max_time:.4f} ms  ({pytorch_time/max_time:.2f}x vs cuBLAS)")
    print()

    # Softmax / Log Softmax - use smaller dimension to avoid OOM
    print("Benchmarking Log Softmax...")
    x = torch.randn(config['seq_len'], config['batch_size'], config['nhid'], device=device)  # Use nhid instead of ntoken

    softmax_result = benchmark_operation(
        'LogSoftmax',
        max_log_softmax,
        lambda x: torch.log_softmax(x, dim=-1),
        x,
        iterations=config['iterations'],
        warmup=config['warmup'],
        device=device
    )
    operation_results.append(softmax_result)
    print(f"  MAX:     {softmax_result['max_ms']:.4f} ms")
    print(f"  PyTorch: {softmax_result['pytorch_ms']:.4f} ms")
    print(f"  Speedup: {softmax_result['speedup']:.2f}x ({softmax_result['faster']} faster)")
    print()

    # LayerNorm
    print("Benchmarking LayerNorm...")
    x = torch.randn(config['batch_size'], config['seq_len'], config['ninp'], device=device)
    gamma = torch.ones(config['ninp'], device=device)
    beta = torch.zeros(config['ninp'], device=device)
    ln = torch.nn.LayerNorm(config['ninp']).to(device)

    ln_result = benchmark_operation(
        'LayerNorm',
        max_layer_norm,
        lambda x, g, b: ln(x),
        x, gamma, beta,
        iterations=config['iterations'],
        warmup=config['warmup'],
        device=device
    )
    operation_results.append(ln_result)
    print(f"  MAX:     {ln_result['max_ms']:.4f} ms")
    print(f"  PyTorch: {ln_result['pytorch_ms']:.4f} ms")
    print(f"  Speedup: {ln_result['speedup']:.2f}x ({ln_result['faster']} faster)")
    print()

    # Calculate speedups
    speedup = pytorch_time / max_time
    faster = "MAX" if speedup > 1.0 else "PyTorch-Eager"

    print("="*70)
    print("FULL MODEL SUMMARY")
    print("="*70)
    print(f"PyTorch-Eager:  {pytorch_time:.4f} ms/iter (baseline)")
    if scripted_time is not None:
        script_speedup = pytorch_time / scripted_time
        print(f"PyTorch-Script: {scripted_time:.4f} ms/iter ({script_speedup:.2f}x vs Eager)")
    print(f"MAX Graph:      {max_time:.4f} ms/iter ({speedup:.2f}x vs Eager)")

    if scripted_time is not None:
        max_vs_script = scripted_time / max_time
        print(f"\nMAX vs TorchScript: {max_vs_script:.2f}x")
        print(f"  This shows MAX's kernel advantage (both use graph compilation)")

    print(f"\n🏆 Winner: {faster}")
    print()

    # Save results
    results = {
        'config': config,
        'full_model': {
            'pytorch_eager_time_ms': pytorch_time,
            'pytorch_script_time_ms': scripted_time if scripted_time is not None else None,
            'max_time_ms': max_time,
            'speedup_vs_eager': speedup,
            'script_speedup_vs_eager': script_speedup if scripted_time is not None else None,
            'max_vs_script': max_vs_script if scripted_time is not None else None,
            'faster': faster
        },
        'operations': operation_results,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }

    output_file = 'transformer_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print("="*70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark WikiText2 Transformer')
    parser.add_argument('--ntoken', type=int, default=28782, help='Vocabulary size')
    parser.add_argument('--ninp', type=int, default=200, help='Embedding dimension')
    parser.add_argument('--nhead', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--nhid', type=int, default=200, help='FFN hidden dimension')
    parser.add_argument('--nlayers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=35, help='Sequence length')
    parser.add_argument('--iterations', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    args = parser.parse_args()

    config = {
        'ntoken': args.ntoken,
        'ninp': args.ninp,
        'nhead': args.nhead,
        'nhid': args.nhid,
        'nlayers': args.nlayers,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'iterations': args.iterations,
        'warmup': args.warmup
    }

    try:
        run_transformer_benchmark(config)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()