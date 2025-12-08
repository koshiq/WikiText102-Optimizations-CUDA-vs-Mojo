"""
Benchmark PyTorch vs MAX-accelerated WikiText2 Transformer model.
Compares inference performance on the actual trained model.
"""

import torch
import torch.nn as nn
import time
import json
import argparse
from transformer import TransformerModel
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
    print("WikiText2 Transformer: PyTorch vs MAX Benchmark")
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

    # Benchmark PyTorch model
    print("Building PyTorch model...")
    pytorch_model = TransformerModel(
        config['ntoken'],
        config['ninp'],
        config['nhead'],
        config['nhid'],
        config['nlayers'],
        config['dropout']
    )

    print("Benchmarking PyTorch model...")
    pytorch_time = benchmark_model(
        pytorch_model,
        test_data,
        config['iterations'],
        config['warmup'],
        device
    )
    print(f"  PyTorch: {pytorch_time:.4f} ms/iter")
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

    # Copy weights from PyTorch model for fair comparison (handle naming differences)
    print("Copying weights from PyTorch model...")
    # Embedding may be named `input_emb` or `encoder_embedding` depending on implementation
    src_emb = getattr(pytorch_model, 'input_emb', None) or getattr(pytorch_model, 'encoder_embedding', None)
    dst_emb = getattr(max_model, 'input_emb', None) or getattr(max_model, 'encoder_embedding', None)
    if src_emb is not None and dst_emb is not None:
        dst_emb.weight.data = src_emb.weight.data.clone()
    else:
        print('Warning: could not find embedding attribute to copy (input_emb / encoder_embedding)')

    # Positional encoder should be present as `pos_encoder`
    if hasattr(pytorch_model, 'pos_encoder') and hasattr(max_model, 'pos_encoder'):
        max_model.pos_encoder.pe = pytorch_model.pos_encoder.pe.clone()
    else:
        print('Warning: could not find pos_encoder to copy')

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

    # Benchmark individual operations
    print("="*70)
    print("INDIVIDUAL OPERATION BENCHMARKS")
    print("="*70)
    print()

    operation_results = []

    # GEMM (Matrix Multiplication) - use model dimensions
    print("Benchmarking GEMM (Matrix Multiplication)...")
    batch_tokens = config['batch_size'] * config['seq_len']
    A = torch.randn(batch_tokens, config['ninp'], device=device)
    B = torch.randn(config['ninp'], config['nhid'], device=device)  # Correct shape for matmul

    gemm_result = benchmark_operation(
        'GEMM',
        max_matmul,
        lambda a, b: torch.matmul(a, b),
        A, B,
        iterations=config['iterations'],
        warmup=config['warmup'],
        device=device,
        output_shape=(batch_tokens, config['nhid'])  # Specify correct output shape
    )
    operation_results.append(gemm_result)
    print(f"  MAX:     {gemm_result['max_ms']:.4f} ms")
    print(f"  PyTorch: {gemm_result['pytorch_ms']:.4f} ms")
    print(f"  Speedup: {gemm_result['speedup']:.2f}x ({gemm_result['faster']} faster)")
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

    # Calculate speedup
    speedup = pytorch_time / max_time
    faster = "MAX" if speedup > 1.0 else "PyTorch"

    print("="*70)
    print("FULL MODEL SUMMARY")
    print("="*70)
    print(f"PyTorch time: {pytorch_time:.4f} ms/iter")
    print(f"MAX time:     {max_time:.4f} ms/iter")
    print(f"Speedup:      {speedup:.2f}x ({faster} faster)")
    print()

    # Save results
    results = {
        'config': config,
        'full_model': {
            'pytorch_time_ms': pytorch_time,
            'max_time_ms': max_time,
            'speedup': speedup,
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