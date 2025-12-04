# MAX Graph API Transformer Benchmark - Demo Commands

This guide shows how to run the benchmarks showcasing PyTorch vs MAX-accelerated operations.

## Setup

All commands should be run from the `pytorchMAX` directory:

```bash
cd /home/koshiq/WikiText102-Optimizations-CUDA-vs-Mojo/pytorchMAX
```

## Quick Demo (Recommended)

### 1. Small Model (Fast - Shows Overhead)
```bash
pixi run python benchmark_transformer.py
```

**Expected Results:**
- Full model: ~0.26x (PyTorch faster - overhead dominates)
- GEMM: ~1.5-2.0x (MAX faster)
- Softmax/LayerNorm: Slower (overhead)

### 2. Large Model (Shows MAX Advantage)
```bash
pixi run python benchmark_transformer.py \
    --ninp 1024 --nhid 4096 --nlayers 6 \
    --batch-size 16 --seq-len 256 --iterations 50
```

**Expected Results:**
- **Full model: ~1.12x speedup** (MAX faster!)
- **GEMM: ~1.90x speedup** (MAX faster!)
- Softmax/LayerNorm: Still slower (overhead)

## Detailed Benchmark Configurations

### Medium Model (512 embedding, 2048 hidden)
```bash
pixi run python benchmark_transformer.py \
    --ninp 512 --nhid 2048 --nlayers 4 \
    --batch-size 32 --seq-len 128
```

### Very Large Model (Maximum speedup)
```bash
pixi run python benchmark_transformer.py \
    --ninp 1024 --nhid 4096 --nlayers 6 \
    --batch-size 16 --seq-len 256 --iterations 50
```

## What Gets Benchmarked

The script benchmarks:

1. **Full Transformer Model**
   - PyTorch baseline
   - MAX-accelerated version
   - Overall speedup

2. **Individual Operations**
   - **GEMM (Matrix Multiplication)** - FFN layers (~70% of compute)
   - **LogSoftmax** - Output layer
   - **LayerNorm** - Normalization layers

## Output Format

The benchmark shows:

```
======================================================================
WikiText2 Transformer: PyTorch vs MAX Benchmark
======================================================================

✓ Using GPU: NVIDIA GeForce RTX 4070 Laptop GPU

Model Configuration:
  Vocabulary size: 28782
  Embedding dim: 1024
  ...

Building PyTorch model...
Benchmarking PyTorch model...
  PyTorch: 134.50 ms/iter

Building MAX-accelerated model...
Benchmarking MAX model...
  MAX: 120.20 ms/iter

======================================================================
INDIVIDUAL OPERATION BENCHMARKS
======================================================================

Benchmarking GEMM (Matrix Multiplication)...
  MAX:     1.84 ms
  PyTorch: 3.50 ms
  Speedup: 1.90x (MAX faster)

Benchmarking Log Softmax...
  MAX:     1.35 ms
  PyTorch: 0.61 ms
  Speedup: 0.46x (PyTorch faster)

Benchmarking LayerNorm...
  MAX:     0.25 ms
  PyTorch: 0.07 ms
  Speedup: 0.30x (PyTorch faster)

======================================================================
FULL MODEL SUMMARY
======================================================================
PyTorch time: 134.50 ms/iter
MAX time:     120.20 ms/iter
Speedup:      1.12x (MAX faster)

Results saved to: transformer_benchmark_results.json
======================================================================
```

## Key Findings

1. **GEMM Acceleration**: 1.90x speedup on large matrices
   - This is the dominant operation (~70% of compute time)
   - MAX's optimized matrix multiplication shines here

2. **Overall Model Speedup**: 1.12x on large models
   - GEMM speedup dominates overall performance
   - Overhead from smaller operations is amortized

3. **Smaller Operations**: Slower with MAX
   - LogSoftmax, LayerNorm have compilation overhead
   - PyTorch's highly optimized kernels are faster for small ops

## Results File

Results are saved to `transformer_benchmark_results.json`:

```bash
cat transformer_benchmark_results.json
```

## Comparison with Microbenchmark

To compare with standalone operation benchmarks:

```bash
pixi run python benchmark.py --hidden-size 512 --batch-size 32 --seq-len 128
```

This shows pure operation performance without model overhead.

## Environment Info

- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU (8GB)
- **Framework**: MAX 25.7.0, PyTorch 2.9.1
- **CUDA**: 13.0 (driver), 12.8 (PyTorch)
- **Python**: 3.11

## Troubleshooting

### Out of Memory
If you get OOM errors, reduce model size:
```bash
pixi run python benchmark_transformer.py \
    --ninp 512 --nhid 1024 --nlayers 2 \
    --batch-size 16 --seq-len 64
```

### Cache Issues
If you get library path errors, clear MAX cache:
```bash
rm -rf .pixi/envs/default/share/max/.max_cache
```

## For Your Report

Use these commands for your project milestone:

**Phase 2-3: MAX Graph Integration**
```bash
# Show individual operation speedups
pixi run python benchmark_transformer.py --ninp 512 --nhid 2048 --nlayers 4

# Show overall model acceleration
pixi run python benchmark_transformer.py --ninp 1024 --nhid 4096 --nlayers 6 --iterations 50
```

**Key Metrics to Report:**
- GEMM: 1.90x speedup (MAX faster)
- Full model (large): 1.12x speedup
- Demonstrates MAX acceleration for compute-intensive operations
