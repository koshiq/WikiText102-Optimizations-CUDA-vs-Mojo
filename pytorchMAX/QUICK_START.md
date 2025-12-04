# Quick Start - PyTorch vs MAX Transformer Benchmark

## One Command Demo (Best Results)

```bash
pixi run python benchmark_transformer.py --ninp 1024 --nhid 4096 --nlayers 6 --batch-size 16 --seq-len 256 --iterations 50
```

**Expected Output:**
- ✅ **GEMM: 1.90x speedup** (MAX faster)
- ✅ **Full Model: 1.12x speedup** (MAX faster)

---

## What This Shows

This benchmark demonstrates **MAX Graph API acceleration** of PyTorch Transformer operations:

### Operations Accelerated:
1. **GEMM (Matrix Multiplication)** - Feed-forward network layers
2. **LogSoftmax** - Output probability computation
3. **LayerNorm** - Normalization layers

### Key Results:
- **GEMM gets 1.90x speedup** - compute-intensive operations benefit
- **Overall model gets 1.12x speedup** at large scale
- Smaller operations have overhead but GEMM dominates (~70% of time)

---

## All Available Commands

### Fast Demo (2 seconds)
```bash
pixi run python benchmark_transformer.py
```

### Medium Model (balanced)
```bash
pixi run python benchmark_transformer.py --ninp 512 --nhid 2048 --nlayers 4 --batch-size 32 --seq-len 128
```

### Large Model (best speedup)
```bash
pixi run python benchmark_transformer.py --ninp 1024 --nhid 4096 --nlayers 6 --batch-size 16 --seq-len 256 --iterations 50
```

---

## Output Interpretation

```
INDIVIDUAL OPERATION BENCHMARKS
----------------------------------------------------------------------
Benchmarking GEMM (Matrix Multiplication)...
  MAX:     1.84 ms      ← MAX implementation time
  PyTorch: 3.50 ms      ← PyTorch baseline time
  Speedup: 1.90x (MAX faster)   ← MAX is 1.90x faster! ✅
```

```
FULL MODEL SUMMARY
----------------------------------------------------------------------
PyTorch time: 134.50 ms/iter
MAX time:     120.20 ms/iter
Speedup:      1.12x (MAX faster)   ← Overall 1.12x speedup ✅
```

---

## Files

- `benchmark_transformer.py` - Main benchmark script
- `model_max.py` - MAX-accelerated Transformer model
- `model.py` - PyTorch baseline model
- `transformer_benchmark_results.json` - Detailed results

---

## Technical Details

**MAX Graph API Used:**
```python
@max.torch.graph_op
def max_matmul(A, B):
    return ops.matmul(A, B)

@max.torch.graph_op
def max_log_softmax(x):
    return ops.logsoftmax(x, axis=-1)

@max.torch.graph_op
def max_layer_norm(x, weight, bias):
    return ops.layer_norm(x, weight, bias, epsilon=1e-5)
```

**Integration:**
- Replaces PyTorch Linear layers in FFN with `MaxLinear` (using `max_matmul`)
- Replaces PyTorch LayerNorm with `MaxLayerNorm` (using `max_layer_norm`)
- Replaces final log_softmax with `max_log_softmax`

---

## GPU Requirements

- NVIDIA GPU with CUDA support
- 8GB VRAM (RTX 4070 tested)
- Reduce batch size if OOM occurs

---

For detailed documentation, see: [DEMO_COMMANDS.md](../DEMO_COMMANDS.md)
