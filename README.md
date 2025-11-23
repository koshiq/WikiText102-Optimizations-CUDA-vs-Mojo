# WikiText-2 Transformer Language Model

A PyTorch implementation of a Transformer-based language model trained on the WikiText-2 dataset, with support for CUDA acceleration and deployment on Modal's A100 GPUs.

## Features

- **Transformer Architecture**: Modern transformer-based language model with positional encoding
- **WikiText-2 Dataset**: Trained on the WikiText-2 benchmark dataset
- **GPU Acceleration**: Full support for CUDA and Modal A100 GPUs
- **Comprehensive Benchmarking**: Tools for measuring training time, inference speed, memory usage, and perplexity
- **Profiling Support**: PyTorch profiler integration for performance analysis
- **Text Generation**: Sample new text from trained models

## Project Structure

```
.
├── wikitext2-transformer/
│   ├── main.py              # Main training script
│   ├── model.py             # Transformer model definition
│   ├── data.py              # Data loading utilities
│   ├── generate.py          # Text generation script
│   ├── benchmark.py         # Comprehensive benchmarking
│   ├── profile_model.py     # Performance profiling
│   ├── requirements.txt     # Python dependencies
│   └── data/
│       └── wikitext-2/
│           ├── train.txt
│           ├── valid.txt
│           └── test.txt
├── modal_train.py           # Modal deployment script for A100
└── README.md
```

## Installation

### Local Setup

```bash
cd wikitext2-transformer
pip install -r requirements.txt
```

### Modal Setup

```bash
pip install modal
modal setup
```

## Usage

### Training Locally

**CPU Training:**
```bash
cd wikitext2-transformer
python main.py --data ./data/wikitext-2
```

**GPU Training:**
```bash
python main.py --data ./data/wikitext-2 --accel
```

**With Custom Parameters:**
```bash
python main.py \
  --data ./data/wikitext-2 \
  --epochs 40 \
  --batch_size 20 \
  --emsize 200 \
  --nhid 200 \
  --nlayers 4 \
  --nhead 2 \
  --lr 20 \
  --dropout 0.2 \
  --use-optimizer \
  --accel
```

### Training on Modal A100 GPU

**Quick Start:**
```bash
# Train on A100
modal run modal_train.py --mode train --epochs 40

# Benchmark on A100
modal run modal_train.py --mode benchmark --epochs 5 --runs 3

# Profile on A100
modal run modal_train.py --mode profile
```

**Deploy as a persistent app:**
```bash
modal deploy modal_train.py
```

### Text Generation

```bash
cd wikitext2-transformer
python generate.py \
  --checkpoint model.pt \
  --words 1000 \
  --temperature 1.0 \
  --outf generated.txt
```

### Benchmarking

```bash
cd wikitext2-transformer
python benchmark.py \
  --epochs 5 \
  --runs 3 \
  --output benchmark_results.json \
  --accel
```

The benchmark results include:
- Training time per epoch
- Inference time
- Tokens per second
- Memory usage (CPU and GPU)
- Perplexity scores

### Profiling

```bash
cd wikitext2-transformer
python profile_model.py --profile-batches 100 --accel
```

This generates:
- `trace_training.json` - Training loop profiling trace
- `trace_inference.json` - Inference loop profiling trace
- `profiler_stacks.txt` - Detailed stack traces

View traces in Chrome at `chrome://tracing`

## Model Architecture

- **Type**: Transformer Encoder
- **Positional Encoding**: Sinusoidal
- **Default Configuration**:
  - Embedding size: 200
  - Hidden size: 200
  - Number of layers: 4
  - Attention heads: 2
  - Dropout: 0.2

## Command Line Arguments

### Training (main.py)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `./data/wikitext-2` | Location of the data corpus |
| `--emsize` | int | 200 | Size of word embeddings |
| `--nhid` | int | 200 | Number of hidden units per layer |
| `--nlayers` | int | 4 | Number of layers |
| `--nhead` | int | 2 | Number of attention heads |
| `--lr` | float | 20 | Initial learning rate |
| `--epochs` | int | 40 | Number of training epochs |
| `--batch_size` | int | 20 | Batch size |
| `--bptt` | int | 35 | Sequence length |
| `--dropout` | float | 0.2 | Dropout rate |
| `--clip` | float | 0.25 | Gradient clipping threshold |
| `--save` | str | `model.pt` | Path to save the model |
| `--accel` | flag | False | Enable GPU acceleration |
| `--use-optimizer` | flag | False | Use AdamW optimizer |
| `--dry-run` | flag | False | Verify code without full training |

## Modal Deployment Details

The Modal deployment script (`modal_train.py`) provides:

1. **Automatic GPU provisioning**: A100 40GB GPU
2. **Code mounting**: Your local code is automatically synced
3. **Three modes**:
   - `train`: Full training run, returns trained model
   - `benchmark`: Performance benchmarking with statistics
   - `profile`: Detailed performance profiling
4. **4-hour timeout**: Suitable for long training runs
5. **Result persistence**: Models and results are downloaded locally

### Modal Configuration

```python
gpu=modal.gpu.A100(count=1, size="40GB")  # A100 40GB
# or
gpu=modal.gpu.A100(count=1, size="80GB")  # A100 80GB
# or
gpu=modal.gpu.A100(count=2, size="40GB")  # 2x A100 40GB
```

## Performance Tips

1. **Use `--use-optimizer`**: AdamW optimizer often converges faster
2. **Increase batch size on A100**: Try `--batch_size 64` or higher
3. **Adjust learning rate**: If using AdamW, reduce `--lr` to 0.001
4. **Gradient accumulation**: For larger effective batch sizes
5. **Mixed precision**: Consider adding AMP for faster training

## Results

Typical perplexity scores on WikiText-2:
- **Training**: ~100-150 after 40 epochs
- **Validation**: ~115-135
- **Test**: ~110-130

Performance on A100:
- Training: ~2000-3000 tokens/sec
- Inference: ~5000-8000 tokens/sec

## License

This project is based on PyTorch examples and follows the same license.

## Citation

If you use this code, please cite:

```
@misc{wikitext2-transformer,
  author = {Your Name},
  title = {WikiText-2 Transformer Language Model},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/wikitext2-transformer}
}
```

## References

- [WikiText-2 Dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Modal Documentation](https://modal.com/docs)
