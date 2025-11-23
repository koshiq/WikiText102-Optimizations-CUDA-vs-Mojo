"""
Comprehensive benchmarking script for PyTorch language model.
Measures training time, inference time, memory usage, and perplexity.
"""
import argparse
import time
import math
import json
import os
from datetime import datetime
import torch
import torch.nn as nn
import psutil
import data
from model import TransformerModel

parser = argparse.ArgumentParser(description='Benchmark PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=4,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs to benchmark')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nhead', type=int, default=2,
                    help='number of heads in transformer')
parser.add_argument('--accel', action='store_true',
                    help='use accelerator if available')
parser.add_argument('--warmup-batches', type=int, default=10,
                    help='number of warmup batches before timing')
parser.add_argument('--output', type=str, default='benchmark_results.json',
                    help='output file for benchmark results')
parser.add_argument('--runs', type=int, default=3,
                    help='number of benchmark runs for averaging')
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)

# Device setup
if args.accel and torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load data
corpus = data.Corpus(args.data)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# Build model
ntokens = len(corpus.dictionary)
model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
criterion = nn.NLLLoss()

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_mem_mb = mem_info.rss / 1024 / 1024

    gpu_mem_mb = 0
    if device.type == 'cuda':
        gpu_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    return cpu_mem_mb, gpu_mem_mb

def evaluate(data_source, desc="Evaluation"):
    """Evaluate model and return loss and perplexity."""
    model.eval()
    total_loss = 0.
    start_time = time.perf_counter()

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output = model(data)
            output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()

    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / (len(data_source) - 1)
    perplexity = math.exp(avg_loss)

    # Calculate tokens per second
    num_tokens = (len(data_source) - 1) * eval_batch_size
    tokens_per_sec = num_tokens / elapsed

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'time_seconds': elapsed,
        'tokens_per_second': tokens_per_sec
    }

def train_epoch(lr):
    """Train for one epoch and return metrics."""
    model.train()
    total_loss = 0.
    batch_times = []

    # Warmup
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        if batch >= args.warmup_batches:
            break
        data, targets = get_batch(train_data, i)
        model.zero_grad()
        output = model(data)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

    # Actual training with timing
    epoch_start = time.perf_counter()
    num_batches = 0

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        batch_start = time.perf_counter()

        data, targets = get_batch(train_data, i)
        model.zero_grad()
        output = model(data)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()
        batch_times.append(time.perf_counter() - batch_start)
        num_batches += 1

    epoch_time = time.perf_counter() - epoch_start
    avg_loss = total_loss / num_batches

    # Calculate throughput
    num_tokens = (len(train_data) - 1) * args.batch_size
    tokens_per_sec = num_tokens / epoch_time

    return {
        'loss': avg_loss,
        'perplexity': math.exp(avg_loss),
        'time_seconds': epoch_time,
        'avg_batch_time_ms': (sum(batch_times) / len(batch_times)) * 1000,
        'tokens_per_second': tokens_per_sec,
        'num_batches': num_batches
    }

def run_benchmark():
    """Run complete benchmark suite."""
    print("\n" + "="*80)
    print("BENCHMARK CONFIGURATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Vocabulary size: {ntokens}")
    print(f"Embedding size: {args.emsize}")
    print(f"Hidden size: {args.nhid}")
    print(f"Number of layers: {args.nlayers}")
    print(f"Number of heads: {args.nhead}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.bptt}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Number of runs: {args.runs}")

    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    results = {
        'config': {
            'device': str(device),
            'vocab_size': ntokens,
            'emsize': args.emsize,
            'nhid': args.nhid,
            'nlayers': args.nlayers,
            'nhead': args.nhead,
            'batch_size': args.batch_size,
            'bptt': args.bptt,
            'epochs': args.epochs,
            'total_params': total_params,
            'seed': args.seed,
        },
        'runs': []
    }

    for run in range(args.runs):
        print(f"\n{'='*80}")
        print(f"RUN {run + 1}/{args.runs}")
        print("="*80)

        # Reset model
        torch.manual_seed(args.seed + run)  # Different seed per run
        model_run = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
        model.load_state_dict(model_run.state_dict())

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        lr = args.lr
        run_results = {
            'epochs': [],
            'final_test': {}
        }

        total_train_time = 0

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.perf_counter()

            # Training
            train_metrics = train_epoch(lr)

            # Validation
            val_metrics = evaluate(val_data, "Validation")

            epoch_total_time = time.perf_counter() - epoch_start_time
            total_train_time += epoch_total_time

            # Memory usage
            cpu_mem, gpu_mem = get_memory_usage()

            epoch_result = {
                'epoch': epoch,
                'train': train_metrics,
                'validation': val_metrics,
                'total_epoch_time': epoch_total_time,
                'cpu_memory_mb': cpu_mem,
                'gpu_memory_mb': gpu_mem,
            }

            run_results['epochs'].append(epoch_result)

            print(f"Epoch {epoch:2d} | "
                  f"Train Loss: {train_metrics['loss']:.3f} | "
                  f"Train PPL: {train_metrics['perplexity']:8.2f} | "
                  f"Val Loss: {val_metrics['loss']:.3f} | "
                  f"Val PPL: {val_metrics['perplexity']:8.2f} | "
                  f"Time: {epoch_total_time:.2f}s | "
                  f"Tokens/sec: {train_metrics['tokens_per_second']:.0f}")

        # Final test evaluation
        print("\nRunning final test evaluation...")
        test_metrics = evaluate(test_data, "Test")
        run_results['final_test'] = test_metrics
        run_results['total_training_time'] = total_train_time

        cpu_mem, gpu_mem = get_memory_usage()
        run_results['peak_cpu_memory_mb'] = cpu_mem
        run_results['peak_gpu_memory_mb'] = gpu_mem

        print(f"\nTest Loss: {test_metrics['loss']:.3f} | "
              f"Test Perplexity: {test_metrics['perplexity']:8.2f} | "
              f"Total Training Time: {total_train_time:.2f}s")
        print(f"Peak CPU Memory: {cpu_mem:.2f} MB | Peak GPU Memory: {gpu_mem:.2f} MB")

        results['runs'].append(run_results)

    # Aggregate statistics
    test_perplexities = [run['final_test']['perplexity'] for run in results['runs']]
    test_times = [run['final_test']['time_seconds'] for run in results['runs']]
    train_times = [run['total_training_time'] for run in results['runs']]

    results['summary'] = {
        'test_perplexity': {
            'mean': sum(test_perplexities) / len(test_perplexities),
            'min': min(test_perplexities),
            'max': max(test_perplexities),
        },
        'inference_time_seconds': {
            'mean': sum(test_times) / len(test_times),
            'min': min(test_times),
            'max': max(test_times),
        },
        'training_time_seconds': {
            'mean': sum(train_times) / len(train_times),
            'min': min(train_times),
            'max': max(train_times),
        },
    }

    # Save results
    results['timestamp'] = datetime.now().isoformat()
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Test Perplexity: {results['summary']['test_perplexity']['mean']:.2f} "
          f"(±{results['summary']['test_perplexity']['max'] - results['summary']['test_perplexity']['min']:.2f})")
    print(f"Training Time: {results['summary']['training_time_seconds']['mean']:.2f}s "
          f"(±{results['summary']['training_time_seconds']['max'] - results['summary']['training_time_seconds']['min']:.2f}s)")
    print(f"Inference Time: {results['summary']['inference_time_seconds']['mean']:.3f}s "
          f"(±{results['summary']['inference_time_seconds']['max'] - results['summary']['inference_time_seconds']['min']:.3f}s)")
    print(f"\nResults saved to: {args.output}")
    print("="*80)

    return results

if __name__ == '__main__':
    run_benchmark()
