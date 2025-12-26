"""
Compare training performance: PyTorch vs MAX-Graph Transformer.
Trains both models for a few epochs and compares speed and loss.
"""

import argparse
import time
import math
import torch
import torch.nn as nn
import data
from transformer import TransformerModel as PyTorchTransformer
from maxGraph_transformer_trainable import TransformerModel as MAXTransformer


def batchify(data_tensor, bsz, device):
    """Arrange data into columns for batch processing."""
    nbatch = data_tensor.size(0) // bsz
    data_tensor = data_tensor.narrow(0, 0, nbatch * bsz)
    data_tensor = data_tensor.view(bsz, -1).t().contiguous()
    return data_tensor.to(device)


def get_batch(source, i, bptt):
    """Get a batch of data."""
    seq_len = min(bptt, len(source) - 1 - i)
    data_seq = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data_seq, target


def train_epoch(model, train_data, optimizer, criterion, ntokens, bptt):
    """Train for one epoch and return average loss and time."""
    model.train()
    total_loss = 0.
    num_batches = 0
    start_time = time.time()

    for i in range(0, train_data.size(0) - 1, bptt):
        # Get batch
        data, targets = get_batch(train_data, i, bptt)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)

        # Compute loss
        loss = criterion(output.view(-1, ntokens), targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # Update weights
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    elapsed_time = time.time() - start_time
    avg_loss = total_loss / num_batches

    return avg_loss, elapsed_time


def evaluate(model, eval_data, criterion, ntokens, bptt):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()

    return total_loss / (len(eval_data) - 1)


def main():
    parser = argparse.ArgumentParser(description='Compare PyTorch vs MAX-Graph training')
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--emsize', type=int, default=200,
                        help='embedding dimension')
    parser.add_argument('--nhid', type=int, default=200,
                        help='hidden units in feedforward')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=2,
                        help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate')
    parser.add_argument('--lr', type=float, default=5.0,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("TRAINING COMPARISON: PyTorch vs MAX-Graph Transformer")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load data
    print("Loading WikiText-2 dataset...")
    corpus = data.Corpus(args.data)

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)

    ntokens = len(corpus.dictionary)
    print(f"Vocabulary size: {ntokens}")
    print()

    # Model configuration
    print("Model Configuration:")
    print(f"  Embedding dim: {args.emsize}")
    print(f"  Hidden dim: {args.nhid}")
    print(f"  Layers: {args.nlayers}")
    print(f"  Attention heads: {args.nhead}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.bptt}")
    print(f"  Training epochs: {args.epochs}")
    print()

    # Loss function
    criterion = nn.NLLLoss()

    results = {}

    # ========================================================================
    # Train PyTorch Model
    # ========================================================================
    print("=" * 80)
    print("TRAINING PYTORCH MODEL")
    print("=" * 80)

    torch.manual_seed(args.seed)  # Reset seed for fair comparison

    pytorch_model = PyTorchTransformer(
        ntoken=ntokens,
        ninp=args.emsize,
        nhead=args.nhead,
        nhid=args.nhid,
        nlayers=args.nlayers,
        dropout=args.dropout
    ).to(device)

    pytorch_optimizer = torch.optim.SGD(pytorch_model.parameters(), lr=args.lr)
    pytorch_scheduler = torch.optim.lr_scheduler.StepLR(pytorch_optimizer, 1.0, gamma=0.95)

    pytorch_times = []
    pytorch_train_losses = []
    pytorch_val_losses = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_time = train_epoch(
            pytorch_model, train_data, pytorch_optimizer, criterion,
            ntokens, args.bptt
        )
        val_loss = evaluate(pytorch_model, val_data, criterion, ntokens, args.bptt)

        pytorch_times.append(train_time)
        pytorch_train_losses.append(train_loss)
        pytorch_val_losses.append(val_loss)

        print(f"Epoch {epoch:2d} | Time: {train_time:6.2f}s | "
              f"Train Loss: {train_loss:5.2f} | Train PPL: {math.exp(train_loss):8.2f} | "
              f"Val Loss: {val_loss:5.2f} | Val PPL: {math.exp(val_loss):8.2f}")

        pytorch_scheduler.step()

    pytorch_total_time = sum(pytorch_times)
    pytorch_avg_loss = sum(pytorch_train_losses) / len(pytorch_train_losses)

    print()
    print(f"PyTorch Total Training Time: {pytorch_total_time:.2f}s")
    print(f"PyTorch Average Train Loss: {pytorch_avg_loss:.4f}")
    print()

    # Free PyTorch model and aggressively clean up memory
    del pytorch_model, pytorch_optimizer, pytorch_scheduler
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Force garbage collection
    import gc
    gc.collect()

    # Wait a moment for cleanup
    import time as time_module
    time_module.sleep(2)

    print(f"GPU Memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.2f} MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f} MB")
    print()

    # ========================================================================
    # Train MAX-Graph Model
    # ========================================================================
    print("=" * 80)
    print("TRAINING MAX-GRAPH MODEL")
    print("=" * 80)

    torch.manual_seed(args.seed)  # Reset seed for fair comparison

    max_model = MAXTransformer(
        ntoken=ntokens,
        ninp=args.emsize,
        nhead=args.nhead,
        nhid=args.nhid,
        nlayers=args.nlayers,
        dropout=args.dropout
    ).to(device)

    max_optimizer = torch.optim.SGD(max_model.parameters(), lr=args.lr)
    max_scheduler = torch.optim.lr_scheduler.StepLR(max_optimizer, 1.0, gamma=0.95)

    max_times = []
    max_train_losses = []
    max_val_losses = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_time = train_epoch(
            max_model, train_data, max_optimizer, criterion,
            ntokens, args.bptt
        )
        val_loss = evaluate(max_model, val_data, criterion, ntokens, args.bptt)

        max_times.append(train_time)
        max_train_losses.append(train_loss)
        max_val_losses.append(val_loss)

        print(f"Epoch {epoch:2d} | Time: {train_time:6.2f}s | "
              f"Train Loss: {train_loss:5.2f} | Train PPL: {math.exp(train_loss):8.2f} | "
              f"Val Loss: {val_loss:5.2f} | Val PPL: {math.exp(val_loss):8.2f}")

        max_scheduler.step()

    max_total_time = sum(max_times)
    max_avg_loss = sum(max_train_losses) / len(max_train_losses)

    print()
    print(f"MAX-Graph Total Training Time: {max_total_time:.2f}s")
    print(f"MAX-Graph Average Train Loss: {max_avg_loss:.4f}")
    print()

    # ========================================================================
    # Comparison Summary
    # ========================================================================
    speedup = pytorch_total_time / max_total_time
    faster = "MAX-Graph" if speedup > 1.0 else "PyTorch"

    print("=" * 80)
    print("TRAINING COMPARISON SUMMARY")
    print("=" * 80)
    print()
    print(f"PyTorch Total Time:    {pytorch_total_time:8.2f}s")
    print(f"MAX-Graph Total Time:  {max_total_time:8.2f}s")
    print(f"Speedup:               {speedup:8.2f}x ({faster} faster)")
    print()
    print(f"PyTorch Avg Loss:      {pytorch_avg_loss:8.4f}")
    print(f"MAX-Graph Avg Loss:    {max_avg_loss:8.4f}")
    print()

    # Per-epoch comparison
    print("Per-Epoch Time Comparison:")
    print("-" * 80)
    print(f"{'Epoch':<8} {'PyTorch (s)':<15} {'MAX-Graph (s)':<15} {'Speedup':<10}")
    print("-" * 80)
    for i in range(args.epochs):
        epoch_speedup = pytorch_times[i] / max_times[i]
        print(f"{i+1:<8} {pytorch_times[i]:<15.2f} {max_times[i]:<15.2f} {epoch_speedup:<10.2f}x")
    print("-" * 80)
    print()

    # Save results
    results = {
        'config': vars(args),
        'pytorch': {
            'total_time': pytorch_total_time,
            'avg_loss': pytorch_avg_loss,
            'epoch_times': pytorch_times,
            'train_losses': pytorch_train_losses,
            'val_losses': pytorch_val_losses,
        },
        'max_graph': {
            'total_time': max_total_time,
            'avg_loss': max_avg_loss,
            'epoch_times': max_times,
            'train_losses': max_train_losses,
            'val_losses': max_val_losses,
        },
        'speedup': speedup,
        'faster': faster,
    }

    import json
    with open('training_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: training_comparison_results.json")
    print("=" * 80)


if __name__ == '__main__':
    main()
