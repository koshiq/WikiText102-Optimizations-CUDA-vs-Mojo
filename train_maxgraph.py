"""
Train Transformer model with MAX Graph acceleration.
Based on the original train.py but uses trainable MAX-Graph operations.
"""

import argparse
import time
import math
import torch
import torch.nn as nn
import data
from maxGraph_transformer_trainable import TransformerModel


def batchify(data_tensor, bsz, device):
    """Arrange data into columns for batch processing."""
    nbatch = data_tensor.size(0) // bsz
    data_tensor = data_tensor.narrow(0, 0, nbatch * bsz)
    data_tensor = data_tensor.view(bsz, -1).t().contiguous()
    return data_tensor.to(device)


def get_batch(source, i, bptt):
    """Get a batch of data for training/evaluation."""
    seq_len = min(bptt, len(source) - 1 - i)
    data_seq = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data_seq, target


def train(model, train_data, optimizer, criterion, ntokens, bptt, epoch, log_interval):
    """Train for one epoch."""
    model.train()
    total_loss = 0.
    start_time = time.time()

    num_batches = len(train_data) // bptt

    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
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

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {optimizer.param_groups[0]["lr"]:02.2f} | ms/batch {elapsed * 1000 / log_interval:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
            total_loss = 0
            start_time = time.time()


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
    parser = argparse.ArgumentParser(description='Train WikiText-2 Transformer with MAX Graph')
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--emsize', type=int, default=200,
                        help='embedding dimension')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units in feedforward')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=2,
                        help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate')
    parser.add_argument('--lr', type=float, default=5.0,
                        help='initial learning rate')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='report interval')
    parser.add_argument('--save', type=str, default='model_maxgraph.pt',
                        help='path to save the final model')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading WikiText-2 dataset...")
    corpus = data.Corpus(args.data)

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)

    ntokens = len(corpus.dictionary)
    print(f"Vocabulary size: {ntokens}")

    # Create model
    print("Creating MAX-Graph accelerated Transformer model...")
    model = TransformerModel(
        ntoken=ntokens,
        ninp=args.emsize,
        nhead=args.nhead,
        nhid=args.nhid,
        nlayers=args.nlayers,
        dropout=args.dropout
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # Training loop
    best_val_loss = float('inf')
    print("\nStarting training with MAX Graph acceleration...")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Train
        train(model, train_data, optimizer, criterion, ntokens, args.bptt,
              epoch, args.log_interval)

        # Evaluate on validation set
        val_loss = evaluate(model, val_data, criterion, ntokens, args.bptt)

        print('-' * 80)
        print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {math.exp(val_loss):8.2f}')
        print('-' * 80)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save)
            print(f"Saved best model to {args.save}")

        # Learning rate scheduling
        scheduler.step()

    # Test
    print("\nTesting on test set...")
    model.load_state_dict(torch.load(args.save))
    test_loss = evaluate(model, test_data, criterion, ntokens, args.bptt)
    print('=' * 80)
    print(f'| End of training | test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):8.2f}')
    print('=' * 80)


if __name__ == '__main__':
    main()
