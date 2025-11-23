"""
Profiling script for PyTorch language model.
Identifies bottlenecks and generates performance reports.
"""
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import data
from model import TransformerModel

parser = argparse.ArgumentParser(description='Profile PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=4,
                    help='number of layers')
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
parser.add_argument('--profile-batches', type=int, default=100,
                    help='number of batches to profile')
parser.add_argument('--accel', action='store_true',
                    help='use accelerator if available')
args = parser.parse_args()

# Set random seed
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

# Build model
ntokens = len(corpus.dictionary)
model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
criterion = nn.NLLLoss()

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def profile_training():
    """Profile training loop with PyTorch profiler."""
    print("\n" + "="*80)
    print("PROFILING TRAINING")
    print("="*80)

    model.train()

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("training_loop"):
            for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
                if batch >= args.profile_batches:
                    break

                with record_function("get_batch"):
                    data, targets = get_batch(train_data, i)

                with record_function("forward"):
                    model.zero_grad()
                    output = model(data)
                    output = output.view(-1, ntokens)

                with record_function("loss_computation"):
                    loss = criterion(output, targets)

                with record_function("backward"):
                    loss.backward()

                with record_function("gradient_clipping"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

                with record_function("optimizer_step"):
                    for p in model.parameters():
                        p.data.add_(p.grad, alpha=-0.01)

    print("\n--- Top 10 CPU Time Consuming Operations ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    if device.type == 'cuda':
        print("\n--- Top 10 CUDA Time Consuming Operations ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\n--- Top 10 Memory Consuming Operations ---")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    # Export for TensorBoard
    prof.export_chrome_trace("trace_training.json")
    print("\nChrome trace exported to: trace_training.json")

    # Export stack traces
    prof.export_stacks("profiler_stacks.txt", "self_cpu_time_total")
    print("Stack traces exported to: profiler_stacks.txt")

def profile_inference():
    """Profile inference/evaluation loop."""
    print("\n" + "="*80)
    print("PROFILING INFERENCE")
    print("="*80)

    model.eval()

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("inference_loop"):
            with torch.no_grad():
                for i in range(0, min(val_data.size(0) - 1, args.profile_batches * args.bptt), args.bptt):
                    with record_function("get_batch"):
                        data, targets = get_batch(val_data, i)

                    with record_function("forward"):
                        output = model(data)
                        output = output.view(-1, ntokens)

                    with record_function("loss_computation"):
                        loss = criterion(output, targets)

    print("\n--- Top 10 CPU Time Consuming Operations ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    if device.type == 'cuda':
        print("\n--- Top 10 CUDA Time Consuming Operations ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Export for TensorBoard
    prof.export_chrome_trace("trace_inference.json")
    print("\nChrome trace exported to: trace_inference.json")

if __name__ == '__main__':
    print("\nModel Architecture:")
    print(f"- Vocabulary size: {ntokens}")
    print(f"- Embedding size: {args.emsize}")
    print(f"- Hidden size: {args.nhid}")
    print(f"- Number of layers: {args.nlayers}")
    print(f"- Number of heads: {args.nhead}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Sequence length: {args.bptt}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")

    profile_training()
    profile_inference()

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. View trace_training.json and trace_inference.json in chrome://tracing")
    print("2. Review profiler_stacks.txt for detailed stack traces")
    print("3. Use NVIDIA Nsight Systems for GPU profiling: nsys profile python profile_model.py")
