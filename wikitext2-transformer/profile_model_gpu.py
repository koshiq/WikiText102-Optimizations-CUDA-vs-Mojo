"""
Profiling script for PyTorch language model with explicit GPU support.
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

parser = argparse.ArgumentParser(description='Profile PyTorch Transformer Language Model on GPU')
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
parser.add_argument('--cpu', action='store_true',
                    help='use CPU instead of GPU')
args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)

# Device setup - explicitly use CUDA unless --cpu is specified
if args.cpu:
    device = torch.device("cpu")
    print("Using device: CPU (forced)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

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

                # Synchronize GPU to ensure accurate timing
                if device.type == 'cuda':
                    torch.cuda.synchronize()

    print("\n--- Top 10 CPU Time Consuming Operations ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    if device.type == 'cuda':
        print("\n--- Top 10 CUDA Time Consuming Operations ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\n--- Top 10 Memory Consuming Operations ---")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    # Export for TensorBoard
    prof.export_chrome_trace("trace_training_gpu.json")
    print("\nChrome trace exported to: trace_training_gpu.json")

    # Export stack traces
    prof.export_stacks("profiler_stacks_gpu.txt", "self_cpu_time_total")
    print("Stack traces exported to: profiler_stacks_gpu.txt")

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

                    # Synchronize GPU to ensure accurate timing
                    if device.type == 'cuda':
                        torch.cuda.synchronize()

    print("\n--- Top 10 CPU Time Consuming Operations ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    if device.type == 'cuda':
        print("\n--- Top 10 CUDA Time Consuming Operations ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Export for TensorBoard
    prof.export_chrome_trace("trace_inference_gpu.json")
    print("\nChrome trace exported to: trace_inference_gpu.json")

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
    print("\nGenerated files:")
    print("1. trace_training_gpu.json - View in chrome://tracing")
    print("2. trace_inference_gpu.json - View in chrome://tracing")
    print("3. profiler_stacks_gpu.txt - Detailed stack traces")

    if device.type == 'cuda':
        print("\nFor detailed GPU profiling with Nsight Systems, run:")
        print("  nsys profile --trace=cuda,nvtx,cudnn,cublas --cuda-memory-usage=true --output=gpu_profile python profile_model_gpu.py")
