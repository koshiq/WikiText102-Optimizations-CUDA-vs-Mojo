"""
Modal deployment script for training WikiText-2 Transformer on A100 GPU.

This script runs the training on Modal's cloud infrastructure with A100 GPU support.
"""

import modal

# Create a Modal app
app = modal.App("wikitext2-transformer-training")

# Define the image with all dependencies including MAX Engine
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "build-essential")
    .run_commands(
        # Install MAX Engine for Mojo kernels
        "wget -qO- https://get.modular.com | sh -",
        "modular auth $MODULAR_API_KEY",
        "modular install max",
    )
    .pip_install(
        "torch>=2.6",
        "psutil>=5.9.0",
        "max>=24.6",  # MAX Engine Python package
    )
    .env({"MAX_ENABLE_GPU": "1"})  # Enable GPU for MAX Engine
)

# Mount the local code directory
code_mount = modal.Mount.from_local_dir(
    "./wikitext2-transformer",
    remote_path="/root/wikitext2-transformer"
)

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="40GB"),  # Use A100 40GB GPU
    mounts=[code_mount],
    timeout=3600 * 4,  # 4 hours timeout
    secrets=[modal.Secret.from_name("modular-api-key")],  # Modular API key for MAX Engine
)
def train_model(
    epochs: int = 40,
    batch_size: int = 20,
    emsize: int = 200,
    nhid: int = 200,
    nlayers: int = 4,
    nhead: int = 2,
    lr: float = 20.0,
    dropout: float = 0.2,
    use_optimizer: bool = True,
    use_mojo: bool = True,
):
    """
    Train the Transformer language model on WikiText-2 dataset.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        emsize: Size of word embeddings
        nhid: Number of hidden units per layer
        nlayers: Number of transformer layers
        nhead: Number of attention heads
        lr: Initial learning rate
        dropout: Dropout rate
        use_optimizer: Whether to use AdamW optimizer
        use_mojo: Whether to use Mojo-optimized kernels
    """
    import sys
    import os
    import subprocess

    # Change to the code directory
    os.chdir("/root/wikitext2-transformer")

    # Build command
    cmd = [
        "python", "main.py",
        "--data", "./data/wikitext-2",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--emsize", str(emsize),
        "--nhid", str(nhid),
        "--nlayers", str(nlayers),
        "--nhead", str(nhead),
        "--lr", str(lr),
        "--dropout", str(dropout),
        "--save", "/root/model.pt",
        "--accel",  # Enable GPU acceleration
    ]

    if use_optimizer:
        cmd.append("--use-optimizer")

    if use_mojo:
        cmd.append("--use-mojo")

    print(f"Running command: {' '.join(cmd)}")
    print("="*80)

    # Run training
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    # Read and return the trained model
    with open("/root/model.pt", "rb") as f:
        model_bytes = f.read()

    return model_bytes


@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    mounts=[code_mount],
    timeout=3600,
)
def benchmark_model(
    epochs: int = 5,
    runs: int = 3,
    batch_size: int = 20,
):
    """
    Run comprehensive benchmarks on the model.

    Args:
        epochs: Number of epochs to benchmark
        runs: Number of benchmark runs for averaging
        batch_size: Batch size for benchmarking
    """
    import subprocess
    import json
    import os

    os.chdir("/root/wikitext2-transformer")

    cmd = [
        "python", "benchmark.py",
        "--data", "./data/wikitext-2",
        "--epochs", str(epochs),
        "--runs", str(runs),
        "--batch_size", str(batch_size),
        "--output", "/root/benchmark_results.json",
        "--accel",
    ]

    print(f"Running benchmark: {' '.join(cmd)}")
    print("="*80)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed with return code {result.returncode}")

    # Read and return results
    with open("/root/benchmark_results.json", "r") as f:
        results = json.load(f)

    return results


@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    mounts=[code_mount],
    timeout=3600,
    secrets=[modal.Secret.from_name("modular-api-key")],
)
def benchmark_mojo_vs_cuda(
    batch_size: int = 20,
    seq_len: int = 35,
    hidden_size: int = 200,
    warmup: int = 10,
    iterations: int = 100,
):
    """
    Benchmark Mojo kernels vs PyTorch CUDA.

    Args:
        batch_size: Batch size for testing
        seq_len: Sequence length
        hidden_size: Hidden dimension
        warmup: Warmup iterations
        iterations: Benchmark iterations
    """
    import subprocess
    import json
    import os

    os.chdir("/root/wikitext2-transformer")

    cmd = [
        "python", "-m", "mojo.benchmark_mojo_vs_cuda",
        "--device", "cuda",
        "--batch-size", str(batch_size),
        "--seq-len", str(seq_len),
        "--hidden-size", str(hidden_size),
        "--warmup", str(warmup),
        "--iterations", str(iterations),
        "--output", "/root/mojo_benchmark.json",
    ]

    print(f"Running Mojo vs CUDA benchmark: {' '.join(cmd)}")
    print("="*80)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed with return code {result.returncode}")

    # Read and return results
    with open("/root/mojo_benchmark.json", "r") as f:
        results = json.load(f)

    return results


@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    mounts=[code_mount],
    timeout=3600,
    secrets=[modal.Secret.from_name("modular-api-key")],
)
def profile_model(profile_batches: int = 100):
    """
    Profile the model to identify performance bottlenecks.

    Args:
        profile_batches: Number of batches to profile
    """
    import subprocess
    import os

    os.chdir("/root/wikitext2-transformer")

    cmd = [
        "python", "-m", "cuda.profile_model_gpu",
        "--data", "./data/wikitext-2",
        "--profile-batches", str(profile_batches),
    ]

    print(f"Running profiler: {' '.join(cmd)}")
    print("="*80)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Profiling failed with return code {result.returncode}")

    # Read profiling traces
    traces = {}
    if os.path.exists("/root/wikitext2-transformer/trace_training.json"):
        with open("/root/wikitext2-transformer/trace_training.json", "r") as f:
            traces["training"] = f.read()

    if os.path.exists("/root/wikitext2-transformer/trace_inference.json"):
        with open("/root/wikitext2-transformer/trace_inference.json", "r") as f:
            traces["inference"] = f.read()

    if os.path.exists("/root/wikitext2-transformer/profiler_stacks.txt"):
        with open("/root/wikitext2-transformer/profiler_stacks.txt", "r") as f:
            traces["stacks"] = f.read()

    return traces


@app.local_entrypoint()
def main(
    mode: str = "train",
    epochs: int = 40,
    batch_size: int = 20,
    runs: int = 3,
    use_mojo: bool = False,
):
    """
    Local entrypoint for running Modal functions.

    Args:
        mode: Operation mode: 'train', 'benchmark', 'profile', or 'mojo-benchmark'
        epochs: Number of epochs
        batch_size: Batch size
        runs: Number of benchmark runs (for benchmark mode)
        use_mojo: Whether to use Mojo-optimized kernels
    """
    print(f"Running in {mode} mode on Modal A100 GPU...")
    if use_mojo:
        print("Mojo kernels enabled")

    if mode == "train":
        print("Starting training...")
        model_bytes = train_model.remote(
            epochs=epochs,
            batch_size=batch_size,
            use_mojo=use_mojo,
        )

        # Save model locally
        output_path = "./model_a100.pt"
        with open(output_path, "wb") as f:
            f.write(model_bytes)
        print(f"\nModel saved to {output_path}")
        print(f"Model size: {len(model_bytes) / 1024 / 1024:.2f} MB")

    elif mode == "benchmark":
        print("Starting benchmark...")
        results = benchmark_model.remote(
            epochs=epochs,
            runs=runs,
            batch_size=batch_size,
            use_mojo=use_mojo,
        )

        # Save results locally
        import json
        output_path = "./benchmark_results_a100.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark results saved to {output_path}")

        # Print summary
        if "summary" in results:
            print("\n" + "="*80)
            print("BENCHMARK SUMMARY")
            print("="*80)
            summary = results["summary"]
            if "test_perplexity" in summary:
                print(f"Test Perplexity: {summary['test_perplexity']['mean']:.2f}")
            if "training_time_seconds" in summary:
                print(f"Training Time: {summary['training_time_seconds']['mean']:.2f}s")
            if "inference_time_seconds" in summary:
                print(f"Inference Time: {summary['inference_time_seconds']['mean']:.3f}s")

    elif mode == "mojo-benchmark":
        print("Starting Mojo vs CUDA kernel benchmark...")
        results = benchmark_mojo_vs_cuda.remote(
            batch_size=batch_size,
        )

        # Save results locally
        import json
        output_path = "./mojo_vs_cuda_benchmark.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark results saved to {output_path}")

        # Print summary
        if "summary" in results:
            print("\n" + "="*80)
            print("MOJO VS CUDA BENCHMARK SUMMARY")
            print("="*80)
            summary = results["summary"]
            print(f"Total Mojo Time:    {summary['total_mojo_time_ms']:.3f} ms")
            print(f"Total PyTorch Time: {summary['total_pytorch_time_ms']:.3f} ms")
            print(f"Overall Speedup:    {summary['overall_speedup']:.2f}x")
            print("\nPer-Operation Results:")
            for bench in results["benchmarks"]:
                print(f"  {bench['operation']:12s}: {bench['speedup']:.2f}x ({bench['faster']} faster)")

    elif mode == "profile":
        print("Starting profiling...")
        traces = profile_model.remote()

        # Save traces locally
        for name, content in traces.items():
            output_path = f"./{name}_trace_a100.json" if name != "stacks" else "./profiler_stacks_a100.txt"
            with open(output_path, "w") as f:
                f.write(content)
            print(f"Saved {name} trace to {output_path}")

    else:
        print(f"Unknown mode: {mode}")
        print("Valid modes: train, benchmark, profile, mojo-benchmark")
