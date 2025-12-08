# WikiText102 Transformer — CUDA vs Mojo (MAX)

This repository contains an implementation of a Transformer model trained / evaluated on WikiText-2/102 and an experimental comparison between two optimization approaches:

- CUDA custom kernels (C++/CUDA) used from Python / PyTorch
- Mojo / MAX graph accelerated implementations (Mojo ecosystem, Modular MAX)

The project also explores implementing low-level kernels (GEMM, SoftMax, LayerNorm) to measure performance differences and target optimization opportunities highlighted by tracing and profiling.

**Project Goals**
- Compare inference/training performance between PyTorch (CuBLAS/cuDNN) and Mojo/MAX graph on the same Transformer architecture.
- Implement custom CUDA kernels for bottleneck ops (GEMM, SoftMax, LayerNorm) and measure benefits.
- Provide repeatable benchmarking and profiling harnesses.

**Milestones & Status**
- Fix C++ pybind bindings and expose parameters: **Completed**
- Python wrappers to present C++ tensors as `nn.Parameter` and device helpers: **Completed**
- Clean up duplicate kernels and package extension for editable install: **Completed**
- Build & install extension in editable mode and run short benchmarks: **Completed**
- Add `final-publish` branch snapshot with cleaned repository (no venv): **Completed**
- Profiling + analysis to identify GEMM/SoftMax/LayerNorm as primary hotspots: **Completed**
- Implement further kernel performance optimizations & MAX autograd support: **In progress / Planned**

**Repository (high-level) structure**
- `cuda/`: C++/CUDA extension sources and packaging helpers.
  - `cuda/custom_ops.cpp`: C++ bindings for custom ops.
  - `cuda/cuda_kernels/`: CUDA kernel sources (`gemm_kernel.cu`, `softmax_kernel.cu`, `layernorm_kernel.cu`).
- `mojo/`: Mojo sample kernels and Mojo helpers for the MAX experiments.
- `customCudaKernel_transformer.py`: PyTorch model variant that uses the C++/CUDA extension via `WrappedCustomLinear` and `WrappedCustomLayerNorm`.
- `transformer.py`: Reference PyTorch Transformer implementation used for baseline runs.
- `maxGraph_transformer.py`, `max_as_mojo_transformer.py`, `max_transformer.py`: MAX/Mojo graph variants and helpers (note: some MAX files are inference-only and detach outputs).
- `benchmark.py`: End-to-end micro/throughput benchmark harness.
- `comparison.py`: Script to copy weights and compare PyTorch vs MAX vs CUDA runs.
- `train.py`: Training driver (small experiments).
- `profiler.py`: Tracing and profiler harness used to identify hotspots.
- `data/`: WikiText-2 sample data used for experiments.
- `model.pt`, `trace_*.json`, `transformer_benchmark_results.json`: artifacts and saved model/trace data (some are large; consider Git LFS for production pushes).

**Setting up the development environment (venv `ML`)**
1. Create and activate the venv (we use `ML` in this repo):

```bash
python3 -m venv ML
source ML/bin/activate
```

2. Install Python dependencies:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

3. Build / install the CUDA extension in editable mode (recommended):

```bash
python -m pip install -e ./cuda --no-build-isolation
```

Note: building the extension requires a working CUDA toolchain (nvcc) and a matching C/C++ compiler. If your system CUDA differs from the PyTorch CUDA, you'll get a warning but builds usually proceed; ensure `LD_LIBRARY_PATH` includes the venv torch libs if import errors (libc10) appear.

**Installing Modular / Mojo (MAX) nightly**
To run MAX/Mojo experiments (Modular's pre-release packages), install the nightly index:

```bash
pip install --pre modular \
  --index-url https://dl.modular.com/public/nightly/python/simple/
```

This installs the `modular` package and MAX tooling used by the `mojo/` samples.


```bash
source ML/bin/activate
python -m pip install -e ./cuda --no-build-isolation
```

- Run a quick benchmark (compare runtime variants):

```bash
python benchmark.py   # or --model pytorch / --model max
```

- Run the comparison harness (short run):

```bash
python comparison.py
```

- Profile a training/inference run (creates traces):

```bash
python profiler.py --mode inference --out trace_inference.json
python profiler.py --mode training --out trace_training.json
```

- Train (small experiments):

```bash
python train.py --epochs 1 --batch-size 8
```

**Transformer variants (what to use when)**
- `transformer.py`: Standard PyTorch baseline implementation (uses `torch.nn.Linear`, `LayerNorm`, `F.softmax`).
- `customCudaKernel_transformer.py`: Uses the C++/CUDA extension for linear/softmax/layernorm primitives — used to validate the custom kernels and measure end-to-end differences.
- `maxGraph_transformer.py`: MAX (Mojo) graph accelerated variant, currently focused on inference (some ops detach and pre-allocate buffers). Good for exploring MAX performance characteristics.
- `max_as_mojo_transformer.py` / `max_transformer.py`: additional MAX/Mojo wrappers and prototypes (see file headers for usage details).

**Custom kernels and why**
Profiling showed that matrix multiply (GEMM), SoftMax, and LayerNorm occupy the majority of time in the Transformer inner loops. This repo includes prototype CUDA kernels for these ops in `cuda/cuda_kernels/` so we can:

- benchmark our kernels vs cuBLAS/cuDNN
- iterate on tiling/packing strategies for GEMM
- experiment with numerically stable and fused softmax + attention kernels


---
