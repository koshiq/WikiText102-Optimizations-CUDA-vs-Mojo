"""
Modal deployment script for WikiText-2 Transformer on A100 GPU.

Includes:
 - Mojo stdlib fix (copy to MOJO_PATH so mojo.paths loads correctly)
 - Full deep diagnostics (env vars, sys.path, filesystem map)
 - Correct heredoc import verification in subprocess
 - Actual benchmark execution & result parsing
 - Forced rebuild marker so Modal does NOT cache bad images
"""

import os
import sys
import subprocess
import json
from modal import App, Image, gpu

app = App("wikitext2-transformer-training")

# =====================================================================
# PATH CONSTANTS
# =====================================================================
PIXI_ROOT = "/root/mojo_project"
PIXI_ENV  = f"{PIXI_ROOT}/.pixi/envs/default"
SITE_PACKAGES = f"{PIXI_ENV}/lib/python3.13/site-packages"

MOJO_SITE   = f"{SITE_PACKAGES}/mojo"
MOJO_PATH   = f"{PIXI_ENV}/lib/mojo"
MOJO_TARGET = f"{MOJO_PATH}/mojo"    # copy entire mojo stdlib here

PROJECT_ROOT = "/root/wikitext2-transformer"

# =====================================================================
# IMAGE BUILD
# =====================================================================

image = (
    Image.debian_slim(python_version="3.13")
    .apt_install("curl", "git", "rsync", "build-essential")
    .run_commands(
        "echo 'REBUILD_TRIGGER: 2025-12-02'",

        # Install pixi
        "curl -fsSL https://pixi.sh/install.sh -o /tmp/install_pixi.sh",
        "bash /tmp/install_pixi.sh",

        # Create pixi project
        f"mkdir -p {PIXI_ROOT}",

        # Init pixi with MAX channel
        f"export PATH=$HOME/.pixi/bin:$PATH && cd {PIXI_ROOT} && "
        f"pixi init -c https://conda.modular.com/max-nightly -c conda-forge",

        # Install Pixi environment: Python 3.13 + pillow + modular (MAX runtime)
        f"export PATH=$HOME/.pixi/bin:$PATH && cd {PIXI_ROOT} && "
        f"pixi add python==3.13 pillow>=11.0.0 modular pip setuptools wheel",

        f"{PIXI_ENV}/bin/python -m pip install --upgrade pip",
        f"{PIXI_ENV}/bin/pip install torch==2.9.1 torchvision==0.24.1",


        # ---------------------------------------------------------
        # 3. Copy Mojo stdlib into MAX runtime path
        # ---------------------------------------------------------
        f"mkdir -p {MOJO_TARGET}",
        f"rsync -av {MOJO_SITE}/ {MOJO_TARGET}/",

        # ---------------------------------------------------------
        # 4. Verify MAX + Mojo load successfully
        # ---------------------------------------------------------
        f"{PIXI_ENV}/bin/python -c 'import max, mojo.paths; print(\"[BUILD] MAX + Mojo OK\")'"
    )



# works with my cpu
# image = (
#     Image.debian_slim(python_version="3.13")
#     .apt_install("curl", "git", "rsync", "build-essential")
#     .run_commands(
#         # 🔥 FORCE REBUILD MARKER — MODIFY THIS COMMENT TO REBUILD
#         "echo 'REBUILD_TRIGGER: 2025-12-01'",

#         # Install pixi
#         "curl -fsSL https://pixi.sh/install.sh -o /tmp/install_pixi.sh",
#         "bash /tmp/install_pixi.sh",

#         # Create pixi project
#         f"mkdir -p {PIXI_ROOT}",

#         # Init pixi environment with Modular MAX channel
#         f"export PATH=$HOME/.pixi/bin:$PATH && cd {PIXI_ROOT} && "
#         f"pixi init -c https://conda.modular.com/max-nightly -c conda-forge",

#         # Install core components inside pixi env
#         # f"export PATH=$HOME/.pixi/bin:$PATH && cd {PIXI_ROOT} && pixi add python==3.11",
#         f"export PATH=$HOME/.pixi/bin:$PATH && cd {PIXI_ROOT} && pixi add python==3.13 pillow>=11.0.0",
#         f"export PATH=$HOME/.pixi/bin:$PATH && cd {PIXI_ROOT} && pixi add modular",
#         # f"export PATH=$HOME/.pixi/bin:$PATH && cd {PIXI_ROOT} && pixi add pytorch",
#         # Add the specific CUDA runtime libraries that PyTorch uses
#         # f"export PATH=$HOME/.pixi/bin:$PATH && cd {PIXI_ROOT} && pixi add cuda-version=12.1",

#         # Add the pytorch package that is built against the above cuda version
#         f"export PATH=$HOME/.pixi/bin:$PATH && cd {PIXI_ROOT} && pixi add pytorch",


#         # Copy Mojo standard library to MAX runtime location
#         f"mkdir -p {MOJO_TARGET}",
#         f"rsync -av {MOJO_SITE}/ {MOJO_TARGET}/",

#         # Build-time verification
#         f"export PATH=$HOME/.pixi/bin:$PATH && cd {PIXI_ROOT} && "
#         f"pixi run python -c 'import max, mojo.paths; print(\"[BUILD] MAX + Mojo OK\")'"
#     )
    .env({
        "MAX_ENABLE_GPU": "1",
        "PATH": f"/root/.pixi/bin:{PIXI_ENV}/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": f"{SITE_PACKAGES}:{PROJECT_ROOT}:{PROJECT_ROOT}/mojo_hpml:{MOJO_PATH}",
        "LD_LIBRARY_PATH": f"{PIXI_ENV}/lib",
        "MODULAR_HOME": PIXI_ENV,
        "MAX_HOME": PIXI_ENV,
        "MOJO_PATH": MOJO_PATH,
    })
    .add_local_dir("./wikitext2-transformer", remote_path=PROJECT_ROOT)
)

# =====================================================================
# FUNCTION: FULL DEBUG + REAL BENCHMARK
# =====================================================================
@app.function(image=image, gpu="A100-40GB", timeout=3600)
def benchmark_mojo_vs_cuda(
    batch_size: int = 20,
    seq_len: int = 35,
    hidden_size: int = 200,
    warmup: int = 10,
    iterations: int = 100,
):

    print("=" * 80)
    print("🔧 MAX + MOJO ENVIRONMENT CHECK")
    print("=" * 80)

    # -----------------------------------------------------------------
    # 1. Print ALL environment variables
    # -----------------------------------------------------------------
    print("\n📌 ENVIRONMENT VARIABLES:")
    for key in ["PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "MODULAR_HOME", "MOJO_PATH"]:
        print(f"{key}: {os.environ.get(key)}")

    # -----------------------------------------------------------------
    # 2. sys.path
    # -----------------------------------------------------------------
    print("\n📌 sys.path in MAIN PROCESS:")
    for i, p in enumerate(sys.path):
        print(f"  [{i}] {p}")

    # -----------------------------------------------------------------
    # 3. Filesystem diagnostics
    # -----------------------------------------------------------------
    print("\n📌 FILESYSTEM CHECK:")
    paths = [
        (MOJO_SITE, "MOJO stdlib exists"),
        (MOJO_PATH, "MOJO_PATH exists"),
        (MOJO_TARGET, "MOJO_TARGET exists"),
        (f"{MOJO_TARGET}/paths.py", "mojo.paths exists"),
    ]
    for p, name in paths:
        print(f"{name}: {os.path.exists(p)}")

    # -----------------------------------------------------------------
    # 4. Deep subprocess import test using HEREDOC
    # -----------------------------------------------------------------
    print("\n📌 Running subprocess import check...")
    CHECK = r"""
import sys, os, pkgutil, importlib, importlib.util

print("=== PYTHON EXECUTABLE ===")
print(sys.executable)
print()

print("=== sys.prefix ===")
print(sys.prefix)
print()

print("=== sys.path ===")
for i,p in enumerate(sys.path):
    print(f"[{i}] {p}")
print()

print("=== Searching for 'mojo_hpml' package ===")
mojo_spec = importlib.util.find_spec("mojo_hpml")
print("mojo_spec:", mojo_spec)
print("origin:", getattr(mojo_spec, 'origin', None))
print("submodule_search_locations:", getattr(mojo_spec, 'submodule_search_locations', None))
print()

print("=== CONTENTS OF mojo_hpml package directory ===")
if mojo_spec and mojo_spec.submodule_search_locations:
    import os
    for loc in mojo_spec.submodule_search_locations:
        print("DIR:", loc)
        try:
            print(os.listdir(loc))
        except Exception as e:
            print("ERROR listing:", e)
print()

print("=== Trying: import mojo_hpml ===")
import mojo_hpml
print("mojo_hpml module:", mojo_hpml)
print("mojo_hpml.__file__:", getattr(mojo_hpml, '__file__', None))
print("mojo_hpml.__path__:", getattr(mojo_hpml, '__path__', None))
print()
"""



    heredoc = f"""python3 - << 'EOF'
{CHECK}
EOF
"""

    proc = subprocess.run(
        heredoc,
        shell=True,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )

    print("\n📤 Subprocess STDOUT:")
    print(proc.stdout)

    if proc.stderr:
        print("\n📥 Subprocess STDERR:")
        print(proc.stderr)

    if proc.returncode != 0:
        print("\n❌ Environment FAILED")
        raise RuntimeError("Environment setup failed")

    print("\n✅ Environment OK (mojo.paths import works!)")

    # -----------------------------------------------------------------
    # 5. RUN THE ACTUAL BENCHMARK
    # -----------------------------------------------------------------
    print("\n🚀 Running benchmark...")

    benchmark_script = f"{PROJECT_ROOT}/mojo_hpml/benchmark_mojo_vs_cuda.py"
    output_file = "/root/mojo_benchmark.json"

    bench_cmd = [
        "python3",
        # "/usr/local/bin/python3",
        benchmark_script,
        "--device", "cuda",
        "--batch-size", str(batch_size),
        "--seq-len", str(seq_len),
        "--hidden-size", str(hidden_size),
        "--warmup", str(warmup),
        "--iterations", str(iterations),
        "--output", output_file,
    ]

    try:
        subprocess.run(bench_cmd, check=True, env=os.environ.copy())
    except subprocess.CalledProcessError as e:
        print("❌ Benchmark crashed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    # -----------------------------------------------------------------
    # 6. RETURN RESULTS
    # -----------------------------------------------------------------
    if os.path.exists(output_file):
        with open(output_file) as f:
            return json.load(f)

    return {"error": "Benchmark output not found", "summary": {}}

# =====================================================================
# LOCAL ENTRYPOINT
# =====================================================================
@app.local_entrypoint()
def main(mode: str = "mojo-benchmark"):
    print(f"Running in {mode} mode...")

    if mode == "mojo-benchmark":
        results = benchmark_mojo_vs_cuda.remote()

        print("\n======= BENCHMARK COMPLETE =======")
        if "error" in results:
            print("❌ Error:", results["error"])
            return

        # Summary printout
        summary = results.get("summary", {})
        print("\n=== MOJO VS CUDA SUMMARY ===")
        print(f"Total Mojo Time:    {summary.get('total_mojo_time_ms', 0):.3f} ms")
        print(f"Total PyTorch Time: {summary.get('total_pytorch_time_ms', 0):.3f} ms")
        print(f"Overall Speedup:    {summary.get('overall_speedup', 0):.2f}x")

        print("\n=== DETAILS ===")
        for bench in results.get("benchmarks", []):
            print(f"{bench['operation']:12s}: {bench['speedup']:.2f}x ({bench['faster']} faster)")
