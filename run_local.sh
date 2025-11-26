#!/bin/bash
# Quick start script for running training, benchmarking, and profiling locally

set -e  # Exit on error

cd "$(dirname "$0")/wikitext2-transformer"

echo "=================================="
echo "WikiText-2 Transformer Quick Start"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Choose what to run:"
echo "1) Quick training (5 epochs, dry run)"
echo "2) Full training (40 epochs)"
echo "3) Benchmark (5 epochs, 3 runs)"
echo "4) Profile model"
echo "5) All (quick train + benchmark + profile)"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "=== Running Quick Training (5 epochs) ==="
        python main.py \
            --data ./data/wikitext-2 \
            --epochs 5 \
            --batch_size 20 \
            --emsize 200 \
            --nhid 200 \
            --nlayers 4 \
            --nhead 2 \
            --lr 20 \
            --dropout 0.2 \
            --save model_quick.pt \
            --use-optimizer \
            --accel
        echo ""
        echo "✓ Model saved to model_quick.pt"
        ;;

    2)
        echo ""
        echo "=== Running Full Training (40 epochs) ==="
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
            --save model_full.pt \
            --use-optimizer \
            --accel
        echo ""
        echo "✓ Model saved to model_full.pt"
        ;;

    3)
        echo ""
        echo "=== Running Benchmark (5 epochs, 3 runs) ==="
        python benchmark.py \
            --data ./data/wikitext-2 \
            --epochs 5 \
            --runs 3 \
            --batch_size 20 \
            --emsize 200 \
            --nhid 200 \
            --nlayers 4 \
            --nhead 2 \
            --dropout 0.2 \
            --output benchmark_results.json \
            --accel
        echo ""
        echo "✓ Benchmark results saved to benchmark_results.json"

        # Display summary if jq is available
        if command -v jq &> /dev/null; then
            echo ""
            echo "=== Benchmark Summary ==="
            cat benchmark_results.json | jq '.summary'
        fi
        ;;

    4)
        echo ""
        echo "=== Running Profiler (100 batches) ==="
        python profile_model.py \
            --data ./data/wikitext-2 \
            --profile-batches 100 \
            --batch_size 20 \
            --emsize 200 \
            --nhid 200 \
            --nlayers 4 \
            --nhead 2 \
            --dropout 0.2 \
            --accel
        echo ""
        echo "✓ Profiling complete!"
        echo "  - trace_training.json (view in chrome://tracing)"
        echo "  - trace_inference.json (view in chrome://tracing)"
        echo "  - profiler_stacks.txt"
        ;;

    5)
        echo ""
        echo "=== Running All (Quick Train + Benchmark + Profile) ==="

        echo ""
        echo "[1/3] Quick Training..."
        python main.py \
            --data ./data/wikitext-2 \
            --epochs 5 \
            --batch_size 20 \
            --lr 0.001 \
            --save model_quick.pt \
            --use-optimizer \
            --accel
        echo "✓ Training complete"

        echo ""
        echo "[2/3] Benchmarking..."
        python benchmark.py \
            --data ./data/wikitext-2 \
            --epochs 5 \
            --runs 3 \
            --batch_size 20 \
            --output benchmark_results.json \
            --accel
        echo "✓ Benchmark complete"

        echo ""
        echo "[3/3] Profiling..."
        python profile_model.py \
            --data ./data/wikitext-2 \
            --profile-batches 100 \
            --batch_size 20 \
            --accel
        echo "✓ Profiling complete"

        echo ""
        echo "=================================="
        echo "All tasks completed successfully!"
        echo "=================================="
        echo ""
        echo "Generated files:"
        echo "  - model_quick.pt"
        echo "  - benchmark_results.json"
        echo "  - trace_training.json"
        echo "  - trace_inference.json"
        echo "  - profiler_stacks.txt"
        ;;

    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Done!"
