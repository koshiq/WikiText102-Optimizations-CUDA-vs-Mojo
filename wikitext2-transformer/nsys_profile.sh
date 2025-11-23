#!/bin/bash
# NVIDIA Nsight Systems profiling script for PyTorch Transformer Model
# Ensures proper CUDA tracing and file format

set -e

echo "=================================="
echo "Nsight Systems GPU Profiling"
echo "=================================="
echo ""

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "Error: nsys command not found"
    echo "Please install NVIDIA Nsight Systems or add it to PATH"
    echo "Example: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')"

echo ""
echo "Profiling options:"
echo "1) Quick profile (10 batches, ~30 seconds)"
echo "2) Standard profile (50 batches, ~2 minutes)"
echo "3) Comprehensive profile (100 batches, ~5 minutes)"
echo "4) Custom"
echo ""
read -p "Select option [1-4]: " choice

case $choice in
    1)
        BATCHES=10
        OUTPUT="quick_profile"
        ;;
    2)
        BATCHES=50
        OUTPUT="standard_profile"
        ;;
    3)
        BATCHES=100
        OUTPUT="comprehensive_profile"
        ;;
    4)
        read -p "Number of batches: " BATCHES
        read -p "Output filename (without extension): " OUTPUT
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

OUTPUT_FILE="${OUTPUT}.nsys-rep"

echo ""
echo "Starting profiling with:"
echo "  - Batches: $BATCHES"
echo "  - Output: $OUTPUT_FILE"
echo ""

# Run nsys profile with proper settings
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --cuda-graph-trace=node \
    --python-sampling=true \
    --output="$OUTPUT_FILE" \
    python profile_model_gpu.py --profile-batches="$BATCHES"

echo ""
echo "=================================="
echo "Profiling Complete!"
echo "=================================="
echo ""
echo "Generated file: $OUTPUT_FILE"
echo ""
echo "View results:"
echo "  1. Command line stats:  nsys stats $OUTPUT_FILE"
echo "  2. GUI:                 nsys-ui $OUTPUT_FILE"
echo "  3. Export to SQLite:    nsys export --type=sqlite --output=${OUTPUT}.sqlite $OUTPUT_FILE"
echo ""

# Automatically generate stats
echo "Generating summary statistics..."
echo ""
nsys stats "$OUTPUT_FILE" 2>/dev/null || echo "Note: nsys stats failed - you may need to use nsys-ui for GUI analysis"

echo ""
echo "Done!"
