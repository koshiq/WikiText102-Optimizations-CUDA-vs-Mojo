#!/bin/bash
# Setup script for Mojo kernel integration

set -e

echo "=================================================="
echo "Setting up Mojo Kernel Integration"
echo "=================================================="
echo ""

# Check if MODULAR_API_KEY is set
if [ -z "$MODULAR_API_KEY" ]; then
    echo "Error: MODULAR_API_KEY environment variable not set"
    echo ""
    echo "Please get your API key from: https://www.modular.com/max"
    echo "Then run: export MODULAR_API_KEY='your-api-key'"
    exit 1
fi

# Install Modular CLI if not present
if ! command -v modular &> /dev/null; then
    echo "Installing Modular CLI..."
    curl https://get.modular.com | sh -
fi

# Authenticate
echo "Authenticating with Modular..."
modular auth $MODULAR_API_KEY

# Install MAX Engine
echo "Installing MAX Engine..."
modular install max

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements_mojo.txt

# Create mojo_kernels directory if it doesn't exist
echo "Creating Mojo kernels directory..."
mkdir -p mojo_kernels

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Package your Mojo kernels into mojo_kernels/"
echo "2. Run benchmarks: python benchmark_mojo_vs_cuda.py"
echo "3. Train model: python main.py --use-mojo --accel"
echo ""
echo "For Modal deployment:"
echo "  modal secret create modular-api-key MODULAR_API_KEY=\$MODULAR_API_KEY"
echo "  modal run modal_train.py --mode=train --use-mojo=True"
