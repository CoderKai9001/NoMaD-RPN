#!/bin/bash
# Fast Habitat-Lab Setup (without full habitat-sim for now)
# This installs the core components needed to test the integration

set -e

echo "=========================================="
echo "Fast Habitat Setup for NoMaD Integration"
echo "=========================================="
echo ""

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nomad_train

# Install habitat-lab (pure Python, fast)
echo "Installing habitat-lab..."
pip install git+https://github.com/facebookresearch/habitat-lab.git@v0.2.3 --quiet

# Fix dependencies
echo "Fixing dependency compatibility..."
pip install "huggingface_hub<0.19.0" --quiet

# Test imports
echo ""
echo "Testing installations..."
python -c "import habitat; print(f'✓ habitat-lab installed: {habitat.__version__}')" || echo "✗ habitat-lab import failed"

echo ""
echo "=========================================="
echo "Partial Installation Complete"
echo "=========================================="
echo ""
echo "habitat-lab is installed (navigation algorithms, episode management)"
echo ""
echo "To run full simulations, you also need habitat-sim:"
echo "  Option 1: Install pre-built wheels (fastest):"
echo "    conda install -c aihabitat habitat-sim-headless"
echo ""
echo "  Option 2: Install from source (if pre-built unavailable):"
echo "    See: https://github.com/facebookresearch/habitat-sim#installation"
echo ""
echo "For now, you can test the core integration without full simulation."
echo ""
