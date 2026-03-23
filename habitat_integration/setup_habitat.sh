#!/bin/bash
# Setup script for Habitat integration with NoMaD-RPN
# Run this to install Habitat and download HM3D dataset

set -e  # Exit on error

echo "========================================"
echo "NoMaD-RPN Habitat Integration Setup"
echo "========================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Activate nomad_train environment
echo "Activating nomad_train environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nomad_train

# Install habitat-sim via conda
echo ""
echo "Step 1: Installing habitat-sim (this may take 5-10 minutes)..."
conda install -c conda-forge -c aihabitat habitat-sim=0.2.3 headless -y

# Install habitat-lab via pip
echo ""
echo "Step 2: Installing habitat-lab..."
pip install habitat-lab==0.2.3

# Fix huggingface_hub compatibility
echo ""
echo "Step 3: Fixing dependency compatibility..."
pip install "huggingface_hub<0.19.0"

# Verify installation
echo ""
echo "Step 4: Verifying installation..."
python -c "import habitat; print(f'✓ Habitat version: {habitat.__version__}')" || echo "✗ Habitat import failed"
python -c "import habitat_sim; print(f'✓ habitat-sim version: {habitat_sim.__version__}')" || echo "✗ habitat-sim import failed"

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Download HM3D dataset (see DATASET_SETUP.md)"
echo "2. Configure NoMaD checkpoint path in config/nomad.yaml"
echo "3. Update HM3D scene paths in config/evaluation.yaml"
echo "4. Run test: python test_integration.py"
echo ""
