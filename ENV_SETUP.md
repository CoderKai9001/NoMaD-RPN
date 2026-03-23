# Environment Setup Requirements

This document tracks the complete setup instructions for the `nomad_train` conda environment, combining the base General Navigation Models (GNM/NoMaD) setup with our headless Habitat Simulator integration for automated exploration and testing.

## 1. Base NoMaD Environment Setup
The base environment provides the necessary dependencies for training and inference with PyTorch, diffusion policies, and ROS integrations.

```bash
# 1. Create the conda environment
conda env create -f train/train_environment.yml

# 2. Activate the environment (depending on your yml, it may be named vint_train or nomad_train)
conda activate nomad_train

# 3. Install the local train/ modeling package
pip install -e train/

# 4. Clone and install the Stanford diffusion_policy package
git clone git@github.com:real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
```

## 2. Habitat Simulator Setup (Headless Execution)
To run the automated `explore_and_navigate.py` scripts on cluster setups or machines without displays, we bypass the complete `habitat-lab` suite and install a strict headless configuration of `habitat-sim` (version 0.2.3). 

Using the `conda` packages is the most reliable way to enforce correct C++ backend binaries:

```bash
# Optional but recommended: set conda solver to mamba for faster environment solving
conda config --set solver libmamba

# Install the headless habitat-sim binaries (v0.2.3) directly without upgrading base dependencies
conda install -c aihabitat -c conda-forge habitat-sim-headless=0.2.3 --no-deps -y
```

*(Note: If you ever need to build `habitat-sim` directly from its source repository, make sure your CMake version is strictly `<4.0` due backward compatibility hooks: `pip install "cmake<4.0"`)*

## 3. Additional Integration Dependencies
The integration scripts require additional libraries for handling quaternion rotations (extracted from SLAM/Odometry data) and creating `.mp4` video renderings of the agent's point of view:

```bash
# Install OpenCV (for cv2.VideoWriter) and Numpy Quaternion (for positional data alignments)
pip install opencv-python numpy-quaternion
```

## 4. Verification Check
To verify everything is properly connected, run the following Python test inside your terminal:

```bash
python -c "import torch; import habitat_sim; import quaternion; import cv2; print('Environment setup successfully!')"
```