#!/bin/bash
# Setup script for AIMO3 Recipe environment
# Usage: ./setup_env.sh [env_name]

set -e

ENV_NAME="${1:-aimo3}"
PYTHON_VERSION="3.11"

echo "=== AIMO3 Recipe Environment Setup ==="
echo "Environment name: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment if it doesn't exist
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists. Activating..."
else
    echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Verify we're in the right environment
echo "Active Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Install the package in editable mode
echo "Installing aimo3-recipe package..."
pip install -e .

# Install tinker-cookbook from GitHub
echo "Installing tinker-cookbook from GitHub..."
pip install git+https://github.com/thinking-machines-lab/tinker-cookbook.git

# Install dev dependencies
echo "Installing development dependencies..."
pip install -e ".[dev]"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run Tinker training, set your API key:"
echo "  export TINKER_API_KEY=your_key_here"
echo ""
echo "Or create a .env file:"
echo "  echo 'TINKER_API_KEY=your_key_here' > .env"
echo ""
echo "Then run training:"
echo "  source .env && python aimo3_recipe/training/tinker_rl_math.py max_samples=10"
