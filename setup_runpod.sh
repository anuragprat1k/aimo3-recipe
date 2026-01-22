#!/bin/bash
# Setup script for RunPod environment
set -e

echo "=== RunPod Setup Script ==="

# Install Python requirements
echo "Installing Python requirements..."
pip install -r /workspace/aimo3-recipe/requirements-runpod.txt

# Install Flash Attention 2 (requires special build flags)
echo "Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation

# Install GitHub CLI (gh)
echo "Installing GitHub CLI..."
(type -p wget >/dev/null || (apt-get update && apt-get install wget -y)) \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install gh -y

# Install tmux
echo "Installing tmux..."
apt-get install tmux -y

# Install Claude Code
echo "Installing Claude Code..."
curl -fsSL https://claude.ai/install.sh | bash

# Add local bin to PATH
echo "Updating PATH..."
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

echo ""
echo "=== Verifying Installation ==="

# Check PyTorch and CUDA
echo "Checking PyTorch and CUDA..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    # Quick tensor test
    x = torch.randn(100, 100, device='cuda')
    y = torch.matmul(x, x)
    print('CUDA tensor test: PASSED')
else:
    print('WARNING: CUDA not available!')
    exit(1)
"

# Check Flash Attention
echo "Checking Flash Attention..."
python3 -c "
try:
    import flash_attn
    print(f'Flash Attention version: {flash_attn.__version__}')
    print('Flash Attention: PASSED')
except ImportError as e:
    print(f'Flash Attention: NOT INSTALLED ({e})')
"

# Check vLLM
echo "Checking vLLM..."
python3 -c "
try:
    import vllm
    print(f'vLLM version: {vllm.__version__}')
    print('vLLM: PASSED')
except ImportError as e:
    print(f'vLLM: NOT INSTALLED ({e})')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Authenticate with GitHub: gh auth login"
echo "  2. Authenticate with Claude: claude config set apiKey YOUR_API_KEY"
echo "  3. Start training: python -m aimo3_recipe.training.train --stage rl --base-model Qwen/Qwen3-1.7B --output-dir ./outputs --save-samples --eval-steps 15 --eval-samples 50 --num-generations 16"
