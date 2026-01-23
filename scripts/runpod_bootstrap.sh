#!/bin/bash
# Bootstrap script for RunPod local RL + vLLM training
set -euo pipefail

echo "=== RunPod Bootstrap (aimo3-recipe) ==="

PIP_TARGET="/workspace/pip-packages"

# Persist installs across container restarts
mkdir -p "${PIP_TARGET}"
export PIP_TARGET="${PIP_TARGET}"
export PYTHONPATH="${PIP_TARGET}:${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# Make env variables persistent for future shells
grep -q "PIP_TARGET=/workspace/pip-packages" ~/.bashrc || \
  echo 'export PIP_TARGET=/workspace/pip-packages' >> ~/.bashrc
grep -q "PYTHONPATH=/workspace/pip-packages" ~/.bashrc || \
  echo 'export PYTHONPATH=/workspace/pip-packages:$PYTHONPATH' >> ~/.bashrc
grep -q "HF_HUB_ENABLE_HF_TRANSFER=1" ~/.bashrc || \
  echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> ~/.bashrc

# Capture existing torch version to avoid overwriting CUDA-enabled build
TORCH_VERSION=$(python3 - <<'PY'
try:
    import torch
    print(torch.__version__)
except Exception:
    print("none")
PY
)
CUDA_VERSION=$(python3 - <<'PY'
try:
    import torch
    print(torch.version.cuda or "none")
except Exception:
    print("none")
PY
)
echo "Existing PyTorch: ${TORCH_VERSION} (CUDA ${CUDA_VERSION})"

echo "Installing core dependencies into ${PIP_TARGET}..."
python3 -m pip install \
  transformers datasets numpy sympy wandb hf_transfer tensorboard psutil \
  --target "${PIP_TARGET}"

echo "Installing torch-dependent packages (no-deps) into ${PIP_TARGET}..."
python3 -m pip install \
  accelerate peft trl \
  --target "${PIP_TARGET}" --no-deps

# TRL-compatible vLLM version (matches repo setup)
echo "Installing vLLM 0.12.0 (no-deps) into ${PIP_TARGET}..."
python3 -m pip install vllm==0.12.0 --target "${PIP_TARGET}" --no-deps

echo "Installing vLLM runtime deps (excluding torch)..."
python3 -m pip install \
  msgspec gguf mistral_common partial_json_parser pillow compressed-tensors \
  --target "${PIP_TARGET}" 2>/dev/null || true

# Verify torch wasn't overwritten
NEW_TORCH_VERSION=$(python3 - <<'PY'
try:
    import torch
    print(torch.__version__)
except Exception:
    print("none")
PY
)
if [ "${TORCH_VERSION}" != "${NEW_TORCH_VERSION}" ]; then
  echo "ERROR: PyTorch changed from ${TORCH_VERSION} to ${NEW_TORCH_VERSION}"
  echo "A CUDA-enabled torch may have been overwritten."
  exit 1
fi

echo "=== Verifying Installation ==="

python3 - <<'PY'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

python3 - <<'PY'
try:
    import vllm
    print(f"vLLM version: {vllm.__version__}")
except Exception as exc:
    print(f"vLLM import failed: {exc}")
    raise
PY

echo ""
echo "=== Bootstrap Complete ==="
echo "Next steps:"
echo "  export WANDB_API_KEY=your_key_here  # optional"
echo "  python -m aimo3_recipe.training.train --stage rl \\"
echo "    --base-model Qwen/Qwen3-1.7B --output-dir ./outputs \\"
echo "    --save-samples --eval-steps 15 --eval-samples 50 --num-generations 16"
