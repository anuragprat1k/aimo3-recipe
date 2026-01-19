#!/bin/bash
# Run the complete AIMO3 training pipeline

set -e

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-14B}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

echo "=============================================="
echo "AIMO3 Training Pipeline"
echo "Base model: $BASE_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

# Stage 1: Chain-of-Thought SFT
echo ""
echo "[Stage 1/3] Chain-of-Thought SFT"
echo "=============================================="
python -m aimo3_recipe.training.train --stage cot \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES}

# Stage 2: Tool-Integrated Reasoning SFT
echo ""
echo "[Stage 2/3] Tool-Integrated Reasoning SFT"
echo "=============================================="
python -m aimo3_recipe.training.train --stage tir \
    --output-dir "$OUTPUT_DIR" \
    ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES}

# Stage 3: RL with correctness rewards
echo ""
echo "[Stage 3/3] RL with Correctness Rewards"
echo "=============================================="
python -m aimo3_recipe.training.train --stage rl \
    --output-dir "$OUTPUT_DIR" \
    ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES}

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "Final model: $OUTPUT_DIR/rl_math"
echo "=============================================="
