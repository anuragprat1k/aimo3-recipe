#!/bin/bash
# Run the AIMO3 training pipeline using Tinker remote training

set -e

# Check for Tinker API key
if [ -z "$TINKER_API_KEY" ]; then
    echo "Error: TINKER_API_KEY environment variable not set"
    echo "Get your API key from https://tinker-docs.thinkingmachines.ai"
    exit 1
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-14B}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

echo "=============================================="
echo "AIMO3 Tinker Training Pipeline"
echo "Model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

# Stage 1: CoT SFT
echo ""
echo "[Stage 1/3] Chain-of-Thought SFT (Tinker)"
echo "=============================================="
python -m aimo3_recipe.training.tinker_sft_math \
    --model_name "$MODEL_NAME" \
    --stage cot \
    --log_dir "$OUTPUT_DIR/tinker_cot" \
    ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}

# Stage 2: TIR SFT
echo ""
echo "[Stage 2/3] Tool-Integrated Reasoning SFT (Tinker)"
echo "=============================================="
python -m aimo3_recipe.training.tinker_sft_math \
    --model_name "$MODEL_NAME" \
    --stage tir \
    --log_dir "$OUTPUT_DIR/tinker_tir" \
    --resume_from "$OUTPUT_DIR/tinker_cot/final" \
    ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}

# Stage 3: RL
echo ""
echo "[Stage 3/3] RL Training (Tinker)"
echo "=============================================="
python -m aimo3_recipe.training.tinker_rl_math \
    --model_name "$MODEL_NAME" \
    --log_dir "$OUTPUT_DIR/tinker_rl" \
    --resume_from "$OUTPUT_DIR/tinker_tir/final" \
    ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "Final model: $OUTPUT_DIR/tinker_rl/final"
echo "=============================================="
