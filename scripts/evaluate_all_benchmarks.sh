#!/bin/bash
# Evaluate model on all math benchmarks

set -e

MODEL_PATH="${1:-./outputs/rl_math}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_results}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"

echo "=============================================="
echo "AIMO3 Model Evaluation"
echo "Model: $MODEL_PATH"
echo "Samples per problem: $NUM_SAMPLES"
echo "=============================================="

BENCHMARKS=("math" "math_level5" "amc" "aime" "gsm8k")

for benchmark in "${BENCHMARKS[@]}"; do
    echo ""
    echo "Evaluating on: $benchmark"
    echo "----------------------------------------------"

    python -m aimo3_recipe.evaluation.evaluate \
        --model "$MODEL_PATH" \
        --benchmark "$benchmark" \
        --num-samples "$NUM_SAMPLES" \
        --output-dir "$OUTPUT_DIR" \
        --use-vllm
done

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
