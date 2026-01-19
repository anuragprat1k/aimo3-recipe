# AIMO3 Recipe

Post-training recipes for LLMs on MATH datasets, targeting the [AI Mathematical Olympiad Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3) competition.

Built on [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) patterns and incorporating techniques from winning AIMO solutions:
- **Project Numina** (AIMO1 winner): Chain-of-Thought + Tool-Integrated Reasoning
- **NemoSkills** (AIMO2 winner): Long-reasoning + GenSelect solution selection

## Overview

This repository provides a complete training pipeline for mathematical reasoning:

```
Stage 1: SFT-CoT     →  Stage 2: SFT-TIR    →  Stage 3: RL
(Chain-of-Thought)      (Tool-Integrated)       (Correctness Rewards)
```

### Key Features

- **Multi-stage training**: CoT → TIR → RL pipeline proven effective in AIMO competitions
- **Tinker SDK integration**: Offload training to remote GPU clusters
- **Multiple inference strategies**: Greedy, majority voting, TIR, GenSelect
- **Comprehensive evaluation**: MATH, AMC, AIME, Olympiad benchmarks

## Installation

### Quick Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/aimo3-recipe.git
cd aimo3-recipe

# Run the setup script (creates conda env and installs all dependencies)
./setup_env.sh

# Activate the environment
conda activate aimo3
```

### Manual Installation

```bash
# Create a Python environment
conda create -n aimo3 python=3.11
conda activate aimo3

# Install the package
pip install -e .

# Install tinker-cookbook from GitHub (required for Tinker training)
pip install git+https://github.com/thinking-machines-lab/tinker-cookbook.git

# For development (includes pytest, ruff, mypy)
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- CUDA 12.0+ (for local training)
- Tinker API key (for remote training) - get one at https://tinker-console.thinkingmachines.ai/

## Quick Start

### 1. Local Training

Run the full 3-stage pipeline locally:

```bash
# Full pipeline (CoT → TIR → RL)
python -m aimo3_recipe.training.train --stage full \
    --base-model Qwen/Qwen2.5-14B \
    --output-dir ./outputs

# Individual stages
python -m aimo3_recipe.training.train --stage cot --base-model Qwen/Qwen2.5-14B
python -m aimo3_recipe.training.train --stage tir
python -m aimo3_recipe.training.train --stage rl
```

### 2. Remote Training with Tinker

Use Tinker SDK to train on remote GPU clusters:

```bash
# Set your Tinker API key (or add to .env file)
export TINKER_API_KEY=your_key_here

# Run SFT training
python aimo3_recipe/training/tinker_sft_math.py \
    model_name=Qwen/Qwen2.5-14B \
    stage=cot \
    log_dir=./outputs/tinker_cot

# Run RL training
python aimo3_recipe/training/tinker_rl_math.py \
    model_name=Qwen/Qwen3-8B \
    log_dir=./outputs/tinker_rl

# Run with smaller test configuration
python aimo3_recipe/training/tinker_rl_math.py \
    model_name=Qwen/Qwen3-8B \
    max_samples=10 \
    batch_size=2 \
    group_size=4
```

Note: Arguments use `key=value` format (no dashes) as this uses the `chz` configuration library.

### 3. Evaluation

```bash
# Evaluate on MATH benchmark
python -m aimo3_recipe.evaluation.evaluate \
    --model ./outputs/rl_math \
    --benchmark math \
    --use-vllm

# With self-consistency (majority voting)
python -m aimo3_recipe.evaluation.evaluate \
    --model ./outputs/rl_math \
    --benchmark aime \
    --num-samples 16
```

### 4. Inference

```bash
# Solve a single problem
python -m aimo3_recipe.inference.generate \
    --model ./outputs/rl_math \
    --strategy tir \
    --problem "Find all positive integers n such that n^2 + 1 divides n^3 + 1."
```

## Training Pipeline

### Stage 1: Chain-of-Thought SFT

Trains the model to produce step-by-step reasoning for math problems.

**Dataset**: [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) (~860k problems)

**Key parameters**:
- Learning rate: 2e-5
- LoRA rank: 64
- Max sequence length: 4096
- Epochs: 3

### Stage 2: Tool-Integrated Reasoning SFT

Teaches the model to generate and use Python code for computation.

**Dataset**: [NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) (~70k problems)

**Format**:
```
Let me solve this step by step.

First, I'll set up the equation...

```python
from sympy import symbols, solve
x = symbols('x')
solutions = solve(x**2 - 5*x + 6, x)
print(solutions)
```
```output
[2, 3]
```

The solutions are x = 2 and x = 3.

Therefore, the answer is \boxed{2, 3}.
```

### Stage 3: Reinforcement Learning

Fine-tunes using correctness rewards (GRPO-style):

**Reward structure**:
- Correct answer: +1.0
- Incorrect answer: -0.5
- Proper \boxed{} format: +0.1

**Key parameters**:
- Learning rate: 1e-6
- Group size: 16 rollouts per problem
- Temperature: 0.7

## Inference Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `greedy` | Single-shot generation | Fast inference |
| `majority_vote` | Sample N solutions, vote on answer | Higher accuracy |
| `weighted_vote` | Weight votes by model confidence | Better than majority |
| `tir` | Tool-Integrated Reasoning | Complex calculations |
| `genselect` | Train model to select best solution | Highest accuracy |

## Project Structure

```
aimo3-recipe/
├── aimo3_recipe/
│   ├── data/
│   │   ├── datasets.py          # Dataset loaders
│   │   ├── preprocessing.py     # Data cleaning
│   │   └── renderers.py         # Chat format renderers
│   ├── training/
│   │   ├── sft_cot.py           # Local CoT training
│   │   ├── sft_tir.py           # Local TIR training
│   │   ├── rl_math.py           # Local RL training
│   │   ├── tinker_sft_math.py   # Tinker SFT training
│   │   ├── tinker_rl_math.py    # Tinker RL training
│   │   └── train.py             # Main entry point
│   ├── evaluation/
│   │   ├── answer_extraction.py # Answer parsing
│   │   ├── benchmarks.py        # Benchmark definitions
│   │   └── evaluate.py          # Evaluation pipeline
│   ├── inference/
│   │   ├── tir_executor.py      # Code execution
│   │   ├── solution_selection.py # Voting/GenSelect
│   │   └── generate.py          # Generation interface
│   └── utils/
│       ├── math_utils.py        # Math utilities
│       └── code_execution.py    # Safe code execution
├── configs/
│   ├── train_qwen_14b.yaml      # Training config
│   ├── tinker_train.yaml        # Tinker config
│   └── eval.yaml                # Evaluation config
└── scripts/                     # Utility scripts
```

## Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | ~860k | CoT solutions from multiple sources |
| [NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) | ~70k | Tool-integrated reasoning solutions |
| [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | 540k | Long-reasoning solutions (NemoSkills) |
| [MATH](https://huggingface.co/datasets/lighteval/MATH) | 12.5k | Hendrycks MATH benchmark |

## Results

Expected performance on benchmarks (Qwen2.5-14B after full pipeline):

| Benchmark | Greedy | SC@16 | TIR | Notes |
|-----------|--------|-------|-----|-------|
| MATH | ~55% | ~65% | ~68% | Full benchmark |
| MATH Level 5 | ~35% | ~45% | ~50% | Hardest problems |
| AMC | ~60% | ~70% | ~75% | Competition level |
| AIME | ~25% | ~35% | ~40% | Harder competition |

## AIMO3 Competition Notes

For the AIMO3 competition specifically:

1. **Compute constraints**: 5 hours on 4x L4 GPUs
2. **Model size**: Consider 14B or smaller for inference efficiency
3. **Quantization**: 8-bit quantization recommended for deployment
4. **Solution selection**: GenSelect or weighted voting with TIR

Recommended inference setup:
```python
from aimo3_recipe.inference.generate import MathGenerator, GenerationConfig

config = GenerationConfig(
    model_name_or_path="./outputs/rl_math",
    strategy="tir",
    num_samples=16,
    use_vllm=True,
)
generator = MathGenerator(config)
solution = generator.solve(problem)
```

## References

- [AIMO Prize](https://aimoprize.com/)
- [Project Numina Technical Report](https://huggingface.co/blog/winning-aimo-progress-prize)
- [NemoSkills AIMO2 Paper](https://arxiv.org/abs/2504.16891)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)

## License

Apache-2.0
