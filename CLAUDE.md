# CLAUDE.md - AI Assistant Guide for aimo3-recipe

This document provides essential context for AI assistants working with the aimo3-recipe codebase.

## Project Overview

**aimo3-recipe** is a post-training pipeline for LLMs targeting mathematical reasoning, specifically designed for the [AI Mathematical Olympiad Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3) competition.

The project implements a three-stage training pipeline:
1. **SFT-CoT** (Chain-of-Thought): Train on step-by-step reasoning
2. **SFT-TIR** (Tool-Integrated Reasoning): Train on Python code execution for computation
3. **RL** (Reinforcement Learning): Fine-tune using correctness rewards (GRPO)

Built on [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) patterns and incorporates techniques from AIMO competition winners:
- **Project Numina** (AIMO1): CoT + TIR approach
- **NemoSkills** (AIMO2): Long-reasoning + GenSelect

## Repository Structure

```
aimo3-recipe/
├── aimo3_recipe/           # Main package
│   ├── data/               # Dataset loaders and preprocessing
│   │   ├── datasets.py     # Dataset loading utilities
│   │   ├── preprocessing.py # Data cleaning
│   │   └── renderers.py    # Chat format renderers
│   ├── training/           # Training implementations
│   │   ├── train.py        # Main CLI entry point
│   │   ├── sft_cot.py      # Stage 1: Chain-of-Thought SFT
│   │   ├── sft_tir.py      # Stage 2: Tool-Integrated Reasoning SFT
│   │   ├── rl_math.py      # Stage 3: Local RL with GRPO
│   │   ├── tinker_sft_math.py  # Tinker SDK SFT training
│   │   └── tinker_rl_math.py   # Tinker SDK RL training
│   ├── evaluation/         # Benchmarking and metrics
│   │   ├── evaluate.py     # Main evaluation pipeline
│   │   ├── evaluate_all.py # Multi-benchmark evaluation
│   │   ├── benchmarks.py   # Benchmark definitions
│   │   └── answer_extraction.py # Answer parsing and verification
│   ├── inference/          # Generation and solution selection
│   │   ├── generate.py     # Generation interface
│   │   ├── tir_executor.py # Python code execution
│   │   └── solution_selection.py # Voting and GenSelect
│   └── utils/              # Utility functions
│       ├── math_utils.py   # Math utilities
│       └── code_execution.py # Safe code execution
├── configs/                # YAML configuration files
│   ├── train_qwen_14b.yaml # Full pipeline config
│   ├── tinker_train.yaml   # Tinker SDK config
│   └── eval.yaml           # Evaluation config
├── scripts/                # Utility scripts
│   ├── evaluate_checkpoints.py
│   └── regenerate_summary.py
├── tests/                  # Pytest test suite
├── pyproject.toml          # Package configuration
├── setup_env.sh            # Conda environment setup
└── setup_runpod.sh         # RunPod cluster setup
```

## Key Entry Points

### Training

```bash
# Full pipeline (all 3 stages)
python -m aimo3_recipe.training.train --stage full --base-model Qwen/Qwen2.5-14B

# Individual stages
python -m aimo3_recipe.training.train --stage cot --base-model Qwen/Qwen2.5-14B
python -m aimo3_recipe.training.train --stage tir
python -m aimo3_recipe.training.train --stage rl --base-model Qwen/Qwen3-0.6B

# Tinker SDK (remote GPU clusters)
python aimo3_recipe/training/tinker_rl_math.py model_name=Qwen/Qwen3-8B log_dir=./outputs
```

### Evaluation

```bash
python -m aimo3_recipe.evaluation.evaluate --model ./outputs/rl_math --benchmark math --use-vllm
```

### Inference

```bash
python -m aimo3_recipe.inference.generate --model ./outputs/rl_math --strategy tir --problem "..."
```

## Code Conventions

### Configuration Style

- **Local training**: Uses Python dataclasses (`RLMathConfig`, `SFTCoTConfig`)
- **Tinker training**: Uses `chz` library with `key=value` CLI syntax (no dashes)
- **Evaluation**: Uses argparse with `--flag` style

### Variable Naming (Tinker)

Tinker modules use suffix conventions for tensor dimensions:
- `_P`: Problem dimension (different questions in batch)
- `_G`: Group dimension (multiple rollouts per problem)
- `_D`: Datum dimension (flattened training examples)

### Answer Format

Models should output answers in `\boxed{}` format:
```
The answer is \boxed{42}.
```

Alternative: GSM8K format `#### answer` is also supported for extraction.

### Reward Structure (RL)

```python
correct_reward = 1.0    # Correct answer
incorrect_reward = -0.5 # Incorrect answer
format_reward = 0.1     # Bonus for proper \boxed{} format
```

## Important Modules

### `aimo3_recipe/evaluation/answer_extraction.py`

Core utilities for math answer verification:
- `extract_boxed_answer(text)`: Extract from `\boxed{}`
- `extract_gsm8k_answer(text)`: Extract from `#### answer`
- `normalize_answer(answer)`: Normalize LaTeX for comparison
- `verify_answer(response, ground_truth)`: Full verification pipeline
- `is_equivalent_answer(pred, gt)`: Checks string, numeric, and symbolic equivalence

### `aimo3_recipe/training/rl_math.py`

Local RL training using TRL's GRPO:
- `RLMathConfig`: All training hyperparameters
- `RLMathTrainer`: Main trainer class
- `MathRewardFunction`: Correctness-based rewards
- `LLMJudgeRewardFunction`: Optional process-based rewards
- `EvalCallback`: Periodic evaluation during training

### `aimo3_recipe/training/tinker_rl_math.py`

Tinker SDK RL training for remote GPU clusters:
- `MathRLConfig`: Configuration dataclass
- `train_math_rl()`: Main training loop
- `compute_reward()`: Reward computation
- Uses `chz.Blueprint` for CLI configuration

## Datasets

| Dataset | Size | Use Case |
|---------|------|----------|
| `AI-MO/NuminaMath-CoT` | ~860k | Stage 1 (CoT SFT) |
| `AI-MO/NuminaMath-TIR` | ~70k | Stage 2 (TIR SFT) |
| `lighteval/MATH-Hard` | ~5k | Stage 3 (RL), Evaluation |
| `nvidia/OpenMathReasoning` | 540k | Alternative training data |

## Dependencies

Core dependencies (from `pyproject.toml`):
- `tinker>=0.8.0` - Remote training SDK
- `chz>=0.4.0` - Configuration library
- `transformers>=4.40.0` - HuggingFace models
- `trl>=0.8.0` - RL training (GRPO)
- `peft>=0.10.0` - LoRA adapters
- `vllm` - Fast inference (optional, recommended)
- `sympy>=1.12` - Symbolic math verification

Dev dependencies: `pytest`, `ruff`, `mypy`, `pre-commit`

## Testing

Tests use pytest with mocked ML dependencies (see `tests/conftest.py`):

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_tinker_rl_math.py -v
```

Heavy dependencies (torch, transformers, tinker) are mocked to allow testing utility functions without GPU.

## Linting and Type Checking

```bash
# Ruff (linting)
ruff check .
ruff format .

# Type checking
mypy aimo3_recipe
```

Ruff config: line-length=100, target Python 3.10

## Git Workflow

**Always pull and rebase before creating or updating a PR:**

```bash
# Fetch latest changes from main branch
git fetch origin main

# Rebase your branch onto main
git rebase origin/main

# If there are conflicts, resolve them, then:
git add .
git rebase --continue

# Force push after rebase (only on feature branches)
git push --force-with-lease
```

This ensures your PR has no merge conflicts and includes the latest changes from main.

## Common Tasks

### Adding a New Benchmark

1. Add benchmark definition to `aimo3_recipe/evaluation/benchmarks.py`
2. Ensure dataset follows format: `problem`, `solution` columns
3. Test with `--max-samples` flag first

### Modifying Reward Function

Edit `MathRewardFunction.__call__()` in `rl_math.py` or `compute_reward()` in `tinker_rl_math.py`.

### Adding Answer Extraction Patterns

Extend patterns in `answer_extraction.py`:
- `extract_answer_from_solution()` for new formats
- `normalize_answer()` for new LaTeX commands

### Checkpointing and Resume

RL training supports checkpoint resume:
```bash
# Auto-resume from latest
python -m aimo3_recipe.training.train --stage rl --resume-from-checkpoint

# Resume from specific checkpoint
python -m aimo3_recipe.training.train --stage rl --resume-from-checkpoint ./outputs/checkpoint-500
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `TINKER_API_KEY` | Tinker SDK authentication |
| `WANDB_API_KEY` | Weights & Biases logging |
| `WANDB_PROJECT` | W&B project name |
| `WANDB_DISABLED` | Disable W&B (falls back to tensorboard) |
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | Required for macOS MPS training |

## PyTorch 2.6+ Compatibility

The codebase includes patches for PyTorch 2.6+ checkpoint loading:
- `train.py` patches `torch.load` to default `weights_only=False`
- `rl_math.py` registers numpy globals with `torch.serialization.add_safe_globals()`

## Mistral Tokenizer Fix

For Mistral models, the codebase automatically adds `fix_mistral_regex=True` to tokenizer kwargs to prevent regex warnings during vLLM inference.

## Performance Notes

- **Multi-GPU training**: Use `accelerate launch -m aimo3_recipe.training.train` with `--no-vllm`
- **vLLM colocate mode**: Default for single-GPU, incompatible with distributed training
- **Evaluation parallelization**: Uses multiprocessing for answer verification (batches >= 100)
- **Tokenized dataset caching**: SFT trainers cache tokenized datasets to disk

## Output Artifacts

Training outputs to `./outputs/` by default:
- `checkpoint-{step}/` - Training checkpoints
- `samples.jsonl` - Saved rollouts (if `--save-samples`)
- `metrics.jsonl` - Training metrics
- TensorBoard logs in output directory
