"""
Main training entry point for AIMO3 recipe.

Orchestrates the multi-stage training pipeline:
1. Stage 1: SFT with Chain-of-Thought
2. Stage 2: SFT with Tool-Integrated Reasoning
3. Stage 3: RL with correctness rewards
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset, Dataset
from aimo3_recipe.training.sft_cot import SFTCoTTrainer, SFTCoTConfig
from aimo3_recipe.training.sft_tir import SFTTIRTrainer, SFTTIRConfig
from aimo3_recipe.training.rl_math import RLMathTrainer, RLMathConfig, MathRewardFunction
from aimo3_recipe.evaluation.answer_extraction import verify_answer


def load_math_datasets(stage: str, max_samples: int | None = None) -> tuple[Dataset, Dataset]:
    """Load appropriate datasets for each training stage."""

    if stage == "cot":
        # NuminaMath-CoT for Chain-of-Thought training
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        # Split into train/eval
        split = dataset.train_test_split(test_size=0.01, seed=42)
        return split["train"], split["test"]

    elif stage == "tir":
        # NuminaMath-TIR for Tool-Integrated Reasoning
        dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train")
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        split = dataset.train_test_split(test_size=0.01, seed=42)
        return split["train"], split["test"]

    elif stage == "rl":
        # Use MATH dataset for RL (has cleaner answer format)
        try:
            dataset = load_dataset("lighteval/MATH-Hard", split="train")
            dataset = dataset.rename_columns({"problem": "problem", "solution": "answer"})
        except Exception as exc:
            print(f"Failed to load lighteval/MATH ({exc}); falling back to NuminaMath-CoT.")
            dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
            dataset = dataset.rename_columns({"solution": "answer"})
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        split = dataset.train_test_split(test_size=0.01, seed=42)
        return split["train"], split["test"]

    else:
        raise ValueError(f"Unknown stage: {stage}")


def run_stage1_cot(
    base_model: str = "Qwen/Qwen2.5-14B",
    output_dir: str = "./outputs/sft_cot",
    max_samples: int | None = None,
    **kwargs,
) -> None:
    """Run Stage 1: Chain-of-Thought SFT."""

    print("=" * 60)
    print("Stage 1: Chain-of-Thought Supervised Fine-Tuning")
    print("=" * 60)

    if "report_to" not in kwargs:
        if os.getenv("WANDB_DISABLED") or not os.getenv("WANDB_API_KEY"):
            kwargs["report_to"] = "tensorboard"

    config = SFTCoTConfig(
        model_name_or_path=base_model,
        output_dir=output_dir,
        **kwargs,
    )

    train_dataset, eval_dataset = load_math_datasets("cot", max_samples)
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    trainer = SFTCoTTrainer(config)
    trainer.train(train_dataset, eval_dataset)

    # Merge and save
    merged_path = f"{output_dir}_merged"
    trainer.save_merged_model(merged_path)
    print(f"Merged model saved to: {merged_path}")


def run_stage2_tir(
    base_model: str = "./outputs/sft_cot_merged",
    output_dir: str = "./outputs/sft_tir",
    max_samples: int | None = None,
    **kwargs,
) -> None:
    """Run Stage 2: Tool-Integrated Reasoning SFT."""

    print("=" * 60)
    print("Stage 2: Tool-Integrated Reasoning SFT")
    print("=" * 60)

    if "report_to" not in kwargs:
        if os.getenv("WANDB_DISABLED") or not os.getenv("WANDB_API_KEY"):
            kwargs["report_to"] = "tensorboard"

    config = SFTTIRConfig(
        model_name_or_path=base_model,
        output_dir=output_dir,
        **kwargs,
    )

    train_dataset, eval_dataset = load_math_datasets("tir", max_samples)
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    trainer = SFTTIRTrainer(config)
    trainer.train(train_dataset, eval_dataset)


def run_stage3_rl(
    base_model: str = "./outputs/sft_tir",
    output_dir: str = "./outputs/rl_math",
    max_samples: int | None = None,
    **kwargs,
) -> None:
    """Run Stage 3: Reinforcement Learning with correctness rewards."""

    print("=" * 60)
    print("Stage 3: RL with Correctness Rewards")
    print("=" * 60)

    if "report_to" not in kwargs:
        if os.getenv("WANDB_DISABLED") or not os.getenv("WANDB_API_KEY"):
            kwargs["report_to"] = "tensorboard"

    config = RLMathConfig(
        model_name_or_path=base_model,
        output_dir=output_dir,
        **kwargs,
    )

    # Reward function using answer verification
    reward_fn = MathRewardFunction(
        answer_verifier=verify_answer,
        correct_reward=config.correct_reward,
        incorrect_reward=config.incorrect_reward,
        format_reward=config.format_reward,
    )

    train_dataset, _ = load_math_datasets("rl", max_samples)
    print(f"Train samples: {len(train_dataset)}")

    trainer = RLMathTrainer(config, reward_fn)
    trainer.train(train_dataset)


def run_full_pipeline(
    base_model: str = "Qwen/Qwen2.5-14B",
    output_base: str = "./outputs",
    max_samples_per_stage: int | None = None,
) -> None:
    """Run the complete 3-stage training pipeline."""

    print("Starting full AIMO3 training pipeline")
    print(f"Base model: {base_model}")
    print(f"Output directory: {output_base}")

    # Stage 1
    run_stage1_cot(
        base_model=base_model,
        output_dir=f"{output_base}/sft_cot",
        max_samples=max_samples_per_stage,
    )

    # Stage 2
    run_stage2_tir(
        base_model=f"{output_base}/sft_cot_merged",
        output_dir=f"{output_base}/sft_tir",
        max_samples=max_samples_per_stage,
    )

    # Stage 3
    run_stage3_rl(
        base_model=f"{output_base}/sft_tir",
        output_dir=f"{output_base}/rl_math",
        max_samples=max_samples_per_stage,
    )

    print("=" * 60)
    print("Training pipeline complete!")
    print(f"Final model: {output_base}/rl_math")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AIMO3 Training Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["cot", "tir", "rl", "full"],
        default="full",
        help="Training stage to run",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-14B",
        help="Base model for training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per stage (for testing)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save a sample of RL rollouts to samples.jsonl",
    )
    parser.add_argument(
        "--sample-save-rate",
        type=float,
        default=None,
        help="Fraction of RL samples to save (e.g., 0.01)",
    )
    parser.add_argument(
        "--samples-filename",
        type=str,
        default=None,
        help="Filename for saved RL samples (relative to output dir)",
    )

    args = parser.parse_args()

    kwargs = {}
    if args.learning_rate:
        kwargs["learning_rate"] = args.learning_rate
    if args.save_samples:
        kwargs["save_samples"] = True
    if args.sample_save_rate is not None:
        kwargs["sample_save_rate"] = args.sample_save_rate
    if args.samples_filename:
        kwargs["samples_filename"] = args.samples_filename

    if args.stage == "full":
        run_full_pipeline(
            base_model=args.base_model,
            output_base=args.output_dir,
            max_samples_per_stage=args.max_samples,
        )
    elif args.stage == "cot":
        run_stage1_cot(
            base_model=args.base_model,
            output_dir=f"{args.output_dir}/sft_cot",
            max_samples=args.max_samples,
            **kwargs,
        )
    elif args.stage == "tir":
        run_stage2_tir(
            output_dir=f"{args.output_dir}/sft_tir",
            max_samples=args.max_samples,
            **kwargs,
        )
    elif args.stage == "rl":
        run_stage3_rl(
            output_dir=f"{args.output_dir}/rl_math",
            max_samples=args.max_samples,
            **kwargs,
        )


if __name__ == "__main__":
    main()
