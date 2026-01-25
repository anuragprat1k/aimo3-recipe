"""
Main training entry point for AIMO3 recipe.

Orchestrates the multi-stage training pipeline:
1. Stage 1: SFT with Chain-of-Thought
2. Stage 2: SFT with Tool-Integrated Reasoning
3. Stage 3: RL with correctness rewards
"""

# PyTorch 2.6+ compatibility: monkey-patch torch.load to handle checkpoint loading
# This must happen BEFORE any other imports to ensure it applies to all ranks
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Default to weights_only=False for backward compatibility with checkpoints
    # that contain numpy arrays in RNG state
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import argparse
import os
import re
from pathlib import Path
from datasets import load_dataset, Dataset
from aimo3_recipe.training.sft_cot import SFTCoTTrainer, SFTCoTConfig
from aimo3_recipe.training.sft_tir import SFTTIRTrainer, SFTTIRConfig
from aimo3_recipe.training.rl_math import RLMathTrainer, RLMathConfig, MathRewardFunction, LLMJudgeRewardFunction
from aimo3_recipe.evaluation.answer_extraction import verify_answer


def get_latest_checkpoint(output_dir: str) -> str | None:
    """
    Find the latest checkpoint in the given output directory.

    Checkpoints are expected to be in the format 'checkpoint-{step}'.
    Returns the path to the latest checkpoint, or None if no checkpoints exist.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    # Find all checkpoint directories
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)$")
    checkpoints = []

    for item in output_path.iterdir():
        if item.is_dir():
            match = checkpoint_pattern.match(item.name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, item))

    if not checkpoints:
        return None

    # Sort by step number and return the latest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_checkpoint = checkpoints[0][1]

    return str(latest_checkpoint)


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
    resume_from_checkpoint: str | None = None,
    **kwargs,
) -> None:
    """Run Stage 3: Reinforcement Learning with correctness rewards."""

    print("=" * 60)
    print("Stage 3: RL with Correctness Rewards")
    print("=" * 60)

    # Handle resume from checkpoint
    checkpoint_path = None
    if resume_from_checkpoint == "auto":
        checkpoint_path = get_latest_checkpoint(output_dir)
        if checkpoint_path:
            print(f"Auto-detected latest checkpoint: {checkpoint_path}")
        else:
            print("No existing checkpoints found, starting from scratch")
    elif resume_from_checkpoint:
        checkpoint_path = resume_from_checkpoint
        if not Path(checkpoint_path).exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        print(f"Resuming from specified checkpoint: {checkpoint_path}")

    if "report_to" not in kwargs:
        if os.getenv("WANDB_DISABLED") or not os.getenv("WANDB_API_KEY"):
            kwargs["report_to"] = "tensorboard"

    config = RLMathConfig(
        model_name_or_path=base_model,
        output_dir=output_dir,
        **kwargs,
    )

    # Create reward function based on config
    if config.use_llm_judge:
        print(f"Using LLM Judge reward function (weight: {config.llm_judge_weight})")
        # Load separate judge model if specified
        judge_model = None
        judge_tokenizer = None
        if config.llm_judge_model:
            print(f"Loading separate judge model: {config.llm_judge_model}")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            judge_tokenizer = AutoTokenizer.from_pretrained(config.llm_judge_model)
            if judge_tokenizer.pad_token is None:
                judge_tokenizer.pad_token = judge_tokenizer.eos_token
            judge_model = AutoModelForCausalLM.from_pretrained(
                config.llm_judge_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            judge_model.eval()
        else:
            print("Using policy model as judge")

        reward_fn = LLMJudgeRewardFunction(
            answer_verifier=verify_answer,
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            correctness_weight=config.judge_correctness_weight,
            consistency_weight=config.judge_consistency_weight,
            clarity_weight=config.judge_clarity_weight,
            llm_judge_weight=config.llm_judge_weight,
            correct_reward=config.correct_reward,
            incorrect_reward=config.incorrect_reward,
            format_reward=config.format_reward,
        )
    else:
        # Standard reward function using answer verification
        reward_fn = MathRewardFunction(
            answer_verifier=verify_answer,
            correct_reward=config.correct_reward,
            incorrect_reward=config.incorrect_reward,
            format_reward=config.format_reward,
        )

    train_dataset, eval_dataset = load_math_datasets("rl", max_samples)
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    trainer = RLMathTrainer(config, reward_fn)
    trainer.train(train_dataset, eval_dataset, resume_from_checkpoint=checkpoint_path)


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
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU execution (no GPU)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Run evaluation every N steps during RL training",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="Number of eval samples to use per evaluation",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (higher = faster, default: 8)",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=None,
        help="Number of generations per sample for GRPO (default: 4, recommended: 8-16)",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Disable vLLM for generation",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable Flash Attention 2 (enabled by default)",
    )

    parser.add_argument(
        "--report-to",
        type=str,
        choices=["tensorboard", "wandb", "none"],
        default=None,
        help="Where to report metrics (default: tensorboard)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=None,
        help="Log training metrics every N steps (default: 5)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="rl_math_grpo",
        help="Name for the training run (useful for distinguishing runs, default: rl_math_grpo)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (default: 100)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help="Resume training from a checkpoint. If a path is provided, resume from that checkpoint. "
        "If no path is provided (just the flag), automatically find and resume from the latest checkpoint "
        "in the output directory.",
    )

    # LLM Judge arguments
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use LLM judge for process-based rewards (evaluates reasoning quality)",
    )
    parser.add_argument(
        "--llm-judge-model",
        type=str,
        default=None,
        help="Model to use as judge (default: use policy model)",
    )
    parser.add_argument(
        "--llm-judge-weight",
        type=float,
        default=None,
        help="Weight for LLM judge score vs answer correctness (0-1, default: 0.5)",
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
    if args.force_cpu:
        kwargs["force_cpu"] = True
        kwargs["device_map"] = "cpu"
        kwargs["torch_dtype"] = "float32"
    if args.eval_steps is not None:
        kwargs["eval_steps"] = args.eval_steps
    if args.eval_samples is not None:
        kwargs["eval_samples"] = args.eval_samples
    if args.eval_batch_size is not None:
        kwargs["eval_batch_size"] = args.eval_batch_size
    if args.num_generations is not None:
        kwargs["num_generations"] = args.num_generations
    if args.no_vllm:
        kwargs["use_vllm"] = False
    if args.no_flash_attn:
        kwargs["use_flash_attention"] = False
    if args.use_llm_judge:
        kwargs["use_llm_judge"] = True
    if args.llm_judge_model is not None:
        kwargs["llm_judge_model"] = args.llm_judge_model
    if args.llm_judge_weight is not None:
        kwargs["llm_judge_weight"] = args.llm_judge_weight
    if args.report_to is not None:
        kwargs["report_to"] = args.report_to
    if args.logging_steps is not None:
        kwargs["logging_steps"] = args.logging_steps
    if args.run_name is not None:
        kwargs["run_name"] = args.run_name
    if args.num_epochs is not None:
        kwargs["num_train_epochs"] = args.num_epochs
    if args.save_steps is not None:
        kwargs["save_steps"] = args.save_steps

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
            base_model=args.base_model,
            output_dir=f"{args.output_dir}/{args.run_name}",
            max_samples=args.max_samples,
            resume_from_checkpoint=args.resume_from_checkpoint,
            **kwargs,
        )


if __name__ == "__main__":
    main()
