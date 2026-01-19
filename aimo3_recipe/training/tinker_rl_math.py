"""
Tinker-based RL Training Loop for Mathematical Reasoning

This module implements GRPO-style RL training using the Tinker SDK,
which offloads distributed training to remote GPU clusters.

Based on:
- Project Numina (AIMO1): CoT + TIR pipeline
- NemoSkills (AIMO2): Long-reasoning + solution selection
- Tinker Cookbook patterns for RL training
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import chz
import tinker
from datasets import load_dataset, Dataset

# Tinker imports
from tinker_cookbook.renderers import Qwen3Renderer, Renderer
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.utils.ml_log import setup_logging, Logger


logger = logging.getLogger(__name__)


@dataclass
class MathRLConfig:
    """Configuration for Tinker-based math RL training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-14B"
    lora_rank: int = 64

    # Training hyperparameters
    batch_size: int = 64
    group_size: int = 16  # Number of rollouts per problem
    learning_rate: Optional[float] = None  # Auto-computed based on LoRA rank
    max_tokens: int = 4096

    # Dataset
    dataset_name: str = "AI-MO/NuminaMath-CoT"
    dataset_split: str = "train"
    max_samples: Optional[int] = None

    # Reward configuration
    correct_reward: float = 1.0
    incorrect_reward: float = -0.5
    format_bonus: float = 0.1

    # Training loop
    num_epochs: int = 1
    save_every: int = 100
    eval_every: int = 0
    log_dir: str = "./outputs/tinker_rl_math"

    # Generation
    temperature: float = 0.7
    top_p: float = 0.95

    # Resume from checkpoint
    resume_from: Optional[str] = None

    # Logging options
    wandb_project: Optional[str] = None  # Set to enable WandB logging (e.g., "aimo3-rl-math")
    wandb_name: Optional[str] = None  # Optional run name for WandB
    verbose: bool = False  # Enable verbose loss output every step


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the answer from \\boxed{} in the response."""
    # Handle nested braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()  # Return last boxed answer
    return None


def normalize_math_answer(answer: str) -> str:
    """Normalize mathematical answer for comparison."""
    if answer is None:
        return ""

    answer = answer.strip()

    # Remove LaTeX formatting
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\left|\\right', '', answer)

    # Normalize fractions
    answer = re.sub(r'\\frac\{(\d+)\}\{(\d+)\}', r'\1/\2', answer)
    answer = re.sub(r'\\dfrac\{(\d+)\}\{(\d+)\}', r'\1/\2', answer)

    # Remove spaces and convert to lowercase for comparison
    answer = answer.replace(' ', '').lower()

    return answer


def compute_reward(
    response: str,
    ground_truth: str,
    config: MathRLConfig,
) -> float:
    """
    Compute reward for a model response.

    Rewards:
    - Correct answer: +1.0
    - Incorrect answer: -0.5
    - Proper boxed format: +0.1 bonus
    """
    extracted = extract_boxed_answer(response)
    gt_answer = extract_boxed_answer(ground_truth) or ground_truth

    reward = 0.0

    # Format bonus for using boxed
    if extracted is not None:
        reward += config.format_bonus

    # Correctness check
    if extracted is not None:
        pred_normalized = normalize_math_answer(extracted)
        gt_normalized = normalize_math_answer(gt_answer)

        if pred_normalized == gt_normalized:
            reward += config.correct_reward
        else:
            # Try numeric comparison for floating point answers
            try:
                pred_float = float(eval(pred_normalized))
                gt_float = float(eval(gt_normalized))
                if abs(pred_float - gt_float) < 1e-6:
                    reward += config.correct_reward
                else:
                    reward += config.incorrect_reward
            except:
                reward += config.incorrect_reward
    else:
        reward += config.incorrect_reward

    return reward


def build_config_blueprint() -> chz.Blueprint:
    """Build configuration blueprint for CLI usage."""
    return chz.Blueprint(MathRLConfig)


def get_renderer(model_name: str, tokenizer) -> Renderer:
    """Get appropriate chat renderer for the model."""
    # Use Qwen3Renderer for Qwen models, default renderer otherwise
    if "qwen" in model_name.lower():
        return Qwen3Renderer(tokenizer)
    else:
        # Default to Qwen3 format (most common for math models)
        return Qwen3Renderer(tokenizer)


async def train_math_rl(config: MathRLConfig) -> None:
    """
    Main RL training loop for math reasoning using Tinker.

    Implements GRPO-style training:
    1. Sample multiple solutions per problem
    2. Compute rewards based on answer correctness
    3. Calculate advantages (reward - mean_group_reward)
    4. Train on positive advantage samples
    """

    # Setup logging with optional WandB integration
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ml_logger (supports WandB, JSON, and console logging)
    ml_logger = setup_logging(
        log_dir=str(log_dir),
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
    )

    logger.info(f"Starting math RL training with config: {config}")
    if config.wandb_project:
        logger.info(f"WandB logging enabled for project: {config.wandb_project}")

    # Load tokenizer for renderer
    from transformers import AutoTokenizer
    logger.info(f"Loading tokenizer for {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Initialize renderer
    renderer = get_renderer(config.model_name, tokenizer)

    # Compute learning rate based on LoRA rank if not specified
    learning_rate = config.learning_rate
    if learning_rate is None:
        learning_rate = get_lr(config.model_name, config.lora_rank)
    logger.info(f"Using learning rate: {learning_rate}")

    # Load dataset
    logger.info(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    logger.info(f"Dataset size: {len(dataset)}")

    # Initialize Tinker client
    logger.info("Initializing Tinker service client...")
    service_client = tinker.ServiceClient()

    # Create training client with LoRA
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )

    # Resume from checkpoint if specified
    step = 0
    if config.resume_from:
        logger.info(f"Resuming from checkpoint: {config.resume_from}")
        training_client.load_state(config.resume_from)
        step = int(Path(config.resume_from).stem.split("-")[-1])

    # Get sampling client for generation
    sampling_client = training_client.save_weights_and_get_sampling_client()

    # Training metrics
    total_rewards = []
    correct_count = 0
    total_count = 0

    # Main training loop
    for epoch in range(config.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")

        # Shuffle dataset each epoch
        dataset = dataset.shuffle(seed=epoch)

        # Sampling params for generation
        sampling_params = tinker.types.SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=renderer.get_stop_sequences(),
        )

        # Adam params for optimization
        adam_params = tinker.types.AdamParams(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.0,
            grad_clip_norm=1.0,
        )

        for batch_start in range(0, len(dataset), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(dataset))
            batch = dataset.select(range(batch_start, batch_end))

            datums: list[tinker.types.Datum] = []

            for example in batch:
                problem = example["problem"]
                ground_truth = example["solution"]

                # Format prompt using renderer's message format
                messages = [{"role": "user", "content": problem}]
                prompt = renderer.build_generation_prompt(messages)

                # Generate multiple solutions
                sample_response = sampling_client.sample(
                    prompt=prompt,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                ).result()

                # Compute rewards for each response
                rewards = []
                response_data = []
                for sequence in sample_response.sequences:
                    sampled_tokens = sequence.tokens
                    sampled_logprobs = sequence.logprobs
                    assert sampled_logprobs is not None

                    # Decode tokens to text for reward computation
                    response_text = tokenizer.decode(sampled_tokens, skip_special_tokens=True)
                    reward = compute_reward(response_text, ground_truth, config)
                    rewards.append(reward)
                    response_data.append((sampled_tokens, sampled_logprobs))

                    # Track metrics
                    total_count += 1
                    if reward > 0.5:  # Correct answer
                        correct_count += 1

                total_rewards.extend(rewards)

                # Calculate advantages (GRPO-style reward centering)
                mean_reward = sum(rewards) / len(rewards)
                advantages = [r - mean_reward for r in rewards]

                # Skip if all advantages are zero (no learning signal)
                if all(abs(a) < 1e-6 for a in advantages):
                    continue

                # Create training datums following tinker-cookbook format
                prompt_len = prompt.length - 1  # observation length
                for (sampled_tokens, sampled_logprobs), advantage in zip(response_data, advantages):
                    # Build model input: prompt + response tokens (excluding last token)
                    model_input = prompt.append(tinker.types.EncodedTextChunk(tokens=sampled_tokens[:-1]))

                    # Pad target tokens, logprobs, and advantages
                    target_tokens = [0] * prompt_len + sampled_tokens
                    padded_logprobs = [0.0] * prompt_len + list(sampled_logprobs)
                    padded_advantages = [0.0] * prompt_len + [advantage] * (model_input.length - prompt_len)

                    assert model_input.length == len(target_tokens) == len(padded_logprobs) == len(padded_advantages)

                    datum = tinker.types.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData(data=target_tokens, dtype="int64", shape=[len(target_tokens)]),
                            "logprobs": tinker.TensorData(data=padded_logprobs, dtype="float32", shape=[len(padded_logprobs)]),
                            "advantages": tinker.TensorData(data=padded_advantages, dtype="float32", shape=[len(padded_advantages)]),
                        },
                    )
                    datums.append(datum)

            # Skip if no datums
            if not datums:
                continue

            # Forward-backward pass and optimization step
            fwd_bwd_future = training_client.forward_backward(datums, loss_fn="importance_sampling")
            optim_step_future = training_client.optim_step(adam_params)
            fwd_bwd_result = fwd_bwd_future.result()
            optim_step_future.result()

            step += 1

            # Compute metrics for logging
            avg_reward = sum(total_rewards[-100:]) / max(len(total_rewards[-100:]), 1)
            accuracy = correct_count / total_count if total_count > 0 else 0

            # Build metrics dictionary
            metrics = {
                "train/step": step,
                "train/avg_reward": avg_reward,
                "train/accuracy": accuracy,
                "train/batch_datums": len(datums),
                "train/learning_rate": learning_rate,
            }

            # Add loss metrics from forward_backward result
            if hasattr(fwd_bwd_result, 'metrics') and fwd_bwd_result.metrics:
                for key, value in fwd_bwd_result.metrics.items():
                    metrics[f"loss/{key}"] = value

            # Verbose logging every step
            if config.verbose or step % 10 == 0:
                logger.info(
                    f"Step {step}: avg_reward={avg_reward:.4f}, "
                    f"accuracy={accuracy:.2%}, "
                    f"batch_datums={len(datums)}"
                )
                if config.verbose and hasattr(fwd_bwd_result, 'metrics') and fwd_bwd_result.metrics:
                    loss_str = ", ".join(f"{k}={v:.6f}" for k, v in fwd_bwd_result.metrics.items())
                    logger.info(f"  Loss metrics: {loss_str}")

            # Log to WandB/JSON
            ml_logger.log_metrics(metrics, step=step)

            # Save checkpoint
            if step % config.save_every == 0:
                checkpoint_path = log_dir / f"checkpoint-{step}"
                training_client.save_state(str(checkpoint_path))
                logger.info(f"Saved checkpoint to {checkpoint_path}")

                # Update sampling client with new weights
                sampling_client = training_client.save_weights_and_get_sampling_client()

    # Save final model
    final_path = log_dir / "final"
    training_client.save_state(str(final_path))
    logger.info(f"Training complete! Final model saved to {final_path}")

    # Log final statistics
    final_accuracy = correct_count / total_count if total_count > 0 else 0
    logger.info(f"Final accuracy: {final_accuracy:.2%} ({correct_count}/{total_count})")

    # Log final metrics
    ml_logger.log_metrics({
        "final/accuracy": final_accuracy,
        "final/correct_count": correct_count,
        "final/total_count": total_count,
    }, step=step)

    # Cleanup logger (important for WandB to finish uploading)
    ml_logger.close()


def main():
    """CLI entry point."""
    blueprint = build_config_blueprint()
    config = blueprint.make_from_argv()

    asyncio.run(train_math_rl(config))


if __name__ == "__main__":
    main()
