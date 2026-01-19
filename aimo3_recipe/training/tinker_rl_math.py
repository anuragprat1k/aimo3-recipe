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

    # Setup logging
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(),
        ]
    )

    logger.info(f"Starting math RL training with config: {config}")

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

        for batch_start in range(0, len(dataset), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(dataset))
            batch = dataset.select(range(batch_start, batch_end))

            batch_data = []

            for example in batch:
                problem = example["problem"]
                ground_truth = example["solution"]

                # Format prompt
                prompt = renderer.render_prompt(problem)

                # Generate multiple solutions
                responses = sampling_client.sample(
                    prompt=prompt,
                    n=config.group_size,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )

                # Compute rewards for each response
                rewards = []
                for response in responses:
                    reward = compute_reward(response.text, ground_truth, config)
                    rewards.append(reward)

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

                # Create training data for positive advantage samples
                for response, reward, advantage in zip(responses, rewards, advantages):
                    if advantage > 0:
                        # Prepare training datum
                        full_text = prompt + response.text

                        datum = {
                            "input_text": full_text,
                            "target_text": response.text,
                            "log_probs": response.log_probs,
                            "advantage": advantage,
                        }
                        batch_data.append(datum)

            # Skip if no positive advantage samples
            if not batch_data:
                continue

            # Forward-backward pass on batch
            training_client.forward_backward(
                data=batch_data,
                loss_type="importance_sampling",
            )

            # Optimization step
            training_client.optim_step(
                optimizer="adam",
                learning_rate=learning_rate,
            )

            step += 1

            # Logging
            if step % 10 == 0:
                avg_reward = sum(total_rewards[-100:]) / len(total_rewards[-100:])
                accuracy = correct_count / total_count if total_count > 0 else 0
                logger.info(
                    f"Step {step}: avg_reward={avg_reward:.4f}, "
                    f"accuracy={accuracy:.2%}, "
                    f"batch_samples={len(batch_data)}"
                )

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


def main():
    """CLI entry point."""
    blueprint = build_config_blueprint()
    config = blueprint.make_from_argv()

    asyncio.run(train_math_rl(config))


if __name__ == "__main__":
    main()
