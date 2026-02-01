"""
Tinker-based RL Training Loop for Mathematical Reasoning

This module implements GRPO-style RL training using the Tinker SDK,
which offloads distributed training to remote GPU clusters.

Based on:
- Project Numina (AIMO1): CoT + TIR pipeline
- NemoSkills (AIMO2): Long-reasoning + solution selection
- Tinker Cookbook patterns for RL training

Variable naming convention (from tinker-cookbook):
    _P: Problem dimension (different questions/prompts in a batch)
    _G: Group dimension (multiple rollouts per problem)
    _D: Datum dimension (training examples after flattening)
"""

import json
import logging
import re
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chz
import tinker
import torch
from datasets import load_dataset
from tinker import types
from tinker.types.tensor_data import TensorData
from tqdm import tqdm

# Tinker imports
from tinker_cookbook.renderers import Qwen3Renderer, Renderer
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.utils.ml_log import setup_logging

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


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
    # Big-Math RL-Verified: 250k+ verified problems designed for RL training
    # Reference: arXiv:2502.17387 "Big-Math" (Feb 2025)
    dataset_name: str = "SynthLabsAI/Big-Math-RL-Verified"
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
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    verbose: bool = False

    # Data saving options
    save_samples: bool = False  # Save sampled responses to disk for analysis


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the answer from \\boxed{} in the response."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """
    Extract answer from GSM8K format (#### answer).

    GSM8K uses '####' followed by the final numeric answer.

    Args:
        text: Solution text containing #### answer

    Returns:
        Extracted answer string or None if not found
    """
    if not text or "####" not in text:
        return None

    # Match #### followed by the answer (typically a number)
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if match:
        answer = match.group(1).strip()
        # Clean up any trailing punctuation
        answer = answer.rstrip(".")
        return answer

    return None


def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer using multiple strategies.

    Tries in order:
    1. \\boxed{} format
    2. GSM8K #### format
    """
    # Try boxed first
    answer = extract_boxed_answer(text)
    if answer:
        return answer

    # Try GSM8K format (#### answer)
    answer = extract_gsm8k_answer(text)
    if answer:
        return answer

    return None


def normalize_math_answer(answer: str) -> str:
    """Normalize mathematical answer for comparison."""
    if answer is None:
        return ""

    answer = answer.strip()
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\left|\\right', '', answer)
    answer = re.sub(r'\\frac\{(\d+)\}\{(\d+)\}', r'\1/\2', answer)
    answer = re.sub(r'\\dfrac\{(\d+)\}\{(\d+)\}', r'\1/\2', answer)
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
    gt_answer = extract_answer(ground_truth) or ground_truth

    reward = 0.0

    if extracted is not None:
        reward += config.format_bonus

    if extracted is not None:
        pred_normalized = normalize_math_answer(extracted)
        gt_normalized = normalize_math_answer(gt_answer)

        if pred_normalized == gt_normalized:
            reward += config.correct_reward
        else:
            try:
                pred_float = float(eval(pred_normalized))
                gt_float = float(eval(gt_normalized))
                if abs(pred_float - gt_float) < 1e-6:
                    reward += config.correct_reward
                else:
                    reward += config.incorrect_reward
            except Exception:
                reward += config.incorrect_reward
    else:
        reward += config.incorrect_reward

    return reward


def build_config_blueprint() -> chz.Blueprint:
    """Build configuration blueprint for CLI usage."""
    return chz.Blueprint(MathRLConfig)


def get_renderer(model_name: str, tokenizer) -> Renderer:
    """Get appropriate chat renderer for the model."""
    if "qwen" in model_name.lower():
        return Qwen3Renderer(tokenizer)
    else:
        return Qwen3Renderer(tokenizer)


def train_math_rl(config: MathRLConfig) -> None:
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

    ml_logger = setup_logging(
        log_dir=str(log_dir),
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

    logger.info(f"Starting math RL training with config: {config}")
    if config.wandb_project:
        logger.info(f"WandB logging enabled for project: {config.wandb_project}")

    # Setup samples file if saving samples
    samples_file = None
    if config.save_samples:
        samples_path = log_dir / "samples.jsonl"
        samples_file = open(samples_path, "w")
        logger.info(f"Saving samples to {samples_path}")

    # Load tokenizer for renderer
    from transformers import AutoTokenizer
    logger.info(f"Loading tokenizer for {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Initialize renderer
    renderer = get_renderer(config.model_name, tokenizer)
    logger.info(f"Using renderer: {type(renderer).__name__}")

    # Compute learning rate based on LoRA rank if not specified
    learning_rate = config.learning_rate
    if learning_rate is None:
        learning_rate = get_lr(config.model_name, config.lora_rank)
    logger.info(f"Using learning rate: {learning_rate}")

    # Load dataset
    logger.info(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    # Handle Big-Math column naming (may use 'question' instead of 'problem')
    if "question" in dataset.column_names and "problem" not in dataset.column_names:
        dataset = dataset.rename_column("question", "problem")
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    logger.info(f"Dataset size: {len(dataset)}")

    # Calculate number of batches
    n_batches = (len(dataset) + config.batch_size - 1) // config.batch_size
    logger.info(f"Training for {n_batches} batches per epoch, {config.num_epochs} epoch(s)")

    # Initialize Tinker client
    logger.info("Initializing Tinker service client...")
    service_client = tinker.ServiceClient()
    logger.info("Service client initialized")

    # Create training client with LoRA
    logger.info(f"Creating LoRA training client (rank={config.lora_rank})...")
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )
    logger.info("Training client created")

    # Sampling params for generation
    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )

    # Adam params for optimization
    adam_params = types.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    # Training metrics
    step = 0
    total_rewards = []
    correct_count = 0
    total_count = 0

    # Main training loop
    for epoch in range(config.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")

        # Shuffle dataset each epoch
        dataset = dataset.shuffle(seed=epoch)

        for batch_idx in range(n_batches):
            t_start = time.time()
            metrics: dict[str, float] = {
                "progress/batch": batch_idx,
                "progress/epoch": epoch,
                "optim/lr": learning_rate,
                "progress/done_frac": (batch_idx + 1) / n_batches,
            }

            # Get batch
            batch_start = batch_idx * config.batch_size
            batch_end = min((batch_idx + 1) * config.batch_size, len(dataset))
            batch = dataset.select(range(batch_start, batch_end))

            logger.info(f"Batch {batch_idx + 1}/{n_batches}: Processing {len(batch)} problems...")

            # Save weights for sampling and create sampling client
            logger.info("  Saving weights for sampler...")
            sampling_path = training_client.save_weights_for_sampler(
                name=f"{step:06d}"
            ).result().path
            sampling_client = service_client.create_sampling_client(model_path=sampling_path)
            logger.info(f"  Sampling client created with path: {sampling_path}")

            # Submit all sampling requests first (batch them for efficiency)
            futures_P: list[Future[types.SampleResponse]] = []
            prompts_P: list[types.ModelInput] = []
            ground_truths_P: list[str] = []

            logger.info(f"  Submitting {len(batch)} sampling requests (group_size={config.group_size})...")
            for example in batch:
                problem = example["problem"]
                # Support both Big-Math ("answer") and NuminaMath ("solution") column names
                ground_truth = example.get("answer") or example.get("solution", "")

                messages = [{"role": "user", "content": problem}]
                prompt = renderer.build_generation_prompt(messages)

                # Submit sampling request (non-blocking)
                future = sampling_client.sample(
                    prompt=prompt,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                )
                futures_P.append(future)
                prompts_P.append(prompt)
                ground_truths_P.append(ground_truth)

            logger.info(f"  All sampling requests submitted, waiting for results...")

            # Collect results with progress bar
            datums_D: list[types.Datum] = []
            rewards_P: list[float] = []
            problems_P: list[str] = []

            # Get problems for sample saving
            for example in batch:
                problems_P.append(example["problem"])

            for idx, (future, prompt, ground_truth) in enumerate(tqdm(
                zip(futures_P, prompts_P, ground_truths_P),
                total=len(futures_P),
                desc=f"Sampling batch {batch_idx + 1}",
            )):
                sample_result = future.result()
                problem = problems_P[idx]

                rewards_G: list[float] = []
                sampled_tokens_G: list[list[int]] = []
                logprobs_G: list[list[float]] = []
                responses_G: list[str] = []

                for sequence in sample_result.sequences:
                    sampled_tokens = sequence.tokens
                    sampled_logprobs = sequence.logprobs
                    assert sampled_logprobs is not None

                    sampled_tokens_G.append(sampled_tokens)
                    logprobs_G.append(list(sampled_logprobs))

                    # Decode and compute reward
                    response_text = tokenizer.decode(sampled_tokens, skip_special_tokens=True)
                    responses_G.append(response_text)
                    reward = compute_reward(response_text, ground_truth, config)
                    rewards_G.append(reward)

                    # Track metrics
                    total_count += 1
                    if reward > 0.5:
                        correct_count += 1

                mean_reward = sum(rewards_G) / len(rewards_G)
                advantages_G = [reward - mean_reward for reward in rewards_G]
                rewards_P.append(mean_reward)
                total_rewards.append(mean_reward)

                # Save samples to disk if enabled
                if samples_file is not None:
                    sample_record = {
                        "step": step,
                        "batch_idx": batch_idx,
                        "problem_idx": idx,
                        "problem": problem,
                        "ground_truth": ground_truth,
                        "responses": [
                            {
                                "text": resp,
                                "reward": rew,
                                "advantage": adv,
                                "extracted_answer": extract_boxed_answer(resp),
                            }
                            for resp, rew, adv in zip(responses_G, rewards_G, advantages_G)
                        ],
                        "mean_reward": mean_reward,
                    }
                    samples_file.write(json.dumps(sample_record) + "\n")
                    samples_file.flush()  # Flush to ensure data is written

                # Skip if all advantages are zero (no learning signal)
                if all(advantage == 0.0 for advantage in advantages_G):
                    continue

                # Create training datums
                ob_len = prompt.length - 1
                for sampled_tokens, logprobs, advantage in zip(
                    sampled_tokens_G, logprobs_G, advantages_G
                ):
                    model_input = prompt.append(types.EncodedTextChunk(tokens=sampled_tokens[:-1]))
                    target_tokens = [0] * ob_len + sampled_tokens
                    padded_logprobs = [0.0] * ob_len + logprobs
                    padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)

                    assert (
                        model_input.length
                        == len(target_tokens)
                        == len(padded_logprobs)
                        == len(padded_advantages)
                    ), (
                        f"Length mismatch: model_input={model_input.length}, "
                        f"target_tokens={len(target_tokens)}, "
                        f"logprobs={len(padded_logprobs)}, "
                        f"advantages={len(padded_advantages)}"
                    )

                    datum = types.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                            "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                        },
                    )
                    datums_D.append(datum)

            # Skip if no datums
            if not datums_D:
                logger.warning(f"  No training datums for batch {batch_idx + 1}, skipping...")
                continue

            logger.info(f"  Created {len(datums_D)} training datums")

            # Training step
            logger.info("  Running forward-backward pass...")
            fwd_bwd_future = training_client.forward_backward(datums_D, loss_fn="importance_sampling")
            optim_step_future = training_client.optim_step(adam_params)
            fwd_bwd_result = fwd_bwd_future.result()
            optim_step_future.result()
            logger.info("  Training step complete")

            step += 1

            # Compute metrics
            avg_reward = sum(rewards_P) / len(rewards_P) if rewards_P else 0.0
            accuracy = correct_count / total_count if total_count > 0 else 0.0

            metrics["time/total"] = time.time() - t_start
            metrics["reward/batch_mean"] = avg_reward
            metrics["reward/running_mean"] = sum(total_rewards[-100:]) / max(len(total_rewards[-100:]), 1)
            metrics["train/accuracy"] = accuracy
            metrics["train/datums"] = len(datums_D)
            metrics["train/step"] = step

            # Add loss metrics from forward_backward result
            if hasattr(fwd_bwd_result, 'metrics') and fwd_bwd_result.metrics:
                for key, value in fwd_bwd_result.metrics.items():
                    metrics[f"loss/{key}"] = value

            # Log metrics
            ml_logger.log_metrics(metrics, step=step)

            # Console logging
            logger.info(
                f"  Step {step}: reward={avg_reward:.4f}, "
                f"accuracy={accuracy:.2%}, "
                f"datums={len(datums_D)}, "
                f"time={metrics['time/total']:.1f}s"
            )
            if config.verbose and hasattr(fwd_bwd_result, 'metrics') and fwd_bwd_result.metrics:
                loss_str = ", ".join(f"{k}={v:.6f}" for k, v in fwd_bwd_result.metrics.items())
                logger.info(f"    Loss metrics: {loss_str}")

            # Save checkpoint
            if config.save_every > 0 and step % config.save_every == 0:
                checkpoint_path = log_dir / f"checkpoint-{step:06d}"
                training_client.save_state(str(checkpoint_path))
                logger.info(f"  Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_path = log_dir / "final"
    training_client.save_state(str(final_path))
    logger.info(f"Training complete! Final model saved to {final_path}")

    # Log final statistics
    final_accuracy = correct_count / total_count if total_count > 0 else 0
    logger.info(f"Final accuracy: {final_accuracy:.2%} ({correct_count}/{total_count})")

    ml_logger.log_metrics({
        "final/accuracy": final_accuracy,
        "final/correct_count": correct_count,
        "final/total_count": total_count,
    }, step=step)

    ml_logger.close()

    # Close samples file if open
    if samples_file is not None:
        samples_file.close()
        logger.info(f"Samples saved to {log_dir / 'samples.jsonl'}")

    logger.info("Training completed")


def main():
    """CLI entry point."""
    blueprint = build_config_blueprint()
    config = blueprint.make_from_argv()
    train_math_rl(config)


if __name__ == "__main__":
    main()
