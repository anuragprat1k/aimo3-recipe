"""
Tinker-based SFT Training Loop for Mathematical Reasoning

Implements supervised fine-tuning using the Tinker SDK for:
- Stage 1: Chain-of-Thought (CoT) training
- Stage 2: Tool-Integrated Reasoning (TIR) training

The Tinker SDK offloads training to remote GPU clusters.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

import chz
import tinker
from datasets import load_dataset

from tinker_cookbook.abstractions.renderers import Qwen2ChatRenderer, Llama3ChatRenderer
from tinker_cookbook.abstractions.hparams import compute_lora_learning_rate


logger = logging.getLogger(__name__)


@dataclass
class MathSFTConfig:
    """Configuration for Tinker-based math SFT training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-14B"
    lora_rank: int = 64

    # Training stage
    stage: Literal["cot", "tir"] = "cot"

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: Optional[float] = None
    max_tokens: int = 4096
    num_epochs: int = 3

    # Dataset
    dataset_name: Optional[str] = None  # Auto-selected based on stage
    dataset_split: str = "train"
    max_samples: Optional[int] = None

    # Checkpointing
    save_every: int = 500
    log_dir: str = "./outputs/tinker_sft_math"

    # Resume
    resume_from: Optional[str] = None


def get_dataset_for_stage(stage: str, max_samples: Optional[int] = None):
    """Load appropriate dataset based on training stage."""
    if stage == "cot":
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    elif stage == "tir":
        dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train")
    else:
        raise ValueError(f"Unknown stage: {stage}")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


def get_renderer(model_name: str):
    """Get appropriate chat renderer for the model."""
    if "qwen" in model_name.lower():
        return Qwen2ChatRenderer()
    elif "llama" in model_name.lower():
        return Llama3ChatRenderer()
    else:
        return Qwen2ChatRenderer()


def build_config_blueprint() -> chz.Blueprint:
    """Build configuration blueprint for CLI usage."""
    return chz.Blueprint(MathSFTConfig)


async def train_math_sft(config: MathSFTConfig) -> None:
    """
    Main SFT training loop for math reasoning using Tinker.

    Processes (problem, solution) pairs and trains the model
    to generate solutions given problems.
    """

    # Setup logging
    log_dir = Path(config.log_dir) / config.stage
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(),
        ]
    )

    logger.info(f"Starting math SFT training - Stage: {config.stage}")
    logger.info(f"Config: {config}")

    # Initialize renderer
    renderer = get_renderer(config.model_name)

    # Compute learning rate
    learning_rate = config.learning_rate
    if learning_rate is None:
        learning_rate = compute_lora_learning_rate(config.lora_rank)
    logger.info(f"Using learning rate: {learning_rate}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = get_dataset_for_stage(config.stage, config.max_samples)
    logger.info(f"Dataset size: {len(dataset)}")

    # Initialize Tinker client
    logger.info("Initializing Tinker service client...")
    service_client = tinker.ServiceClient()

    # Create training client
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )

    # Resume from checkpoint
    step = 0
    if config.resume_from:
        logger.info(f"Resuming from: {config.resume_from}")
        training_client.load_state(config.resume_from)
        step = int(Path(config.resume_from).stem.split("-")[-1])

    # Training loop
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")

        dataset = dataset.shuffle(seed=epoch)

        epoch_loss = 0.0
        num_batches = 0

        for batch_start in range(0, len(dataset), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(dataset))
            batch = dataset.select(range(batch_start, batch_end))

            # Prepare training data
            batch_data = []
            for example in batch:
                problem = example["problem"]
                solution = example["solution"]

                # Format as chat
                prompt = renderer.render_prompt(problem)
                target = solution

                # Full sequence for training
                full_text = prompt + target

                datum = {
                    "input_text": full_text,
                    "target_text": target,
                }
                batch_data.append(datum)

            # Forward-backward pass
            loss = training_client.forward_backward(
                data=batch_data,
                loss_type="cross_entropy",
            )

            # Optimization step
            training_client.optim_step(
                optimizer="adam",
                learning_rate=learning_rate,
            )

            step += 1
            epoch_loss += loss
            num_batches += 1

            # Logging
            if step % 10 == 0:
                avg_loss = epoch_loss / num_batches
                logger.info(f"Step {step}: loss={loss:.4f}, avg_loss={avg_loss:.4f}")

            # Save checkpoint
            if step % config.save_every == 0:
                checkpoint_path = log_dir / f"checkpoint-{step}"
                training_client.save_state(str(checkpoint_path))
                logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")

    # Save final model
    final_path = log_dir / "final"
    training_client.save_state(str(final_path))
    logger.info(f"Training complete! Model saved to {final_path}")


async def train_full_pipeline(
    base_model: str = "Qwen/Qwen2.5-14B",
    output_base: str = "./outputs",
    max_samples: Optional[int] = None,
) -> None:
    """
    Run the complete SFT + RL pipeline:
    1. CoT SFT
    2. TIR SFT
    3. RL with correctness rewards
    """
    from aimo3_recipe.training.tinker_rl_math import train_math_rl, MathRLConfig

    logger.info("=" * 60)
    logger.info("AIMO3 Full Training Pipeline")
    logger.info("=" * 60)

    # Stage 1: CoT SFT
    logger.info("\n[Stage 1] Chain-of-Thought SFT")
    cot_config = MathSFTConfig(
        model_name=base_model,
        stage="cot",
        log_dir=f"{output_base}/sft_cot",
        max_samples=max_samples,
    )
    await train_math_sft(cot_config)

    # Stage 2: TIR SFT
    logger.info("\n[Stage 2] Tool-Integrated Reasoning SFT")
    tir_config = MathSFTConfig(
        model_name=base_model,  # Could use CoT checkpoint
        stage="tir",
        log_dir=f"{output_base}/sft_tir",
        max_samples=max_samples,
        resume_from=f"{output_base}/sft_cot/final",
    )
    await train_math_sft(tir_config)

    # Stage 3: RL
    logger.info("\n[Stage 3] RL with Correctness Rewards")
    rl_config = MathRLConfig(
        model_name=base_model,
        log_dir=f"{output_base}/rl_math",
        max_samples=max_samples,
        resume_from=f"{output_base}/sft_tir/final",
    )
    await train_math_rl(rl_config)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"Final model: {output_base}/rl_math/final")
    logger.info("=" * 60)


def main():
    """CLI entry point."""
    blueprint = build_config_blueprint()
    config = blueprint.make_from_argv()

    asyncio.run(train_math_sft(config))


if __name__ == "__main__":
    main()
