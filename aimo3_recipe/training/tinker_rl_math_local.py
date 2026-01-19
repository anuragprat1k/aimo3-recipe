"""
Local RL Training Simulation for Mathematical Reasoning

This module implements a local simulation of GRPO-style RL training
that works without network access for demonstration purposes.
"""

import asyncio
import json
import logging
import os
import re
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(name)s:%(lineno)d [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MathRLConfig:
    """Configuration for local math RL training simulation."""

    # Model (simulated)
    model_name: str = "Qwen/Qwen2.5-14B"
    lora_rank: int = 64

    # Training hyperparameters
    batch_size: int = 64
    group_size: int = 16
    learning_rate: float = 1e-5
    max_tokens: int = 4096

    # Dataset (simulated)
    max_samples: int = 100

    # Reward configuration
    correct_reward: float = 1.0
    incorrect_reward: float = -0.5
    format_bonus: float = 0.1

    # Training loop
    num_epochs: int = 1
    save_every: int = 100
    log_dir: str = "./outputs/tinker_rl_math"

    # Generation
    temperature: float = 0.7
    top_p: float = 0.95

    # Logging options
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    verbose: bool = False


class LocalLogger:
    """Simple logger that saves to JSONL and optionally to WandB offline."""

    def __init__(self, log_dir: str, wandb_project: Optional[str] = None,
                 wandb_name: Optional[str] = None, config: Any = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.wandb_run = None

        # Try WandB in offline mode
        if wandb_project:
            try:
                import wandb
                os.environ["WANDB_MODE"] = "offline"
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config=asdict(config) if config else None,
                    settings=wandb.Settings(init_timeout=30)
                )
                logger.info(f"WandB initialized in offline mode")
            except Exception as e:
                logger.warning(f"Could not initialize WandB: {e}")
                self.wandb_run = None

        logger.info(f"Logging to: {log_dir}")

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to JSONL file and WandB."""
        metrics["timestamp"] = datetime.now().isoformat()

        # Write to JSONL
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Log to WandB if available
        if self.wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except:
                pass

    def close(self):
        """Close the logger."""
        if self.wandb_run:
            try:
                import wandb
                wandb.finish()
            except:
                pass


def generate_mock_math_problems(n: int) -> List[Dict[str, str]]:
    """Generate mock math problems for training."""
    problems = []
    templates = [
        ("What is {a} + {b}?", lambda a, b: a + b),
        ("Calculate {a} * {b}.", lambda a, b: a * b),
        ("If x = {a}, what is 2x + {b}?", lambda a, b: 2*a + b),
        ("Solve: {a} - {b} = ?", lambda a, b: a - b),
        ("What is {a}^2 + {b}?", lambda a, b: a**2 + b),
    ]

    for i in range(n):
        template, solver = random.choice(templates)
        a, b = random.randint(1, 50), random.randint(1, 50)
        problem = template.format(a=a, b=b)
        answer = solver(a, b)
        solution = f"Let me solve this step by step.\n\nGiven the problem: {problem}\n\nCalculating: The answer is \\boxed{{{answer}}}"
        problems.append({
            "problem": problem,
            "solution": solution,
            "answer": str(answer)
        })

    return problems


def simulate_model_response(problem: str, ground_truth: str, temperature: float) -> str:
    """Simulate a model generating a response."""
    # Extract the correct answer from ground truth
    match = re.search(r'\\boxed\{([^}]+)\}', ground_truth)
    correct_answer = match.group(1) if match else "0"

    # Simulate model behavior - sometimes correct, sometimes wrong
    if random.random() < 0.6 + (1 - temperature) * 0.3:  # Higher temp = more random
        # Correct response
        response = f"I'll solve this problem step by step.\n\n{problem}\n\nAfter careful calculation, the answer is \\boxed{{{correct_answer}}}"
    else:
        # Wrong response (off by some amount)
        try:
            wrong = int(correct_answer) + random.randint(-10, 10)
        except:
            wrong = random.randint(1, 100)
        response = f"Let me think about this.\n\n{problem}\n\nI believe the answer is \\boxed{{{wrong}}}"

    return response


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the answer from \\boxed{} in the response."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def compute_reward(response: str, ground_truth: str, config: MathRLConfig) -> float:
    """Compute reward for a model response."""
    extracted = extract_boxed_answer(response)
    gt_answer = extract_boxed_answer(ground_truth) or ground_truth

    reward = 0.0

    # Format bonus for using boxed
    if extracted is not None:
        reward += config.format_bonus

    # Correctness check
    if extracted is not None:
        try:
            if str(extracted).strip() == str(gt_answer).strip():
                reward += config.correct_reward
            else:
                reward += config.incorrect_reward
        except:
            reward += config.incorrect_reward
    else:
        reward += config.incorrect_reward

    return reward


async def train_math_rl(config: MathRLConfig) -> None:
    """
    Main RL training loop simulation for math reasoning.

    Implements simulated GRPO-style training:
    1. Sample multiple solutions per problem
    2. Compute rewards based on answer correctness
    3. Calculate advantages (reward - mean_group_reward)
    4. Simulate training on positive advantage samples
    """

    # Setup logging
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ml_logger = LocalLogger(
        log_dir=str(log_dir),
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
    )

    logger.info(f"Starting math RL training simulation with config:")
    for key, value in asdict(config).items():
        logger.info(f"  {key}: {value}")

    if config.wandb_project:
        logger.info(f"WandB logging enabled for project: {config.wandb_project}")

    # Generate mock dataset
    logger.info(f"Generating mock dataset with {config.max_samples} samples...")
    dataset = generate_mock_math_problems(config.max_samples)
    logger.info(f"Dataset size: {len(dataset)}")

    # Simulated learning rate calculation
    learning_rate = config.learning_rate
    logger.info(f"Using learning rate: {learning_rate}")

    # Training metrics
    total_rewards = []
    correct_count = 0
    total_count = 0
    step = 0

    # Simulate training improvement over time
    base_accuracy = 0.4

    # Main training loop
    for epoch in range(config.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")

        random.shuffle(dataset)

        for batch_start in range(0, len(dataset), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(dataset))
            batch = dataset[batch_start:batch_end]

            batch_rewards = []
            batch_correct = 0

            for example in batch:
                problem = example["problem"]
                ground_truth = example["solution"]

                # Generate multiple solutions (simulated)
                rewards = []
                for _ in range(config.group_size):
                    # Simulate improvement over training
                    adjusted_temp = config.temperature * (1 - step * 0.001)
                    response = simulate_model_response(problem, ground_truth, adjusted_temp)
                    reward = compute_reward(response, ground_truth, config)
                    rewards.append(reward)

                    total_count += 1
                    if reward > 0.5:
                        correct_count += 1
                        batch_correct += 1

                batch_rewards.extend(rewards)

            total_rewards.extend(batch_rewards)

            # Simulate gradient update
            step += 1

            # Compute metrics
            avg_reward = sum(total_rewards[-100:]) / max(len(total_rewards[-100:]), 1)
            accuracy = correct_count / total_count if total_count > 0 else 0

            # Simulate loss metrics
            simulated_loss = max(0.1, 2.0 - step * 0.05 + random.gauss(0, 0.1))
            simulated_kl = max(0.001, 0.1 - step * 0.002 + random.gauss(0, 0.01))

            metrics = {
                "train/step": step,
                "train/avg_reward": avg_reward,
                "train/accuracy": accuracy,
                "train/batch_datums": len(batch) * config.group_size,
                "train/learning_rate": learning_rate,
                "loss/total": simulated_loss,
                "loss/kl_divergence": simulated_kl,
            }

            # Verbose logging
            if config.verbose or step % 10 == 0:
                logger.info(
                    f"Step {step}: avg_reward={avg_reward:.4f}, "
                    f"accuracy={accuracy:.2%}, "
                    f"batch_datums={len(batch) * config.group_size}"
                )
                if config.verbose:
                    logger.info(f"  Loss metrics: total={simulated_loss:.6f}, kl={simulated_kl:.6f}")

            # Log to file and WandB
            ml_logger.log_metrics(metrics, step=step)

            # Save checkpoint
            if step % config.save_every == 0:
                checkpoint_path = log_dir / f"checkpoint-{step}"
                checkpoint_path.mkdir(parents=True, exist_ok=True)

                # Save checkpoint info
                with open(checkpoint_path / "info.json", "w") as f:
                    json.dump({
                        "step": step,
                        "accuracy": accuracy,
                        "avg_reward": avg_reward,
                        "total_count": total_count,
                        "correct_count": correct_count,
                    }, f, indent=2)

                logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final model info
    final_path = log_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)

    final_accuracy = correct_count / total_count if total_count > 0 else 0

    with open(final_path / "results.json", "w") as f:
        json.dump({
            "final_accuracy": final_accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "total_steps": step,
            "config": asdict(config),
        }, f, indent=2)

    logger.info(f"Training complete! Final model saved to {final_path}")
    logger.info(f"Final accuracy: {final_accuracy:.2%} ({correct_count}/{total_count})")

    # Log final metrics
    ml_logger.log_metrics({
        "final/accuracy": final_accuracy,
        "final/correct_count": correct_count,
        "final/total_count": total_count,
    }, step=step)

    # Cleanup logger
    ml_logger.close()

    # Print WandB sync instructions
    wandb_dir = log_dir / "wandb"
    if wandb_dir.exists():
        print(f"\nTo sync WandB data when online, run:")
        for run_dir in wandb_dir.glob("offline-run-*"):
            print(f"  wandb sync {run_dir}")


def main():
    """CLI entry point."""
    import sys

    # Parse command line arguments (key=value format)
    config_dict = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Type conversion
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.lower() == "none":
                value = None
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            config_dict[key] = value

    config = MathRLConfig(**config_dict)
    asyncio.run(train_math_rl(config))


if __name__ == "__main__":
    main()
