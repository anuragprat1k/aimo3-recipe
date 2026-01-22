"""
Reinforcement Learning for Mathematical Reasoning

This module implements RL-based training for math problem solving,
using correctness rewards based on answer verification.

Uses GRPO (Group Relative Policy Optimization) for training.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
import numpy as np


@dataclass
class RLMathConfig:
    """Configuration for RL math training."""

    # Model
    model_name_or_path: str = "./outputs/sft_tir"
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    device_map: str = "auto"
    torch_dtype: Optional[str] = "bfloat16"
    force_cpu: bool = False

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # GRPO
    output_dir: str = "./outputs/rl_math"
    learning_rate: float = 1e-6
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_generations: int = 4  # Group size for GRPO
    max_grad_norm: float = 0.5

    # Generation
    max_completion_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95

    # Reward
    correct_reward: float = 1.0
    incorrect_reward: float = -0.5
    format_reward: float = 0.1  # Bonus for proper \\boxed{} format
    length_penalty: float = 0.0  # Optional penalty for very long responses

    # KL
    beta: float = 0.1  # KL penalty coefficient

    # Training
    num_train_epochs: int = 1
    save_steps: int = 100
    logging_steps: int = 10
    report_to: str = "wandb"
    run_name: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    save_samples: bool = False
    sample_save_rate: float = 0.01
    samples_filename: str = "samples.jsonl"


class MathRewardFunction:
    """
    Reward function for math problem solving.

    Computes rewards based on:
    1. Answer correctness (main signal)
    2. Proper formatting (boxed answer)
    3. Optional length penalty
    """

    __name__ = "math_correctness_reward"

    def __init__(
        self,
        answer_verifier: Callable[[str, str], bool],
        correct_reward: float = 1.0,
        incorrect_reward: float = -0.5,
        format_reward: float = 0.1,
        length_penalty: float = 0.0,
    ):
        self.answer_verifier = answer_verifier
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.format_reward = format_reward
        self.length_penalty = length_penalty

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        answer: list[str],
        **kwargs,
    ) -> list[float]:
        """
        Compute rewards for a batch of responses.

        Args:
            prompts: List of input prompts
            completions: List of model completions
            answer: List of ground truth answers (from dataset)
            **kwargs: Additional columns from dataset (ignored)

        Returns:
            List of reward values
        """
        rewards = []
        for completion, ground_truth in zip(completions, answer):
            reward = 0.0

            # Check for boxed format
            has_boxed = "\\boxed{" in completion
            if has_boxed:
                reward += self.format_reward

            # Check answer correctness
            is_correct = self.answer_verifier(completion, ground_truth)
            if is_correct:
                reward += self.correct_reward
            else:
                reward += self.incorrect_reward

            # Optional length penalty
            if self.length_penalty > 0:
                length_factor = len(completion) / 1000  # Normalize by 1000 chars
                reward -= self.length_penalty * max(0, length_factor - 2)  # Penalize > 2k chars

            rewards.append(reward)

        return rewards


class RLMathTrainer:
    """
    Reinforcement learning trainer for math reasoning.

    Uses GRPO to optimize the model based on answer correctness rewards.
    """

    def __init__(
        self,
        config: RLMathConfig,
        reward_fn: Optional[MathRewardFunction] = None,
    ):
        self.config = config
        self.reward_fn = reward_fn
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Initialize model and tokenizer."""

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="left",  # Left padding for generation
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = None
        if self.config.torch_dtype:
            dtype = getattr(torch, self.config.torch_dtype)

        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": dtype,
        }

        # Don't use device_map with CPU to avoid offloading issues
        if not self.config.force_cpu:
            model_kwargs["device_map"] = self.config.device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **model_kwargs,
        )

        return self.model, self.tokenizer

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset for GRPO training.

        Expected input format:
        - problem: The math problem
        - answer: The ground truth answer (for reward computation)

        Output format:
        - prompt: The formatted prompt for the model
        - answer: The ground truth answer (passed to reward function)
        """

        def format_prompts(examples):
            prompts = []
            for problem in examples["problem"]:
                messages = [{"role": "user", "content": problem}]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)
            return {"prompt": prompts, "answer": examples["answer"]}

        dataset = dataset.map(
            format_prompts,
            batched=True,
            desc="Preparing GRPO dataset",
            remove_columns=[col for col in dataset.column_names if col not in ["answer"]],
        )

        return dataset

    def train(self, dataset: Dataset) -> None:
        """Run GRPO training."""

        if self.model is None:
            self.setup_model()

        dataset = self.prepare_dataset(dataset)

        # Configure wandb
        self._configure_wandb()

        # Build LoRA config if needed
        peft_config = None
        if self.config.use_lora:
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

        # GRPO configuration
        grpo_config = GRPOConfig(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_generations=self.config.num_generations,
            max_completion_length=self.config.max_completion_length,
            max_grad_norm=self.config.max_grad_norm,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            beta=self.config.beta,
            num_train_epochs=self.config.num_train_epochs,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,
            use_cpu=self.config.force_cpu,
            log_completions=True,
        )

        # Initialize GRPO trainer
        self.trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=self.reward_fn,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )

        # Train
        self.trainer.train()

        # Save final model
        self.trainer.save_model(self.config.output_dir)

    def _configure_wandb(self) -> None:
        if self.config.report_to != "wandb":
            return
        if self.config.wandb_project:
            os.environ["WANDB_PROJECT"] = self.config.wandb_project
        if self.config.wandb_name:
            os.environ["WANDB_RUN_NAME"] = self.config.wandb_name
