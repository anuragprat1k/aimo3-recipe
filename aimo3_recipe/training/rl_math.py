"""
Reinforcement Learning for Mathematical Reasoning

This module implements RL-based training for math problem solving,
using correctness rewards based on answer verification.

Supports both REINFORCE-style and PPO-style optimization.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
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

    # PPO
    output_dir: str = "./outputs/rl_math"
    learning_rate: float = 1e-6
    batch_size: int = 64
    mini_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    ppo_epochs: int = 4
    max_grad_norm: float = 0.5

    # Generation
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    num_return_sequences: int = 1

    # Reward
    correct_reward: float = 1.0
    incorrect_reward: float = -0.5
    format_reward: float = 0.1  # Bonus for proper \\boxed{} format
    length_penalty: float = 0.0  # Optional penalty for very long responses

    # KL
    init_kl_coef: float = 0.2
    target_kl: float = 0.1

    # Training
    num_train_epochs: int = 1
    save_steps: int = 100
    logging_steps: int = 10
    report_to: str = "wandb"
    run_name: Optional[str] = None
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
        response: str,
        ground_truth: str,
    ) -> float:
        """Compute reward for a single response."""
        reward = 0.0

        # Check for boxed format
        has_boxed = "\\boxed{" in response
        if has_boxed:
            reward += self.format_reward

        # Check answer correctness
        is_correct = self.answer_verifier(response, ground_truth)
        if is_correct:
            reward += self.correct_reward
        else:
            reward += self.incorrect_reward

        # Optional length penalty
        if self.length_penalty > 0:
            length_factor = len(response) / 1000  # Normalize by 1000 chars
            reward -= self.length_penalty * max(0, length_factor - 2)  # Penalize > 2k chars

        return reward


class RLMathTrainer:
    """
    Reinforcement learning trainer for math reasoning.

    Uses PPO to optimize the model based on answer correctness rewards.
    """

    def __init__(
        self,
        config: RLMathConfig,
        reward_fn: Optional[MathRewardFunction] = None,
    ):
        self.config = config
        self.reward_fn = reward_fn
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.ppo_trainer = None

    def setup_model(self) -> None:
        """Initialize model, reference model, and tokenizer."""

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

        # Load model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=dtype,
            device_map=self.config.device_map,
        )

        # Apply LoRA if specified (wrap the base model inside the value head)
        if self.config.use_lora:
            if getattr(self.model.pretrained_model, "peft_config", None):
                print("PEFT adapter already present; skipping LoRA injection.")
                self.model.is_peft_model = True
                self.ref_model = None
                return
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model.pretrained_model = get_peft_model(self.model.pretrained_model, lora_config)
            self.model.is_peft_model = True
            self.model.pretrained_model.print_trainable_parameters()
            self.ref_model = None
        else:
            # Reference model (frozen copy for KL penalty)
            self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=dtype,
                device_map=self.config.device_map,
            )

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset for RL training.

        Expected format:
        - problem: The math problem
        - answer: The ground truth answer (for reward computation)
        """

        def tokenize_prompts(examples):
            prompts = []
            for problem in examples["problem"]:
                messages = [{"role": "user", "content": problem}]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)

            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                max_length=1024,
                padding=False,
                return_tensors=None,
            )
            tokenized["query"] = prompts
            tokenized["answer"] = examples["answer"]
            return tokenized

        return dataset.map(
            tokenize_prompts,
            batched=True,
            desc="Preparing RL dataset",
        )

    def compute_rewards(
        self,
        responses: list[str],
        ground_truths: list[str],
    ) -> list[float]:
        """Compute rewards for a batch of responses."""
        if self.reward_fn is None:
            raise ValueError("Reward function not set")

        rewards = []
        for response, gt in zip(responses, ground_truths):
            reward = self.reward_fn(response, gt)
            rewards.append(reward)
        return rewards

    def train(self, dataset: Dataset) -> None:
        """Run RL training loop."""

        if self.model is None:
            self.setup_model()

        dataset = self.prepare_dataset(dataset)

        # PPO configuration
        accelerator_kwargs = {"cpu": True} if self.config.force_cpu else {}

        ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            ppo_epochs=self.config.ppo_epochs,
            max_grad_norm=self.config.max_grad_norm,
            init_kl_coef=self.config.init_kl_coef,
            target_kl=self.config.target_kl,
            log_with=self.config.report_to,
            project_kwargs={"logging_dir": self.config.output_dir},
            remove_unused_columns=False,
            accelerator_kwargs=accelerator_kwargs,
        )

        def data_collator(features: list[dict]) -> dict:
            batch = {
                "input_ids": [item["input_ids"] for item in features],
                "answer": [item["answer"] for item in features],
            }
            if features and "query" in features[0]:
                batch["query"] = [item["query"] for item in features]
            return batch

        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=dataset,
            data_collator=data_collator,
        )
        # Ensure generation uses the Accelerator device (MPS/CPU)
        self.ppo_trainer.current_device = self.ppo_trainer.accelerator.device

        # Generation config
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        samples_file = None
        if self.config.save_samples:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            samples_path = output_dir / self.config.samples_filename
            samples_file = samples_path.open("w", encoding="utf-8")

        # Training loop
        for epoch in range(self.config.num_train_epochs):
            for batch_idx, batch in enumerate(self.ppo_trainer.dataloader):
                query_tensors = [torch.tensor(q) for q in batch["input_ids"]]
                ground_truths = batch["answer"]
                if "query" in batch:
                    queries = batch["query"]
                else:
                    queries = [
                        self.tokenizer.decode(q, skip_special_tokens=True)
                        for q in query_tensors
                    ]

                # Generate responses
                response_tensors = self.ppo_trainer.generate(
                    query_tensors,
                    **generation_kwargs,
                )

                # Decode responses
                responses = self.tokenizer.batch_decode(
                    response_tensors,
                    skip_special_tokens=True,
                )

                # Compute rewards
                rewards = self.compute_rewards(responses, ground_truths)
                reward_tensors = [torch.tensor(r) for r in rewards]

                # PPO step
                stats = self.ppo_trainer.step(
                    query_tensors,
                    response_tensors,
                    reward_tensors,
                )

                # Log
                if batch_idx % self.config.logging_steps == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}")
                    print(f"  Mean reward: {np.mean(rewards):.4f}")
                    print(f"  Policy loss: {stats['ppo/loss/policy']:.4f}")
                    log_batch = {}
                    if "query" in batch:
                        log_batch = {"query": batch["query"], "response": responses}
                    self.ppo_trainer.log_stats(stats, log_batch, reward_tensors)

                # Save sample records for inspection
                if samples_file is not None:
                    for query, response, ground_truth, reward in zip(
                        queries, responses, ground_truths, rewards
                    ):
                        if np.random.rand() > self.config.sample_save_rate:
                            continue
                        record = {
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "query": query,
                            "response": response,
                            "ground_truth": ground_truth,
                            "reward": reward,
                        }
                        samples_file.write(json.dumps(record) + "\n")
                    samples_file.flush()

                # Save
                if batch_idx % self.config.save_steps == 0:
                    self.ppo_trainer.save_pretrained(
                        f"{self.config.output_dir}/checkpoint-{epoch}-{batch_idx}"
                    )

        if samples_file is not None:
            samples_file.close()

        # Save final model
        self.ppo_trainer.save_pretrained(self.config.output_dir)
