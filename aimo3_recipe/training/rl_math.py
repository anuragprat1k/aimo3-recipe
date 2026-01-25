"""
Reinforcement Learning for Mathematical Reasoning

This module implements RL-based training for math problem solving,
using correctness rewards based on answer verification.

Uses GRPO (Group Relative Policy Optimization) for training.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import os
import json
import random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
import numpy as np
from tqdm import tqdm

# PyTorch 2.6+ compatibility: register numpy globals for checkpoint loading
# This must happen at module level before any checkpoint loading occurs
import numpy.core.multiarray
torch.serialization.add_safe_globals([
    numpy.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
])


@dataclass
class RLMathConfig:
    """Configuration for RL math training."""

    # Model
    model_name_or_path: str = "./outputs/sft_tir"
    trust_remote_code: bool = True
    use_flash_attention: bool = False  # Disabled by default
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

    # LLM Judge (optional - for process-based rewards)
    use_llm_judge: bool = False
    llm_judge_model: Optional[str] = None  # None = use policy model as judge
    llm_judge_weight: float = 0.5  # Weight for judge score vs correctness
    judge_correctness_weight: float = 0.4
    judge_consistency_weight: float = 0.4
    judge_clarity_weight: float = 0.2

    # KL
    beta: float = 0.1  # KL penalty coefficient

    # Training
    num_train_epochs: int = 1
    save_steps: int = 100
    logging_steps: int = 5
    report_to: str = "tensorboard"
    run_name: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    save_samples: bool = False
    sample_save_rate: float = 0.01
    samples_filename: str = "samples.jsonl"

    # vLLM (for faster generation)
    use_vllm: bool = True
    vllm_mode: str = "colocate"  # "colocate" runs in-process, "server" requires separate vLLM server
    vllm_gpu_memory_utilization: float = 0.5  # GPU memory for vLLM when colocated

    # Evaluation
    eval_steps: int = 100
    eval_samples: int = 50  # Number of eval samples to use per evaluation
    eval_temperature: float = 0.1  # Lower temperature for more deterministic eval
    eval_batch_size: int = 8  # Batch size for evaluation (higher = faster)


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
            has_boxed = r"\boxed{" in completion
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


class LLMJudgeRewardFunction:
    """
    LLM-based reward function that evaluates solution quality step-by-step.

    Evaluates:
    1. Logical consistency between steps
    2. Mathematical correctness of operations
    3. Clarity and neatness of presentation

    Can use either a separate judge model or the policy model itself.
    """

    __name__ = "llm_judge_reward"

    def __init__(
        self,
        answer_verifier: Callable[[str, str], bool],
        judge_model: Optional[AutoModelForCausalLM] = None,
        judge_tokenizer: Optional[AutoTokenizer] = None,
        correctness_weight: float = 0.4,
        consistency_weight: float = 0.4,
        clarity_weight: float = 0.2,
        llm_judge_weight: float = 0.5,
        correct_reward: float = 1.0,
        incorrect_reward: float = -0.5,
        format_reward: float = 0.1,
    ):
        """
        Initialize LLM Judge reward function.

        Args:
            answer_verifier: Function to verify final answer correctness
            judge_model: Model to use as judge (None = will use policy model)
            judge_tokenizer: Tokenizer for judge model
            correctness_weight: Weight for mathematical correctness score
            consistency_weight: Weight for logical consistency score
            clarity_weight: Weight for clarity/neatness score
            llm_judge_weight: Weight for LLM judge vs answer correctness (0-1)
            correct_reward: Reward for correct final answer
            incorrect_reward: Reward for incorrect final answer
            format_reward: Bonus for proper \\boxed{} format
        """
        self.answer_verifier = answer_verifier
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        self.correctness_weight = correctness_weight
        self.consistency_weight = consistency_weight
        self.clarity_weight = clarity_weight
        self.llm_judge_weight = llm_judge_weight
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.format_reward = format_reward
        self._policy_model = None  # Set by trainer if using policy as judge

    def set_policy_model(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """Set the policy model to use as judge (called by trainer)."""
        self._policy_model = model
        if self.judge_tokenizer is None:
            self.judge_tokenizer = tokenizer

    def _get_judge_model(self):
        """Get the model to use for judging."""
        if self.judge_model is not None:
            return self.judge_model
        if self._policy_model is not None:
            return self._policy_model
        raise ValueError("No judge model available. Either provide judge_model or call set_policy_model().")

    def _build_judge_prompt(self, problem: str, solution: str) -> str:
        """Build the prompt for the LLM judge."""
        return f"""You are a mathematics teacher evaluating a student's solution. Analyze the solution step-by-step and rate it on three criteria.

## Problem
{problem}

## Student's Solution
{solution}

## Evaluation Criteria
Rate each criterion from 0 to 10:

1. **LOGICAL_CONSISTENCY**: Are the reasoning steps logically connected? Does each step follow from the previous? Is the overall argument structure sound?

2. **MATHEMATICAL_CORRECTNESS**: Are the calculations accurate? Are the mathematical operations and transformations valid? Are formulas applied correctly?

3. **CLARITY**: Is the solution well-organized? Are steps clearly explained? Is it easy to follow the reasoning?

## Your Response
Provide your scores in exactly this format:
LOGICAL_CONSISTENCY: <score>
MATHEMATICAL_CORRECTNESS: <score>
CLARITY: <score>
BRIEF_FEEDBACK: <one sentence explaining the main strength or weakness>"""

    def _parse_judge_response(self, response: str) -> dict[str, float]:
        """Parse scores from judge response."""
        scores = {
            "consistency": 5.0,
            "correctness": 5.0,
            "clarity": 5.0,
        }

        for line in response.split("\n"):
            line = line.strip()
            try:
                if line.startswith("LOGICAL_CONSISTENCY:"):
                    score_str = line.split(":")[1].strip().split()[0]
                    scores["consistency"] = min(10.0, max(0.0, float(score_str)))
                elif line.startswith("MATHEMATICAL_CORRECTNESS:"):
                    score_str = line.split(":")[1].strip().split()[0]
                    scores["correctness"] = min(10.0, max(0.0, float(score_str)))
                elif line.startswith("CLARITY:"):
                    score_str = line.split(":")[1].strip().split()[0]
                    scores["clarity"] = min(10.0, max(0.0, float(score_str)))
            except (IndexError, ValueError):
                continue

        return scores

    def _extract_problem_from_prompt(self, prompt: str) -> str:
        """Extract the problem text from a chat-formatted prompt."""
        # Try to extract content after common patterns
        markers = ["<|user|>", "[INST]", "Problem:", "Question:", "<|im_start|>user"]
        for marker in markers:
            if marker in prompt:
                parts = prompt.split(marker)
                if len(parts) > 1:
                    # Get content after marker, before any assistant marker
                    content = parts[-1]
                    for end_marker in ["<|assistant|>", "[/INST]", "<|im_end|>", "<|im_start|>assistant"]:
                        if end_marker in content:
                            content = content.split(end_marker)[0]
                    return content.strip()
        return prompt  # Return as-is if no markers found

    @torch.no_grad()
    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        answer: list[str],
        **kwargs,
    ) -> list[float]:
        """
        Compute rewards combining LLM judge scores with answer correctness.

        Args:
            prompts: List of input prompts
            completions: List of model completions
            answer: List of ground truth answers

        Returns:
            List of reward values
        """
        rewards = []
        judge_model = self._get_judge_model()
        judge_model.eval()

        for prompt, completion, ground_truth in zip(prompts, completions, answer):
            # 1. Compute answer-based reward (same as MathRewardFunction)
            answer_reward = 0.0
            has_boxed = r"\boxed{" in completion
            if has_boxed:
                answer_reward += self.format_reward

            is_correct = self.answer_verifier(completion, ground_truth)
            if is_correct:
                answer_reward += self.correct_reward
            else:
                answer_reward += self.incorrect_reward

            # 2. Compute LLM judge reward
            problem = self._extract_problem_from_prompt(prompt)
            judge_prompt = self._build_judge_prompt(problem, completion)

            inputs = self.judge_tokenizer(
                judge_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(judge_model.device)

            outputs = judge_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.judge_tokenizer.pad_token_id,
            )

            response = self.judge_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            scores = self._parse_judge_response(response)

            # Weighted combination of judge scores (0-1 scale)
            judge_score = (
                self.correctness_weight * (scores["correctness"] / 10) +
                self.consistency_weight * (scores["consistency"] / 10) +
                self.clarity_weight * (scores["clarity"] / 10)
            )

            # Scale judge score to similar range as answer reward
            # Judge score is 0-1, scale to roughly -0.5 to 1.0
            judge_reward = (judge_score * 1.5) - 0.5

            # 3. Combine answer reward and judge reward
            final_reward = (
                (1 - self.llm_judge_weight) * answer_reward +
                self.llm_judge_weight * judge_reward
            )

            rewards.append(final_reward)

        return rewards


class SampleSavingCallback(TrainerCallback):
    """Callback to save a sample of GRPO rollouts to disk."""

    def __init__(
        self,
        output_path: str,
        sample_rate: float = 0.01,
        reward_fn: Optional[MathRewardFunction] = None,
    ):
        self.output_path = Path(output_path)
        self.sample_rate = sample_rate
        self.reward_fn = reward_fn
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear file at start
        self.output_path.write_text("")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Save samples when completions are logged."""
        if logs is None:
            return

        # TRL logs completions under various keys depending on version
        completions = logs.get("completions") or logs.get("generated_texts")
        if not completions:
            return

        prompts = logs.get("prompts", [])
        rewards = logs.get("rewards", [])

        # Handle case where rewards might be a tensor
        if hasattr(rewards, "tolist"):
            rewards = rewards.tolist()
        elif not isinstance(rewards, list):
            rewards = []

        with open(self.output_path, "a") as f:
            for i, completion in enumerate(completions):
                if random.random() > self.sample_rate:
                    continue

                sample = {
                    "step": state.global_step,
                    "prompt": prompts[i] if i < len(prompts) else "",
                    "completion": completion,
                    "reward": rewards[i] if i < len(rewards) else None,
                }
                f.write(json.dumps(sample) + "\n")


class SampleTrackingRewardFunction:
    """Wrapper around reward function that tracks samples for saving."""

    __name__ = "sample_tracking_reward"

    def __init__(
        self,
        base_reward_fn: MathRewardFunction,
        output_path: str,
        sample_rate: float = 0.01,
        log_to_wandb: bool = True,
    ):
        self.base_reward_fn = base_reward_fn
        self.output_path = Path(output_path)
        self.sample_rate = sample_rate
        self.log_to_wandb = log_to_wandb
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear file at start
        self.output_path.write_text("")
        self._step = 0
        self._wandb = None
        if self.log_to_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                print("wandb not installed, skipping wandb logging for samples")
                self.log_to_wandb = False

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        answer: list[str],
        **kwargs,
    ) -> list[float]:
        """Compute rewards and save samples."""
        rewards = self.base_reward_fn(prompts, completions, answer, **kwargs)

        # Collect samples for this batch
        samples_to_save = []
        for i, (prompt, completion, ground_truth, reward) in enumerate(
            zip(prompts, completions, answer, rewards)
        ):
            if random.random() > self.sample_rate:
                continue

            is_correct = self.base_reward_fn.answer_verifier(completion, ground_truth)
            sample = {
                "step": self._step,
                "prompt": prompt,
                "completion": completion,
                "ground_truth": ground_truth,
                "reward": reward,
                "is_correct": is_correct,
            }
            samples_to_save.append(sample)

        # Save to disk
        if samples_to_save:
            with open(self.output_path, "a") as f:
                for sample in samples_to_save:
                    f.write(json.dumps(sample) + "\n")

            # Log to wandb
            if self.log_to_wandb and self._wandb and self._wandb.run is not None:
                # Log as a table for better visualization
                table_data = []
                for sample in samples_to_save:
                    table_data.append([
                        sample["step"],
                        sample["prompt"][:500],  # Truncate for display
                        sample["completion"][:2000],  # Truncate for display
                        sample["ground_truth"],
                        sample["reward"],
                        sample["is_correct"],
                    ])
                table = self._wandb.Table(
                    columns=["step", "prompt", "completion", "ground_truth", "reward", "is_correct"],
                    data=table_data,
                )

                # Use commit=False to avoid step conflicts with trainer's logging
                num_correct = sum(1 for s in samples_to_save if s["is_correct"])
                self._wandb.log({
                    "samples": table,
                    "samples/batch_accuracy": num_correct / len(samples_to_save) if samples_to_save else 0,
                    "samples/batch_mean_reward": np.mean([s["reward"] for s in samples_to_save]),
                    "samples/batch_size": len(samples_to_save),
                }, commit=False)

        self._step += 1
        return rewards


class MetricsLoggingCallback(TrainerCallback):
    """Callback to print training metrics to console."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Print metrics when trainer logs them."""
        if not state.is_world_process_zero:
            return

        if logs is None:
            return

        # Format step info
        step_info = f"Step {state.global_step}"
        if "epoch" in logs:
            step_info += f" (Epoch {logs['epoch']:.2f})"

        # Filter and format metrics
        metrics_str = ", ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in sorted(logs.items())
            if k not in ("epoch", "total_flos", "train_runtime", "train_samples_per_second", "train_steps_per_second")
        )

        if metrics_str:
            print(f"[Train] {step_info} - {metrics_str}", flush=True)


class EvalCallback(TrainerCallback):
    """Callback to evaluate model accuracy on held-out problems during RL training."""

    def __init__(
        self,
        eval_dataset: Dataset,
        tokenizer: AutoTokenizer,
        answer_verifier: Callable[[str, str], bool],
        eval_steps: int = 100,
        eval_samples: int = 50,
        max_completion_length: int = 2048,
        temperature: float = 0.1,
        report_to: str = "wandb",
        eval_batch_size: int = 8,
        output_dir: str = None,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.answer_verifier = answer_verifier
        self.eval_steps = eval_steps
        self.eval_samples = min(eval_samples, len(eval_dataset))
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.report_to = report_to
        self.eval_batch_size = eval_batch_size
        self._wandb = None
        self._tb_writer = None
        if self.report_to == "wandb":
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                pass
        elif self.report_to == "tensorboard" and output_dir:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=output_dir)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run evaluation every eval_steps."""
        # Only run on main process in distributed training
        if not state.is_world_process_zero:
            return

        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return

        if model is None:
            return

        print(f"\n[Eval] Running evaluation at step {state.global_step}...", flush=True)
        metrics = self._run_evaluation(model, state.global_step)

        # Log metrics
        print(f"[Eval] Step {state.global_step}: {metrics}", flush=True)
        if self._wandb and self._wandb.run is not None:
            self._wandb.log(metrics, step=state.global_step)
        elif self._tb_writer is not None:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(key, value, state.global_step)
            self._tb_writer.flush()

    def _run_evaluation(self, model, step: int) -> dict:
        """Generate on eval samples and compute accuracy using batched generation."""
        model.eval()

        # Sample subset of eval data
        indices = random.sample(range(len(self.eval_dataset)), self.eval_samples)
        eval_subset = self.eval_dataset.select(indices)

        correct = 0
        total = 0
        has_boxed = 0
        total_length = 0
        eval_results = []

        # Collect all prompts and ground truths
        all_prompts = [ex["prompt"] for ex in eval_subset]
        all_ground_truths = [ex["answer"] for ex in eval_subset]

        # Process in batches
        num_batches = (len(all_prompts) + self.eval_batch_size - 1) // self.eval_batch_size

        # Save original padding side for restoration
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"  # Left-pad for batched generation

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Evaluating", leave=False):
                start_idx = batch_idx * self.eval_batch_size
                end_idx = min(start_idx + self.eval_batch_size, len(all_prompts))
                batch_prompts = all_prompts[start_idx:end_idx]
                batch_ground_truths = all_ground_truths[start_idx:end_idx]

                # Tokenize batch
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    padding=True,
                ).to(model.device)

                # Generate for entire batch
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_completion_length,
                    temperature=self.temperature,
                    top_p=0.95,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                # Decode completions (exclude prompt tokens)
                prompt_lengths = inputs["attention_mask"].sum(dim=1)
                for i, (output, prompt_len, prompt, ground_truth) in enumerate(
                    zip(outputs, prompt_lengths, batch_prompts, batch_ground_truths)
                ):
                    completion = self.tokenizer.decode(
                        output[prompt_len:],
                        skip_special_tokens=True,
                    )

                    # Check correctness
                    is_correct = self.answer_verifier(completion, ground_truth)
                    if is_correct:
                        correct += 1
                    total += 1

                    # Track format compliance
                    if r"\boxed{" in completion:
                        has_boxed += 1

                    total_length += len(completion)

                    eval_results.append({
                        "prompt": prompt[:200],
                        "completion": completion[:500],
                        "ground_truth": ground_truth,
                        "is_correct": is_correct,
                    })

        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side

        accuracy = correct / total if total > 0 else 0
        boxed_rate = has_boxed / total if total > 0 else 0
        avg_length = total_length / total if total > 0 else 0

        metrics = {
            "eval/accuracy": accuracy,
            "eval/boxed_rate": boxed_rate,
            "eval/avg_completion_length": avg_length,
            "eval/num_samples": total,
            "eval/num_correct": correct,
        }

        # Log sample table to wandb
        if self._wandb and self._wandb.run is not None and eval_results:
            table_data = [
                [r["prompt"], r["completion"], r["ground_truth"], r["is_correct"]]
                for r in eval_results[:10]  # Limit to 10 for display
            ]
            table = self._wandb.Table(
                columns=["prompt", "completion", "ground_truth", "is_correct"],
                data=table_data,
            )
            # Use commit=False to avoid step conflicts with trainer's logging
            self._wandb.log({f"eval/samples_step_{step}": table}, commit=False)

        model.train()
        return metrics


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

        # Don't use device_map with CPU or in distributed mode
        # Distributed training (accelerate/torchrun) handles device placement
        is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
        if not self.config.force_cpu and not is_distributed:
            model_kwargs["device_map"] = self.config.device_map

        # Use Flash Attention 2 if enabled and available
        if self.config.use_flash_attention and not self.config.force_cpu:
            model_kwargs["attn_implementation"] = "flash_attention_2"

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

    def train(
        self,
        dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        """Run GRPO training.

        Args:
            dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            resume_from_checkpoint: Optional path to checkpoint to resume from
        """

        if self.model is None:
            self.setup_model()

        # If using LLM judge, set up the judge model reference
        if isinstance(self.reward_fn, LLMJudgeRewardFunction):
            if self.reward_fn.judge_model is None:
                # Use policy model as judge
                print("LLM Judge: Using policy model as judge")
                self.reward_fn.set_policy_model(self.model, self.tokenizer)
            else:
                print(f"LLM Judge: Using separate judge model")

        dataset = self.prepare_dataset(dataset)
        if eval_dataset is not None:
            eval_dataset = self.prepare_dataset(eval_dataset)

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

        # Wrap reward function for sample saving if enabled
        reward_fn = self.reward_fn
        if self.config.save_samples and self.reward_fn is not None:
            samples_path = Path(self.config.output_dir) / self.config.samples_filename
            log_to_wandb = self.config.report_to == "wandb"
            reward_fn = SampleTrackingRewardFunction(
                base_reward_fn=self.reward_fn,
                output_path=str(samples_path),
                sample_rate=self.config.sample_save_rate,
                log_to_wandb=log_to_wandb,
            )
            print(f"Sample saving enabled: {samples_path} (rate: {self.config.sample_save_rate}, wandb: {log_to_wandb})")

        # Setup callbacks
        callbacks = [MetricsLoggingCallback()]

        # Add evaluation callback if eval dataset provided
        if eval_dataset is not None and self.reward_fn is not None:
            eval_callback = EvalCallback(
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                answer_verifier=self.reward_fn.answer_verifier,
                eval_steps=self.config.eval_steps,
                eval_samples=self.config.eval_samples,
                max_completion_length=self.config.max_completion_length,
                temperature=self.config.eval_temperature,
                report_to=self.config.report_to,
                eval_batch_size=self.config.eval_batch_size,
                output_dir=self.config.output_dir,
            )
            callbacks.append(eval_callback)
            print(f"Evaluation enabled: every {self.config.eval_steps} steps on {self.config.eval_samples} samples")

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
            logging_strategy="steps",
            report_to=self.config.report_to,
            use_cpu=self.config.force_cpu,
            log_completions=True,
            # vLLM for faster generation
            use_vllm=self.config.use_vllm and not self.config.force_cpu,
            vllm_mode=self.config.vllm_mode,
            vllm_gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
        )

        # Initialize GRPO trainer
        self.trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=reward_fn,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
            callbacks=callbacks if callbacks else None,
        )

        # Allow numpy globals for checkpoint loading (PyTorch 2.6+ compatibility)
        if resume_from_checkpoint:
            import numpy.core.multiarray
            torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
            # Also add numpy dtypes that may be in checkpoints
            torch.serialization.add_safe_globals([np.ndarray, np.dtype])

        # Train
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model
        self.trainer.save_model(self.config.output_dir)

    def _configure_wandb(self) -> None:
        if self.config.report_to != "wandb":
            return
        if self.config.wandb_project:
            os.environ["WANDB_PROJECT"] = self.config.wandb_project
        if self.config.wandb_name:
            os.environ["WANDB_RUN_NAME"] = self.config.wandb_name
