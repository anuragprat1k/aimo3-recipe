"""
Supervised Fine-Tuning with Tool-Integrated Reasoning (TIR)

This module implements Stage 2 of the AIMO training pipeline:
Training the model to generate code-based solutions that integrate
Python execution for mathematical computation.

Based on approaches from Project Numina (AIMO1) and NemoSkills (AIMO2).
"""

from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import re


@dataclass
class SFTTIRConfig:
    """Configuration for Tool-Integrated Reasoning SFT training."""

    # Model (typically start from CoT fine-tuned model)
    model_name_or_path: str = "./outputs/sft_cot"
    trust_remote_code: bool = True
    use_flash_attention: bool = True

    # LoRA
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Quantization
    load_in_4bit: bool = True

    # Training
    output_dir: str = "./outputs/sft_tir"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 8192  # Longer for code generation

    # Logging
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    report_to: str = "wandb"
    run_name: Optional[str] = None

    # TIR specific
    code_start_token: str = "```python"
    code_end_token: str = "```"
    output_start_token: str = "```output"
    max_code_executions: int = 5


# TIR format markers
TIR_SYSTEM_PROMPT = """You are a mathematical reasoning assistant. When solving problems:
1. Think through the problem step by step
2. When computation is needed, write Python code in ```python blocks
3. After code, show expected output in ```output blocks
4. Use sympy for symbolic math, numpy for numerical computation
5. Always verify your answer and put the final answer in \\boxed{}"""


class SFTTIRTrainer:
    """
    Trainer for Tool-Integrated Reasoning supervised fine-tuning.

    This implements Stage 2 of the AIMO training pipeline, teaching the model
    to integrate Python code execution with mathematical reasoning.
    """

    def __init__(self, config: SFTTIRConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self) -> None:
        """Initialize model and tokenizer."""

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        bnb_config = None
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Load model
        attn_implementation = "flash_attention_2" if self.config.use_flash_attention else "eager"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map="auto",
        )

        if self.config.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

    def format_tir_solution(self, problem: str, solution: str) -> str:
        """
        Format a problem and solution into TIR training format.

        TIR solutions interleave reasoning, code, and outputs:
        - Natural language reasoning
        - ```python code blocks for computation
        - ```output blocks showing execution results
        - Final \\boxed{} answer
        """
        messages = [
            {"role": "system", "content": TIR_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare TIR dataset for training."""

        def format_and_tokenize(examples):
            texts = []
            for problem, solution in zip(examples["problem"], examples["solution"]):
                text = self.format_tir_solution(problem, solution)
                texts.append(text)

            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        dataset = dataset.map(
            format_and_tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing TIR dataset",
        )
        return dataset

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> None:
        """Run TIR training."""

        if self.model is None:
            self.setup_model()

        train_dataset = self.prepare_dataset(train_dataset)
        if eval_dataset is not None:
            eval_dataset = self.prepare_dataset(eval_dataset)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            save_total_limit=self.config.save_total_limit,
            report_to=self.config.report_to,
            run_name=self.config.run_name,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        self.trainer.train()
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from TIR-formatted text."""
    pattern = r"```python\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


def format_code_output(code: str, output: str) -> str:
    """Format code execution result for inclusion in generation."""
    return f"```python\n{code}```\n```output\n{output}\n```"
