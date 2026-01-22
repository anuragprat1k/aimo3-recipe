"""
Supervised Fine-Tuning with Chain-of-Thought for Math Reasoning

This module implements Stage 1 of the AIMO training pipeline:
Training on math problems with text-based Chain-of-Thought reasoning.
Based on the approach from Project Numina (AIMO1 winner).
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import hashlib
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_from_disk
import wandb


@dataclass
class SFTCoTConfig:
    """Configuration for Chain-of-Thought SFT training."""

    # Model
    model_name_or_path: str = "Qwen/Qwen2.5-14B"
    trust_remote_code: bool = True
    use_flash_attention: bool = True

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Training
    output_dir: str = "./outputs/sft_cot"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 4096

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    report_to: str = "wandb"
    run_name: Optional[str] = None

    # Data
    dataset_name: str = "AI-MO/NuminaMath-CoT"
    max_train_samples: Optional[int] = None
    validation_split: float = 0.01
    tokenized_cache_dir: Optional[str] = "./outputs/sft_cot/tokenized_cache"


class SFTCoTTrainer:
    """
    Trainer for Chain-of-Thought supervised fine-tuning on math problems.

    This implements Stage 1 of the AIMO training pipeline, training the model
    to produce step-by-step reasoning for math problems.
    """

    def __init__(self, config: SFTCoTConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self) -> None:
        """Initialize model and tokenizer with optional quantization and LoRA."""

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        use_bnb = self.config.load_in_4bit or self.config.load_in_8bit
        if use_bnb:
            import importlib.util
            has_bnb = importlib.util.find_spec("bitsandbytes") is not None
            has_cuda = torch.cuda.is_available()
            if not has_bnb or not has_cuda:
                print("bitsandbytes/CUDA not available; disabling 4-bit/8-bit loading.")
                self.config.load_in_4bit = False
                self.config.load_in_8bit = False

        # Flash attention fallback (not supported on macOS/MPS or without flash_attn)
        if self.config.use_flash_attention:
            import importlib.util
            has_flash_attn = importlib.util.find_spec("flash_attn") is not None
            if torch.backends.mps.is_available() or not has_flash_attn:
                print("Flash Attention not available; falling back to eager attention.")
                self.config.use_flash_attention = False

        bnb_config = None
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.load_in_8bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

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

        # Prepare for k-bit training if quantized
        if self.config.load_in_4bit or self.config.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
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

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset for training by tokenizing and formatting.

        Expected dataset format:
        - problem: The math problem statement
        - solution: The CoT solution with step-by-step reasoning
        """
        cache_path = self._get_tokenized_cache_path(dataset, purpose="cot")
        if cache_path and cache_path.exists():
            return load_from_disk(str(cache_path))

        def format_and_tokenize(examples):
            texts = []
            for problem, solution in zip(examples["problem"], examples["solution"]):
                # Format as chat conversation
                messages = [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(text)

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
                return_tensors=None,
            )

            # Labels are same as input_ids for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # Apply formatting
        dataset = dataset.map(
            format_and_tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(cache_path))

        return dataset

    def _get_tokenized_cache_path(self, dataset: Dataset, purpose: str) -> Optional[Path]:
        if not self.config.tokenized_cache_dir:
            return None
        tokenizer_id = getattr(self.tokenizer, "name_or_path", "unknown-tokenizer")
        payload = {
            "dataset_fingerprint": getattr(dataset, "_fingerprint", "unknown-dataset"),
            "tokenizer": tokenizer_id,
            "max_seq_length": self.config.max_seq_length,
            "purpose": purpose,
        }
        cache_id = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        cache_root = Path(self.config.tokenized_cache_dir)
        return cache_root / f"sft_cot_{cache_id}"

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> None:
        """Run training."""

        if self.model is None:
            self.setup_model()

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_dataset)
        if eval_dataset is not None:
            eval_dataset = self.prepare_dataset(eval_dataset)

        # Training arguments
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
            remove_unused_columns=False,
            dataloader_pin_memory=True,
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8,
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        self.trainer.train()

        # Save final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)

    def save_merged_model(self, output_path: str) -> None:
        """Merge LoRA weights and save full model."""
        if self.config.use_lora:
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
        else:
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
