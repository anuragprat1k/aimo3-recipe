"""Training modules for AIMO3 recipe."""

from aimo3_recipe.training.sft_cot import SFTCoTTrainer
from aimo3_recipe.training.sft_tir import SFTTIRTrainer
from aimo3_recipe.training.rl_math import RLMathTrainer

__all__ = ["SFTCoTTrainer", "SFTTIRTrainer", "RLMathTrainer"]
