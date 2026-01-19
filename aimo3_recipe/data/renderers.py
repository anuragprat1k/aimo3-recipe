"""
Chat renderers for converting math problems to model input format.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MathChatRenderer:
    """
    Renderer for converting math problems to chat format.

    Supports different system prompts for CoT vs TIR training.
    """

    system_prompt: Optional[str] = None
    add_boxed_instruction: bool = True

    # Default system prompts
    COT_SYSTEM_PROMPT = """You are a mathematical reasoning expert. Solve problems step-by-step, showing your work clearly. Always put your final answer in \\boxed{}."""

    TIR_SYSTEM_PROMPT = """You are a mathematical reasoning assistant that can use Python for computation.

When solving problems:
1. Think through the problem step by step
2. When calculation is needed, write Python code in ```python blocks
3. After code execution, show the output in ```output blocks
4. Use sympy for symbolic math, numpy for numerical computation
5. Always verify your answer and put the final answer in \\boxed{}

Example format:
Let me solve this step by step.

First, I'll set up the equation...

```python
from sympy import symbols, solve
x = symbols('x')
equation = x**2 - 5*x + 6
solutions = solve(equation, x)
print(solutions)
```
```output
[2, 3]
```

The solutions are x = 2 and x = 3.

Therefore, the answer is \\boxed{2, 3}."""

    def render_cot(
        self,
        problem: str,
        solution: Optional[str] = None,
        tokenizer=None,
    ) -> str:
        """
        Render problem (and optionally solution) for CoT format.

        Args:
            problem: Math problem text
            solution: Optional solution for training
            tokenizer: Tokenizer with apply_chat_template method

        Returns:
            Formatted chat string
        """
        system = self.system_prompt or self.COT_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": self._format_problem(problem)},
        ]

        if solution:
            messages.append({"role": "assistant", "content": solution})

        if tokenizer:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=(solution is None),
            )

        # Fallback simple format
        return self._simple_format(messages, add_generation_prompt=(solution is None))

    def render_tir(
        self,
        problem: str,
        solution: Optional[str] = None,
        tokenizer=None,
    ) -> str:
        """
        Render problem for Tool-Integrated Reasoning format.

        Args:
            problem: Math problem text
            solution: Optional TIR solution with code blocks
            tokenizer: Tokenizer with apply_chat_template method

        Returns:
            Formatted chat string
        """
        system = self.system_prompt or self.TIR_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": self._format_problem(problem)},
        ]

        if solution:
            messages.append({"role": "assistant", "content": solution})

        if tokenizer:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=(solution is None),
            )

        return self._simple_format(messages, add_generation_prompt=(solution is None))

    def _format_problem(self, problem: str) -> str:
        """Format problem text with optional instructions."""
        text = problem.strip()

        if self.add_boxed_instruction and "\\boxed" not in text.lower():
            text += "\n\nPut your final answer in \\boxed{}."

        return text

    def _simple_format(self, messages: list[dict], add_generation_prompt: bool = False) -> str:
        """Simple fallback formatting without tokenizer."""
        parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")

        text = "\n".join(parts)

        if add_generation_prompt:
            text += "\n<|assistant|>\n"

        return text


def render_for_evaluation(problem: str, model_type: str = "qwen") -> str:
    """
    Render problem for evaluation (inference only).

    Args:
        problem: Math problem text
        model_type: Model family ("qwen", "llama", "deepseek")

    Returns:
        Formatted prompt for generation
    """
    system = MathChatRenderer.TIR_SYSTEM_PROMPT

    if model_type == "qwen":
        return f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{problem}

Put your final answer in \\boxed{{}}.<|im_end|>
<|im_start|>assistant
"""
    elif model_type == "llama":
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{problem}

Put your final answer in \\boxed{{}}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    elif model_type == "deepseek":
        return f"""<|begin▁of▁sentence|>User: {problem}

Put your final answer in \\boxed{{}}.