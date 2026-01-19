"""
Tool-Integrated Reasoning (TIR) Executor

Enables models to execute Python code during generation,
integrating computation results back into the reasoning process.

Based on approaches from:
- Project Numina (AIMO1): SC-TIR with code verification
- NemoSkills (AIMO2): Iterative TIR with quality filtering
"""

import re
import ast
import sys
import traceback
from io import StringIO
from dataclasses import dataclass
from typing import Optional
from contextlib import redirect_stdout, redirect_stderr


@dataclass
class CodeExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    variables: Optional[dict] = None


def safe_execute_python(
    code: str,
    timeout: float = 10.0,
    max_output_length: int = 2000,
    allowed_imports: Optional[set] = None,
) -> CodeExecutionResult:
    """
    Safely execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        max_output_length: Maximum length of captured output
        allowed_imports: Set of allowed module names (None = use defaults)

    Returns:
        CodeExecutionResult with output or error
    """
    if allowed_imports is None:
        allowed_imports = {
            "sympy", "numpy", "math", "fractions", "decimal",
            "itertools", "functools", "collections", "re",
            "statistics", "random", "cmath",
        }

    # Validate code before execution
    validation_error = validate_code(code, allowed_imports)
    if validation_error:
        return CodeExecutionResult(
            success=False,
            output="",
            error=validation_error,
        )

    # Prepare execution environment
    exec_globals = {"__builtins__": create_safe_builtins()}

    # Pre-import allowed modules
    for module_name in allowed_imports:
        try:
            exec_globals[module_name] = __import__(module_name)
        except ImportError:
            pass

    # Add common sympy imports for convenience
    try:
        import sympy
        exec_globals.update({
            "symbols": sympy.symbols,
            "Symbol": sympy.Symbol,
            "solve": sympy.solve,
            "simplify": sympy.simplify,
            "expand": sympy.expand,
            "factor": sympy.factor,
            "sqrt": sympy.sqrt,
            "Rational": sympy.Rational,
            "pi": sympy.pi,
            "E": sympy.E,
            "I": sympy.I,
            "oo": sympy.oo,
            "sin": sympy.sin,
            "cos": sympy.cos,
            "tan": sympy.tan,
            "log": sympy.log,
            "exp": sympy.exp,
            "diff": sympy.diff,
            "integrate": sympy.integrate,
            "limit": sympy.limit,
            "series": sympy.series,
            "Matrix": sympy.Matrix,
            "gcd": sympy.gcd,
            "lcm": sympy.lcm,
            "factorial": sympy.factorial,
            "binomial": sympy.binomial,
            "Sum": sympy.Sum,
            "Product": sympy.Product,
        })
    except ImportError:
        pass

    # Capture stdout
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    try:
        # Execute with timeout
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        # Set timeout (Unix only)
        old_handler = None
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_globals)

            output = stdout_capture.getvalue()
            if len(output) > max_output_length:
                output = output[:max_output_length] + "\n... (truncated)"

            # Extract final variables for inspection
            variables = {
                k: v for k, v in exec_globals.items()
                if not k.startswith('_') and k not in allowed_imports
            }

            return CodeExecutionResult(
                success=True,
                output=output,
                variables=variables,
            )

        finally:
            if old_handler is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    except TimeoutError as e:
        return CodeExecutionResult(
            success=False,
            output="",
            error=str(e),
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return CodeExecutionResult(
            success=False,
            output=stdout_capture.getvalue(),
            error=error_msg,
        )


def validate_code(code: str, allowed_imports: set) -> Optional[str]:
    """
    Validate code for safety before execution.

    Checks for:
    - Disallowed imports
    - Dangerous built-ins
    - File/network operations

    Returns error message if validation fails, None if safe.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    dangerous_patterns = {
        "eval", "exec", "compile", "open", "file",
        "__import__", "globals", "locals", "vars",
        "getattr", "setattr", "delattr",
        "os", "sys", "subprocess", "shutil",
        "socket", "urllib", "requests", "http",
    }

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names if isinstance(node, ast.Import) else [ast.alias(name=node.module, asname=None)]:
                module_name = alias.name.split('.')[0] if alias.name else ""
                if module_name and module_name not in allowed_imports:
                    return f"Import not allowed: {module_name}"

        # Check function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in dangerous_patterns:
                    return f"Dangerous function call: {node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in dangerous_patterns:
                    return f"Dangerous attribute access: {node.func.attr}"

    return None


def create_safe_builtins() -> dict:
    """Create a restricted set of built-in functions."""
    safe_builtins = {
        # Math and logic
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "complex": complex,
        "divmod": divmod,
        "float": float,
        "hex": hex,
        "int": int,
        "len": len,
        "max": max,
        "min": min,
        "oct": oct,
        "pow": pow,
        "round": round,
        "sum": sum,
        # Collections
        "dict": dict,
        "frozenset": frozenset,
        "list": list,
        "range": range,
        "set": set,
        "tuple": tuple,
        # String
        "chr": chr,
        "ord": ord,
        "str": str,
        # Iteration
        "enumerate": enumerate,
        "filter": filter,
        "iter": iter,
        "map": map,
        "next": next,
        "reversed": reversed,
        "sorted": sorted,
        "zip": zip,
        # Type checking
        "isinstance": isinstance,
        "type": type,
        # Output
        "print": print,
        "format": format,
        "repr": repr,
        # Boolean
        "True": True,
        "False": False,
        "None": None,
    }
    return safe_builtins


class TIRExecutor:
    """
    Executor for Tool-Integrated Reasoning.

    Handles the interaction loop between model generation and code execution,
    allowing the model to generate code, see results, and continue reasoning.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_code_executions: int = 5,
        max_tokens_per_step: int = 1024,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_code_executions = max_code_executions
        self.max_tokens_per_step = max_tokens_per_step

    def extract_code_blocks(self, text: str) -> list[tuple[int, int, str]]:
        """Extract Python code blocks with their positions."""
        pattern = r'```python\n(.*?)```'
        blocks = []
        for match in re.finditer(pattern, text, re.DOTALL):
            blocks.append((match.start(), match.end(), match.group(1)))
        return blocks

    def solve_with_tir(self, problem: str) -> str:
        """
        Solve a math problem using Tool-Integrated Reasoning.

        The model generates text and code, code is executed, and
        results are fed back for continued reasoning.
        """
        # Initial prompt
        system_prompt = """You are a mathematical reasoning assistant that can use Python for computation.

When solving problems:
1. Think through the problem step by step
2. When calculation is needed, write Python code in ```python blocks
3. After code, the output will be shown in ```output blocks
4. Use sympy for symbolic math
5. Put your final answer in \\boxed{}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem + "\n\nPut your final answer in \\boxed{}."},
        ]

        full_response = ""
        code_executions = 0

        while code_executions < self.max_code_executions:
            # Generate next part
            prompt = self.tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": full_response}],
                tokenize=False,
                add_generation_prompt=not full_response,
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens_per_step,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            new_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            full_response += new_text

            # Check for code blocks to execute
            code_blocks = self.extract_code_blocks(new_text)

            if not code_blocks:
                # No code to execute, check if we have a final answer
                if "\\boxed" in full_response:
                    break
                continue

            # Execute the last code block
            _, _, code = code_blocks[-1]
            result = safe_execute_python(code)

            code_executions += 1

            # Append execution result
            if result.success:
                full_response += f"\n```output\n{result.output}```\n"
            else:
                full_response += f"\n```output\nError: {result.error}```\n"

            # Check for final answer after code execution
            if "\\boxed" in full_response:
                break

        return full_response
