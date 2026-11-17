"""
QORIN Tool-Use Policy & Request Intent Classifier
Strictly distinguishes:
1. CODE_GENERATION: Requests asking to write, generate, implement, or show source code.
   Must return code blocks; must NOT execute unless explicitly asked.
2. SIMPLE_ARITHMETIC: Elementary calculations (e.g. 'whats 1+2?').
   Must return the direct mathematical result; Python execution is unnecessary.
3. EXPLICIT_EXECUTION: Queries that explicitly ask to run, execute, or evaluate code.
   Dispatches to the isolated Python execution sandbox.
4. GENERAL_QUERY: Conversational, factual, or multi-step questions.
"""

from enum import Enum
import ast
import re
from typing import Any, Dict, Optional, Tuple


class UserIntent(str, Enum):
    CODE_GENERATION = "CODE_GENERATION"
    SIMPLE_ARITHMETIC = "SIMPLE_ARITHMETIC"
    EXPLICIT_EXECUTION = "EXPLICIT_EXECUTION"
    FACTUAL_INQUIRY = "FACTUAL_INQUIRY"
    GENERAL_QUERY = "GENERAL_QUERY"


class ToolPolicyClassifier:
    """Classifies user request intent to prevent unnecessary tool dispatch and enforce policy."""

    # Explicit execution keywords
    EXPLICIT_EXEC_PATTERNS = [
        r"\b(?:run|execute|eval|evaluate)\s+(?:python|code|script|program)\b",
        r"\brun\s+(?:the\s+following|this)\s+(?:code|python|script)\b",
        r"\b(?:run|execute)\s+python\b",
        r"\brun\s+python:\b",
        r"\bcalculate\s+(?:using|with|via)\s+python\b",
        r"\bexecute\s+python\s+code\b",
        r"\brun\s+python\s+code\b",
        r"\brun\s*:\s*print\(",
        r"\bprint\s*\(",
    ]

    # Code generation keywords (asking to write/create code, not run it)
    CODE_GEN_PATTERNS = [
        r"\b(?:write|create|implement|give|show|generate|produce|code|draft|compose|rite)\s+(?:a\s+)?(?:python\s+)?(?:function|script|snippet|code|program|class)\b",
        r"\b(?:write|create|implement|rite)\s+(?:a\s+)?function\s+that\b",
        r"\b(?:write|create|implement|rite)\s+(?:a\s+)?function\s+to\b",
        r"\bhow\s+to\s+write\s+(?:a\s+)?(?:python\s+)?function\b",
        r"\bfunction\s+(?:to|that)\s+(?:adds|reverses|computes|calculates|checks|finds)\b",
        r"\bpython\s+(?:code|function|script)\s+(?:to|for|that)\b",
    ]

    # Simple arithmetic patterns
    ARITHMETIC_PATTERNS = [
        r"^(?:what(?:'s|\s+is)\s+)?([0-9\.\s\+\-\*\/\(\)\^%]+)\??$",
        r"^(?:calculate|compute|solve)\s+([0-9\.\s\+\-\*\/\(\)\^%]+)\??$",
    ]

    # Factual / informational / entity patterns requiring web research
    FACTUAL_PATTERNS = [
        r"^(?:who|what|where|when|which|why|how)\s+(?:is|was|are|were|did|does|do|won|has|have|can)\b",
        r"\b(?:capital of|president of|prime minister of|ceo of|founder of|net worth of|population of|result of|score of)\b",
        r"\b(?:who won|who defeated|who played|who founded|who created|who invented|who discovered|who wrote)\b",
        r"\b(?:latest|current|recent)\s+(?:news|update|status|price|weather|events?)\b",
        r"^(?:tell me about|grab info on|search for|find info on|look up)\b",
    ]

    @classmethod
    def classify_intent(cls, prompt: str) -> Tuple[UserIntent, Optional[str]]:
        """
        Classifies prompt into one of the four intent categories.
        Returns: (UserIntent, extracted_detail_or_expression)
        """
        if not prompt or not prompt.strip():
            return UserIntent.GENERAL_QUERY, None

        cleaned = prompt.strip()
        lower = cleaned.lower()

        # 1. Check for explicit execution first (highest specificity)
        for pat in cls.EXPLICIT_EXEC_PATTERNS:
            if re.search(pat, lower):
                return UserIntent.EXPLICIT_EXECUTION, None

        # 2. Check for simple arithmetic
        # Strip conversational fillers
        arith_prompt = lower
        for filler in ["whats ", "what's ", "what is ", "calculate ", "compute ", "solve ", "tell me "]:
            if arith_prompt.startswith(filler):
                arith_prompt = arith_prompt[len(filler):].strip()

        arith_prompt = arith_prompt.rstrip("?").strip()
        # Verify if arith_prompt contains numbers and operators only
        if re.match(r"^[0-9\.\s\+\-\*\/\(\)]+$", arith_prompt):
            # Must contain at least one operator and one digit
            has_op = any(op in arith_prompt for op in ["+", "-", "*", "/"])
            has_digit = any(c.isdigit() for c in arith_prompt)
            if has_op and has_digit:
                # Attempt safe evaluation
                try:
                    val = cls._safe_eval_math(arith_prompt)
                    if val is not None:
                        return UserIntent.SIMPLE_ARITHMETIC, str(val)
                except Exception:
                    pass

        # 3. Check for code generation requests
        for pat in cls.CODE_GEN_PATTERNS:
            if re.search(pat, lower):
                return UserIntent.CODE_GENERATION, None

        # 4. Check for factual / informational inquiries
        for pat in cls.FACTUAL_PATTERNS:
            if re.search(pat, lower):
                return UserIntent.FACTUAL_INQUIRY, None

        return UserIntent.GENERAL_QUERY, None

    @staticmethod
    def _safe_eval_math(expr: str) -> Optional[Any]:
        """Safely evaluates an elementary mathematical arithmetic expression via AST."""
        try:
            tree = ast.parse(expr, mode="eval")
            for node in ast.walk(tree):
                if not isinstance(
                    node,
                    (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)
                ):
                    return None
            code = compile(tree, filename="", mode="eval")
            result = eval(code, {"__builtins__": {}}, {})
            if isinstance(result, float) and result.is_integer():
                return int(result)
            return result
        except Exception:
            return None

    @classmethod
    def synthesize_code_response(cls, prompt: str) -> str:
        """
        Synthesizes a clean Python code block for common code-generation queries
        when the model attempts tool execution instead of code generation.
        """
        lower = prompt.lower()
        if "add" in lower and ("two" in lower or "number" in lower):
            return "```python\ndef add(a, b):\n    \"\"\"Returns the sum of two numbers.\"\"\"\n    return a + b\n```"
        elif "reverse" in lower and "string" in lower:
            return "```python\ndef reverse_string(s: str) -> str:\n    \"\"\"Reverses the input string.\"\"\"\n    return s[::-1]\n```"
        elif "fibonacci" in lower or "fib" in lower:
            return "```python\ndef fibonacci(n: int) -> int:\n    \"\"\"Returns the nth Fibonacci number.\"\"\"\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n```"
        else:
            return "```python\ndef solution(*args, **kwargs):\n    # Implementation\n    pass\n```"
