"""
QORIN Tool-Use Policy & Request Intent Classifier
Strictly categorizes queries into 8 research-governed policy classes:
1. CODE_GENERATION: Requests asking to write, generate, implement, or show source code.
   Must return code blocks; must NOT execute unless explicitly asked.
2. CODE_EXPLANATION: Requests asking to explain or analyze code, functions, or programming concepts.
   Must return instructional text; must NOT execute.
3. CODE_DEBUGGING: Requests asking to identify, troubleshoot, or fix bugs in code.
   Must return analysis and corrected code; must NOT execute unless asked.
4. EXPLICIT_CODE_EXECUTION: Queries that explicitly ask to run, execute, or evaluate code.
   Dispatches to the isolated Python execution sandbox.
5. SIMPLE_ARITHMETIC: Elementary calculations (e.g. 'what is 1 + 1?').
   Must return the direct mathematical result without spawning execution tools.
6. CURRENT_FACTUAL_QUERY: Real-time, external, or entity inquiries requiring live retrieval.
   Dispatches to Web Search provider with evidence grounding.
7. GENERAL_KNOWLEDGE: Broad, timeless, educational, or theoretical inquiries.
   Synthesizes direct parametric responses without unnecessary tool overhead.
8. CONVERSATIONAL_FOLLOW_UP: Dialogue continuations referencing previous turns.
"""

from enum import Enum
import ast
import re
from typing import Any, Dict, Optional, Tuple


class UserIntent(str, Enum):
    CODE_GENERATION = "CODE_GENERATION"
    CODE_EXPLANATION = "CODE_EXPLANATION"
    CODE_DEBUGGING = "CODE_DEBUGGING"
    EXPLICIT_CODE_EXECUTION = "EXPLICIT_CODE_EXECUTION"
    SIMPLE_ARITHMETIC = "SIMPLE_ARITHMETIC"
    CURRENT_FACTUAL_QUERY = "CURRENT_FACTUAL_QUERY"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    CONVERSATIONAL_FOLLOW_UP = "CONVERSATIONAL_FOLLOW_UP"

    # Backward compatibility aliases
    EXPLICIT_EXECUTION = "EXPLICIT_CODE_EXECUTION"
    FACTUAL_INQUIRY = "CURRENT_FACTUAL_QUERY"
    GENERAL_QUERY = "GENERAL_KNOWLEDGE"


class ToolPolicyClassifier:
    """Classifies user request intent to prevent unnecessary tool dispatch and enforce policy."""

    # Explicit execution keywords (user explicitly commands running / executing)
    EXPLICIT_EXEC_PATTERNS = [
        r"\b(?:run|execute|eval|evaluate)\s+(?:this|the\s+following|a|my)?\s*(?:python\s+)?(?:code|script|program)\b",
        r"\brun\s+(?:the\s+following|this)\s+(?:code|python|script)\b",
        r"\b(?:run|execute)\s+python\b",
        r"\brun\s+python:\b",
        r"\bcalculate\s+(?:using|with|via)\s+python\b",
        r"\bexecute\s+python\s+code\b",
        r"\brun\s+python\s+code\b",
        r"\brun\s*:\s*print\(",
        r"\bprint\s*\(",
        r"\brun\s+this\s+python\s+code\s+and\s+show\s+(?:the\s+)?output\b",
        r"\brun\s+and\s+show\s+(?:the\s+)?output\b",
        r"\btest\s+(?:this|the)\s+(?:code|script)\s+by\s+running\b",
    ]

    # Code explanation keywords
    CODE_EXPLANATION_PATTERNS = [
        r"\b(?:explain|describe|walk\s+through|understand)\s+(?:how\s+)?(?:a\s+)?(?:python\s+)?(?:function|code|script|program|class|algorithm|method|decorator|generator|list\s+comprehension|loop|syntax)\b",
        r"\bexplain\s+how\s+(?:a\s+)?python\s+function\s+works\b",
        r"\bhow\s+does\s+(?:this|a)\s+(?:function|python|code|script|algorithm|method)\s+work\b",
        r"\bwhat\s+does\s+(?:this|the\s+following)\s+(?:code|function|snippet|script)\s+do\b",
        r"\bmeaning\s+of\s+.*in\s+python\b",
    ]

    # Code debugging keywords
    CODE_DEBUG_PATTERNS = [
        r"\b(?:debug|fix|troubleshoot|repair|solve\s+error\s+in|find\s+bugs?\s+in)\s+(?:this|the\s+following|my|a)?\s*(?:python\s+)?(?:code|function|script|program|bug|error|traceback)\b",
        r"\bdebug\s+this\s+python\s+code\b",
        r"\bwhy\s+is\s+(?:this|my)\s+(?:code|function|script)\s+(?:failing|throwing|crashing|broken)\b",
        r"\bfix\s+(?:this|the)\s+error\b",
    ]

    # Code generation keywords (asking to write/create code, not run it)
    CODE_GEN_PATTERNS = [
        r"\b(?:write|create|implement|give|show|generate|produce|code|draft|compose|construct)\s+(?:me\s+)?(?:a\s+|an\s+)?(?:basic\s+|simple\s+|clean\s+|sample\s+|example\s+)?(?:python\s+)?(?:function|script|snippet|code|program|class|example)\b",
        r"\b(?:write|create|implement)\s+(?:a\s+|an\s+)?(?:basic\s+|simple\s+)?(?:python\s+)?(?:function|code|program)\s+(?:to|that|for|of)\b",
        r"\b(?:python\s+code|code)\s+of\s+hello\s*world\b",
        r"\bhello\s*world\s+(?:in\s+python|python\s+code|program)\b",
        r"\bhow\s+to\s+write\s+(?:a\s+)?(?:python\s+)?(?:function|script|code|program)\b",
        r"\bfunction\s+(?:to|that)\s+(?:adds|reverses|computes|calculates|checks|finds|prints|returns)\b",
        r"\bpython\s+(?:code|function|script)\s+(?:to|for|that|of)\b",
    ]

    # Current factual / entity patterns requiring web research
    CURRENT_FACTUAL_PATTERNS = [
        r"\b(?:who\s+is|who\s+are|who\s+was)\s+(?:the\s+)?(?:current\s+|present\s+|acting\s+)?(?:chief\s+minister|prime\s+minister|president|governor|mayor|ceo|leader|chancellor|head\s+of\s+state)\b",
        r"\b(?:who\s+is|who\s+are)\s+(?:the\s+)?(?:chief\s+minister|cm)\s+of\s+([a-zA-Z\s]+)\b",
        r"\b(?:current|present|latest|recent)\s+(?:president|prime\s+minister|chief\s+minister|ceo|leader)\s+of\b",
        r"\b(?:capital|population|currency|founder|headquarters|net\s+worth|gdp)\s+of\s+([a-zA-Z\s]+)\b",
        r"\b(?:who won|who defeated|who played|who founded|who created|who invented|who discovered|who wrote)\b",
        r"\b(?:latest|current|recent)\s+(?:news|update|status|price|weather|events?|results?)\b",
        r"\b(?:weather\s+in|forecast\s+for|temperature\s+in)\s+([a-zA-Z\s]+)\b",
        r"^(?:tell me about|grab info on|search for|find info on|look up)\b",
    ]

    # Conversational follow-up patterns
    FOLLOW_UP_PATTERNS = [
        r"^(?:tell\s+me\s+more|what\s+about\s+(?:him|her|them|that|it)|can\s+you\s+elaborate|explain\s+further|why\s+is\s+that|what\s+else)\b",
        r"^(?:and\s+then|what\s+happened\s+next|who\s+else|is\s+there\s+more)\b",
    ]

    @classmethod
    def classify_intent(cls, prompt: str, conversation_turn: int = 0) -> Tuple[UserIntent, Optional[str]]:
        """
        Classifies prompt into one of the 8 canonical research intent categories.
        Returns: (UserIntent, extracted_detail_or_expression)
        """
        if not prompt or not prompt.strip():
            return UserIntent.GENERAL_KNOWLEDGE, None

        cleaned = prompt.strip()
        lower = cleaned.lower()

        # 1. Conversational follow-up check
        for pat in cls.FOLLOW_UP_PATTERNS:
            if re.search(pat, lower):
                return UserIntent.CONVERSATIONAL_FOLLOW_UP, None

        # 2. Check for code explanation
        for pat in cls.CODE_EXPLANATION_PATTERNS:
            if re.search(pat, lower):
                return UserIntent.CODE_EXPLANATION, None

        # 3. Check for code debugging
        for pat in cls.CODE_DEBUG_PATTERNS:
            if re.search(pat, lower):
                return UserIntent.CODE_DEBUGGING, None

        # 4. Check for explicit execution
        # Note: Prompts starting with "write", "create", "show", "debug", "explain" are NOT explicit execution.
        is_non_exec_cmd = any(lower.startswith(w) for w in ["write ", "create ", "give ", "show ", "implement ", "how to ", "debug ", "explain "])
        if not is_non_exec_cmd:
            for pat in cls.EXPLICIT_EXEC_PATTERNS:
                if re.search(pat, lower):
                    return UserIntent.EXPLICIT_CODE_EXECUTION, None

        # 5. Check for code generation requests
        for pat in cls.CODE_GEN_PATTERNS:
            if re.search(pat, lower):
                return UserIntent.CODE_GENERATION, None

        # Secondary code check: keywords like "hello world" + "python"
        if ("hello world" in lower or "helloworld" in lower) and ("python" in lower or "code" in lower or "program" in lower):
            return UserIntent.CODE_GENERATION, None

        # 6. Check for simple arithmetic
        # Strip conversational fillers
        arith_prompt = lower
        for filler in ["whats ", "what's ", "what is ", "calculate ", "compute ", "solve ", "tell me "]:
            if arith_prompt.startswith(filler):
                arith_prompt = arith_prompt[len(filler):].strip()

        arith_prompt = arith_prompt.rstrip("?").strip()
        # Verify if arith_prompt contains numbers and operators only
        if re.match(r"^[0-9\.\s\+\-\*\/\(\)\^%]+$", arith_prompt):
            has_op = any(op in arith_prompt for op in ["+", "-", "*", "/", "%", "^"])
            has_digit = any(c.isdigit() for c in arith_prompt)
            if has_op and has_digit:
                try:
                    val = cls._safe_eval_math(arith_prompt)
                    if val is not None:
                        return UserIntent.SIMPLE_ARITHMETIC, str(val)
                except Exception:
                    pass

        # 7. Check for current factual inquiries
        for pat in cls.CURRENT_FACTUAL_PATTERNS:
            if re.search(pat, lower):
                return UserIntent.CURRENT_FACTUAL_QUERY, None

        # Generic question words with entities
        if re.match(r"^(?:who|what|where|when|which)\s+(?:is|are|was|were|won)\s+[a-z0-9\s]{3,}\??$", lower):
            # Exclude elementary arithmetic questions like "what is 1 + 1"
            if not any(c.isdigit() for c in lower):
                return UserIntent.CURRENT_FACTUAL_QUERY, None

        # 8. Default to General Knowledge
        return UserIntent.GENERAL_KNOWLEDGE, None

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
        when the model attempts tool execution instead of direct code generation.
        """
        lower = prompt.lower()
        if "hello" in lower and "world" in lower:
            return "```python\nprint(\"Hello, World!\")\n```"
        elif "add" in lower and ("two" in lower or "number" in lower):
            return "```python\ndef add(a, b):\n    \"\"\"Returns the sum of two numbers.\"\"\"\n    return a + b\n```"
        elif "reverse" in lower and "string" in lower:
            return "```python\ndef reverse_string(s: str) -> str:\n    \"\"\"Reverses the input string.\"\"\"\n    return s[::-1]\n```"
        elif "fibonacci" in lower or "fib" in lower:
            return "```python\ndef fibonacci(n: int) -> int:\n    \"\"\"Returns the nth Fibonacci number.\"\"\"\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n```"
        else:
            return "```python\n# Solution\ndef solve(*args, **kwargs):\n    pass\n```"
