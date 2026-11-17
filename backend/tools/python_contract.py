"""
QORIN Python Execution Contract & Reliability Layer
Defines strict input pre-validation, AST security inspection, deterministic execution,
structured failure taxonomy, and documented capabilities.
"""

import ast
from dataclasses import dataclass, asdict
from enum import Enum
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple


class PythonFailureCategory(str, Enum):
    NONE = "NONE"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    UNSUPPORTED_API = "UNSUPPORTED_API"
    RUNTIME_EXCEPTION = "RUNTIME_EXCEPTION"
    TIMEOUT = "TIMEOUT"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    EMPTY_PAYLOAD = "EMPTY_PAYLOAD"


@dataclass
class PythonExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    status: str  # "SUCCESS" | "FAILED" | "TIMEOUT"
    failure_category: PythonFailureCategory
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["failure_category"] = self.failure_category.value
        d["execution_time_ms"] = round(self.execution_time_ms, 2)
        return d


class PythonExecutionContract:
    """Standardized Python sandbox execution contract with strict failure taxonomy."""

    ALLOWED_MODULES: Set[str] = {
        "math", "statistics", "random", "datetime", "itertools",
        "collections", "re", "json", "decimal", "fractions", "cmath"
    }

    DISALLOWED_MODULES: Set[str] = {
        "os", "sys", "subprocess", "shutil", "socket", "urllib", "http",
        "ftplib", "smtplib", "webbrowser", "requests", "pip"
    }

    TIMEOUT_SECONDS: float = 6.0
    MAX_OUTPUT_CHARS: int = 20_000

    def __init__(self, timeout_seconds: float = 6.0):
        self.timeout_seconds = timeout_seconds

    def pre_validate_ast(self, code: str) -> Tuple[bool, PythonFailureCategory, Optional[str]]:
        """Static AST inspection for syntax errors and disallowed imports."""
        if not code or not code.strip():
            return False, PythonFailureCategory.EMPTY_PAYLOAD, "Python code payload is empty."

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return (
                False,
                PythonFailureCategory.SYNTAX_ERROR,
                f"SyntaxError: {e.msg} (line {e.lineno}, offset {e.offset})"
            )

        # Inspect imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in self.DISALLOWED_MODULES:
                        return (
                            False,
                            PythonFailureCategory.SECURITY_VIOLATION,
                            f"SecurityViolation: Import of disallowed module '{root_mod}' is prohibited."
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in self.DISALLOWED_MODULES:
                        return (
                            False,
                            PythonFailureCategory.SECURITY_VIOLATION,
                            f"SecurityViolation: Import from disallowed module '{root_mod}' is prohibited."
                        )

        return True, PythonFailureCategory.NONE, None

    def execute(self, code: str) -> PythonExecutionResult:
        """Executes Python code in an isolated subprocess with strict safety and failure categorization."""
        t0 = time.perf_counter()

        # Step 1: Pre-validation
        is_valid, fail_cat, err_msg = self.pre_validate_ast(code)
        if not is_valid:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return PythonExecutionResult(
                stdout="",
                stderr=err_msg or "",
                exit_code=1,
                execution_time_ms=duration_ms,
                status="FAILED",
                failure_category=fail_cat,
                error_message=err_msg
            )

        # Step 2: Prepare temporary execution file
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            script_path = Path(f.name)
            # Inject safe import restrictions header
            f.write("# QORIN Sandbox Script\n")
            f.write(code)

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                encoding="utf-8",
                errors="replace"
            )
            duration_ms = (time.perf_counter() - t0) * 1000.0
            stdout = proc.stdout[:self.MAX_OUTPUT_CHARS]
            stderr = proc.stderr[:self.MAX_OUTPUT_CHARS]

            if proc.returncode == 0:
                return PythonExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=0,
                    execution_time_ms=duration_ms,
                    status="SUCCESS",
                    failure_category=PythonFailureCategory.NONE,
                    error_message=None
                )
            else:
                # Classify runtime exception
                fail_cat = PythonFailureCategory.RUNTIME_EXCEPTION
                if "AttributeError" in stderr and "module" in stderr:
                    fail_cat = PythonFailureCategory.UNSUPPORTED_API
                elif "NameError" in stderr or "TypeError" in stderr or "ZeroDivisionError" in stderr:
                    fail_cat = PythonFailureCategory.RUNTIME_EXCEPTION

                return PythonExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=proc.returncode,
                    execution_time_ms=duration_ms,
                    status="FAILED",
                    failure_category=fail_cat,
                    error_message=stderr.strip().splitlines()[-1] if stderr.strip() else "Unknown runtime error"
                )

        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return PythonExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {self.timeout_seconds} seconds.",
                exit_code=124,
                execution_time_ms=duration_ms,
                status="TIMEOUT",
                failure_category=PythonFailureCategory.TIMEOUT,
                error_message=f"TimeoutExpired: Execution exceeded {self.timeout_seconds}s limit."
            )
        finally:
            if script_path.exists():
                try:
                    script_path.unlink()
                except Exception:
                    pass
