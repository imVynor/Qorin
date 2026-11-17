import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SANDBOX_ROOT = REPO_ROOT / "backend/data/sandbox"


class PythonSandboxResult:
    def __init__(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        execution_time_ms: float,
        timed_out: bool = False,
        error: Optional[str] = None,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.execution_time_ms = execution_time_ms
        self.timed_out = timed_out
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "timed_out": self.timed_out,
            "error": self.error,
        }


class PythonSandbox:
    """Executes untrusted Python snippets in an isolated, restricted subprocess environment."""

    TIMEOUT_SECONDS = 5.0
    MAX_OUTPUT_BYTES = 50_000

    # Whitelist of safe standard library modules; dangerous modules blocked at launch
    RESTRICTED_HEADER = """
import sys
# Block sensitive dangerous imports
_BLOCKED = {'socket', 'urllib', 'http', 'ftplib', 'smtplib', 'webbrowser', 'subprocess', 'shutil'}
for mod in _BLOCKED:
    sys.modules[mod] = None
"""

    def __init__(self, sandbox_dir: Optional[Path] = None):
        self.sandbox_dir = sandbox_dir or SANDBOX_ROOT
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, code: str, timeout: float = TIMEOUT_SECONDS) -> PythonSandboxResult:
        # Create an isolated temporary run directory inside sandbox_dir
        run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=str(self.sandbox_dir)))
        script_file = run_dir / "user_script.py"

        # Sanitize code: reject obvious destructive primitives
        forbidden = ["os.system", "os.popen", "subprocess", "__import__('subprocess')", "rmdir", "unlink"]
        for f in forbidden:
            if f in code:
                return PythonSandboxResult(
                    stdout="",
                    stderr=f"SecurityViolation: Use of '{f}' is strictly prohibited in sandbox environment.",
                    exit_code=1,
                    execution_time_ms=0.0,
                    error="Security violation detected.",
                )

        # Write code with security restrictions header
        full_code = self.RESTRICTED_HEADER + "\n" + code
        script_file.write_text(full_code, encoding="utf-8")

        # Strip environment variables to prevent token/key leaks
        clean_env = {
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": "",
            "TEMP": str(run_dir),
            "TMP": str(run_dir),
        }

        # Resolve python executable (using project's virtualenv python)
        py_exe = sys.executable

        start_time = time.perf_counter()
        proc = None
        timed_out = False
        stdout_text = ""
        stderr_text = ""
        exit_code = 0

        try:
            proc = subprocess.Popen(
                [py_exe, "-I", str(script_file)],  # -I isolates from user site-packages/environment
                cwd=str(run_dir),
                env=clean_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            stdout_text, stderr_text = proc.communicate(timeout=timeout)
            exit_code = proc.returncode

        except subprocess.TimeoutExpired:
            timed_out = True
            if proc:
                proc.kill()
                try:
                    stdout_text, stderr_text = proc.communicate(timeout=1.0)
                except Exception:
                    pass
            stderr_text += f"\nTimeoutError: Execution exceeded time limit of {timeout}s."
            exit_code = 124

        except Exception as e:
            stderr_text += f"\nExecutionError: {e}"
            exit_code = 1

        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            # Cleanup temp directory
            try:
                shutil.rmtree(run_dir, ignore_errors=True)
            except Exception:
                pass

        # Truncate runaway output
        if len(stdout_text) > self.MAX_OUTPUT_BYTES:
            stdout_text = stdout_text[: self.MAX_OUTPUT_BYTES] + "\n[...output truncated due to length...]"

        return PythonSandboxResult(
            stdout=stdout_text,
            stderr=stderr_text,
            exit_code=exit_code,
            execution_time_ms=elapsed_ms,
            timed_out=timed_out,
            error=stderr_text.strip() if exit_code != 0 else None,
        )

