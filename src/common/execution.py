"""
Sandboxed execution utilities for running Python code that comes out of an LLM.
Adapted from OpenAI HumanEval code:
https://github.com/openai/human-eval/blob/master/human_eval/execution.py

What is covered:
- Each execution runs in its own process (can be killed if it hangs or crashes)
- Execution is limited by a timeout to stop infinite loops
- Memory limits are enforced by default (256MB)
- stdout and stderr are captured and returned
- Code runs in a temporary directory that is deleted afterwards
- Dangerous functions are disabled (examples: os.system, os.kill, shutil.rmtree, subprocess.Popen)

What is not covered:
- Not a true security sandbox
- Network access is not blocked (e.g. sockets could be opened)
- Python's dynamic features (e.g. ctypes) could bypass restrictions
- No kernel-level isolation (no seccomp, no containers, no virtualization)

Overall this sandbox is good for evaluation of generated code and protects against
accidental destructive behavior, but it is not safe against malicious adversarial code.
"""

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, NoReturn, cast

# -----------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Result of executing Python code in a sandbox."""

    success: bool
    stdout: str
    stderr: str
    error: str | None = None
    timeout: bool = False
    memory_exceeded: bool = False

    def __repr__(self) -> str:
        parts: list[str] = []
        parts.append(f"ExecutionResult(success={self.success}")
        if self.timeout:
            parts.append(", timeout=True")
        if self.memory_exceeded:
            parts.append(", memory_exceeded=True")
        if self.error:
            parts.append(f", error={self.error!r}")
        if self.stdout:
            parts.append(f", stdout={self.stdout!r}")
        if self.stderr:
            parts.append(f", stderr={self.stderr!r}")
        parts.append(")")
        return "".join(parts)


@contextlib.contextmanager
def time_limit(seconds: float) -> Generator[None, None, None]:
    def signal_handler(signum: int, frame: Any) -> NoReturn:
        raise TimeoutException("Timed out!")

    if (
        hasattr(signal, "setitimer")
        and hasattr(signal, "ITIMER_REAL")
        and hasattr(signal, "SIGALRM")
    ):
        # POSIX-only attrs; guarded by hasattr() above so the module imports on
        # Windows, but the signal stubs do not declare them for all platforms.
        signal.setitimer(signal.ITIMER_REAL, seconds)  # type: ignore[attr-defined]
        signal.signal(signal.SIGALRM, signal_handler)  # type: ignore[attr-defined]
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)  # type: ignore[attr-defined]
    else:
        # POSIX interval timers are unavailable (e.g. Windows); the parent
        # process in execute_code() kills the worker on timeout instead.
        yield


@contextlib.contextmanager
def capture_io() -> Generator[tuple[io.StringIO, io.StringIO], None, None]:
    """Capture stdout and stderr, and disable stdin."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    stdin_block = WriteOnlyStringIO()
    with (
        contextlib.redirect_stdout(stdout_capture),
        contextlib.redirect_stderr(stderr_capture),
        redirect_stdin(stdin_block),
    ):
        yield stdout_capture, stderr_capture


@contextlib.contextmanager
def create_tempdir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as dirname, chdir(dirname):
        yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args: Any, **kwargs: Any) -> str:
        raise OSError

    def readline(self, size: int | None = None) -> NoReturn:
        raise OSError

    def readlines(self, hint: int = -1) -> NoReturn:
        raise OSError

    def readable(self, *args: Any, **kwargs: Any) -> bool:
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root: str) -> Generator[None, None, None]:
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: int | None = None) -> None:
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if platform.uname().system != "Darwin":
        try:
            import resource
        except ImportError:
            return
        setrlimit = getattr(resource, "setrlimit", None)
        if setrlimit is None or maximum_memory_bytes is None:
            return
        for rlimit_name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_STACK"):
            rlimit = getattr(resource, rlimit_name, None)
            if rlimit is not None:
                setrlimit(rlimit, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    for _name in ("exit", "quit"):
        setattr(builtins, _name, None)

    os.environ["OMP_NUM_THREADS"] = "1"

    # Null out the destructive functions via setattr() so the sandboxed code
    # sees None instead of the real callables.
    for _name in (
        "kill",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "lchflags",
        "lchmod",
        "lchown",
        "getcwd",
        "chdir",
    ):
        setattr(os, _name, None)

    import shutil

    for _name in ("rmtree", "move", "chown"):
        setattr(shutil, _name, None)

    import subprocess

    setattr(subprocess, "Popen", None)  # noqa: B010

    __builtins__["help"] = None

    import sys

    # Block re-import of debugging / process-introspection modules. The module
    # table intentionally holds None, so the value type is dynamic.
    sys_modules: dict[str, Any] = cast(dict[str, Any], sys.modules)
    for _name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys_modules[_name] = None


def _unsafe_execute(
    code: str,
    timeout: float,
    maximum_memory_bytes: int | None,
    result_dict: dict[str, Any],
) -> None:
    """Execute code in a subprocess with safety guards. Results are written to result_dict."""
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        unlink = os.unlink

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)

        # Default to failure
        result_dict.update(
            {
                "success": False,
                "stdout": "",
                "stderr": "",
                "timeout": False,
                "memory_exceeded": False,
                "error": None,
            }
        )

        try:
            exec_globals: dict[str, Any] = {}
            with capture_io() as (stdout_capture, stderr_capture), time_limit(timeout):
                # WARNING
                # This program exists to execute untrusted model-generated code. Although
                # it is highly unlikely that model-generated code will do something overtly
                # malicious in response to this test suite, model-generated code may act
                # destructively due to a lack of model capability or alignment.
                # Users are strongly encouraged to sandbox this evaluation suite so that it
                # does not perform destructive actions on their host or network. For more
                # information on how OpenAI sandboxes its code, see the accompanying paper.
                # Once you have read this disclaimer and taken appropriate precautions,
                # uncomment the following line and proceed at your own risk:
                exec(code, exec_globals)  # noqa: S102 - this module exists solely to run untrusted code in a sandbox

            result_dict.update(
                {
                    "success": True,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue(),
                }
            )

        except TimeoutException:
            result_dict.update(
                {
                    "timeout": True,
                    "error": "Execution timed out",
                }
            )

        except MemoryError as e:
            result_dict.update(
                {
                    "memory_exceeded": True,
                    "error": f"Memory limit exceeded: {e}",
                }
            )

        except BaseException as e:  # noqa: BLE001 - sandbox must never let untrusted code escape
            result_dict.update(
                {
                    "error": f"{type(e).__name__}: {e}",
                }
            )

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        os.unlink = unlink


def execute_code(
    code: str,
    timeout: float = 5.0,  # 5 seconds default
    maximum_memory_bytes: int | None = 256 * 1024 * 1024,  # 256MB default
) -> ExecutionResult:
    """
    Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute as a string
        timeout: Maximum execution time in seconds (default: 5.0)
        maximum_memory_bytes: Memory limit in bytes (default: 256MB, None to disable)

    Returns:
        ExecutionResult with success status, stdout/stderr, and error information

    Example:
        >>> result = execute_code("print('hello world')")
        >>> result.success
        True
        >>> result.stdout
        'hello world\\n'
    """

    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    p = multiprocessing.Process(
        target=_unsafe_execute, args=(code, timeout, maximum_memory_bytes, result_dict)
    )
    p.start()
    p.join(timeout=timeout + 1)

    if p.is_alive():
        p.kill()
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            error="Execution timed out (process killed)",
            timeout=True,
            memory_exceeded=False,
        )

    if not result_dict:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            error="Execution failed (no result returned)",
            timeout=True,
            memory_exceeded=False,
        )

    return ExecutionResult(
        success=result_dict["success"],
        stdout=result_dict["stdout"],
        stderr=result_dict["stderr"],
        error=result_dict["error"],
        timeout=result_dict["timeout"],
        memory_exceeded=result_dict["memory_exceeded"],
    )
