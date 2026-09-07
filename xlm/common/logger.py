import logging
import os
import re
from typing import Any, ClassVar

import swanlab


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""

    # ANSI color codes
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET: ClassVar[str] = "\033[0m"
    BOLD: ClassVar[str] = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
            )
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == "INFO":
            # Highlight numbers and percentages
            message = re.sub(
                r"(\d+\.?\d*\s*(?:GB|MB|%|docs))",
                rf"{self.BOLD}\1{self.RESET}",
                message,
            )
            message = re.sub(
                r"(Shard \d+)",
                rf"{self.COLORS['INFO']}{self.BOLD}\1{self.RESET}",
                message,
            )
        return message


def setup_default_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])


setup_default_logging()
logger = logging.getLogger(__name__)


def print0(s: str = "", **kwargs: Any) -> None:
    # **kwargs is forwarded to print(), whose keyword params mypy types explicitly
    ddp_rank = int(os.environ.get("RANK", "0"))
    if ddp_rank == 0:
        print(s, **kwargs)


def print_banner() -> None:
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)


class DummySwanLab:
    """Useful if we wish to not use swanlab but have all the same signatures"""

    def __init__(self) -> None:
        pass

    def log(self, *args: object, **kwargs: object) -> None:
        pass

    def finish(self) -> None:
        pass


def init_swanlab(
    project: str,
    run_name: str,
    config: dict[str, Any],
    *,
    enabled: bool = True,
    entity: str | None = None,
) -> Any:
    """Initialize a SwanLab run, falling back to a no-op stub when disabled.

    Use the returned handle for all ``log``/``finish`` calls; when ``enabled``
    is False the handle is a :class:`DummySwanLab` and every call is a no-op.

    Args:
        project: SwanLab project name.
        run_name: Experiment (run) name.
        config: Hyperparameters recorded on the run.
        enabled: Whether to create a real SwanLab run (e.g. master process
            plus the experiment's ``log_wandb`` flag).
        entity: SwanLab workspace/organization username.

    Returns:
        A SwanLab run handle or a :class:`DummySwanLab` stub.
    """
    if not enabled:
        return DummySwanLab()
    init_kwargs: dict[str, Any] = {"project": project, "name": run_name}
    if entity is not None:
        init_kwargs["entity"] = entity
    # swanlab is an untyped third-party SDK (see mypy overrides in pyproject.toml)
    return swanlab.init(**init_kwargs, config=config)
