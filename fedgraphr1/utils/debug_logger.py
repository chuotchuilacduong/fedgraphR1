"""
fedgraphr1/utils/debug_logger.py
==================================
Step-level debug logging for the FL pipeline.

Usage in main.py:
    from fedgraphr1.utils.debug_logger import setup_logging, step

    setup_logging(working_dir="expr/hotpotqa/run1", debug=True)

    step("Round 0 START")
    step("  [a] Server downlink", detail={"lora_tensors": 4})

Log levels:
  - Console: INFO normally, DEBUG when debug=True
  - File (debug.log): always DEBUG — full trace regardless of --debug flag
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

# Module-level step logger — used by step() below
_step_logger = logging.getLogger("fedgraphr1.step")


def setup_logging(working_dir: str, debug: bool = False) -> str:
    """Configure console + file logging for a run.

    Args:
        working_dir: Run output directory (debug.log written here).
        debug:       If True, console also shows DEBUG messages.

    Returns:
        Path to the debug log file.
    """
    os.makedirs(working_dir, exist_ok=True)
    log_path = os.path.join(working_dir, "debug.log")

    root = logging.getLogger()
    # Remove existing handlers to avoid duplicate output
    for h in root.handlers[:]:
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s │ %(name)-20s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler — always DEBUG (full trace)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    root.setLevel(logging.DEBUG)

    logging.getLogger("fedgraphr1").info(
        f"Logging initialised → {log_path}  (console={'DEBUG' if debug else 'INFO'})"
    )
    return log_path


# ─────────────────────────────────────────────────────────────────────────────
# Step logger — structured pipeline trace
# ─────────────────────────────────────────────────────────────────────────────

class StepTimer:
    """Context manager that logs step start/end with elapsed time."""

    def __init__(self, label: str, detail: Optional[Dict] = None):
        self.label = label
        self.detail = detail or {}
        self._start = None

    def __enter__(self):
        self._start = time.time()
        extra = "  " + _fmt_detail(self.detail) if self.detail else ""
        _step_logger.debug(f"[START] {self.label}{extra}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start
        if exc_type:
            _step_logger.debug(f"[FAIL]  {self.label}  ({elapsed:.3f}s)  {exc_val}")
        else:
            _step_logger.debug(f"[DONE]  {self.label}  ({elapsed:.3f}s)")
        return False  # don't suppress exceptions


def step(label: str, detail: Optional[Dict[str, Any]] = None):
    """Log a single pipeline step at DEBUG level.

    Example:
        step("Client 0: entity extraction", {"entities": 12, "hyperedges": 3})
    """
    extra = "  " + _fmt_detail(detail) if detail else ""
    _step_logger.debug(f"[STEP]  {label}{extra}")


def timed_step(label: str, detail: Optional[Dict[str, Any]] = None) -> StepTimer:
    """Return a context manager that logs start + elapsed time.

    Example:
        with timed_step("Server.execute()"):
            server.execute()
    """
    return StepTimer(label, detail)


def _fmt_detail(d: Dict[str, Any]) -> str:
    return "  ".join(f"{k}={v}" for k, v in d.items())
