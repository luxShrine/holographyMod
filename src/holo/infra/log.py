"""Central logging setup (queue listener pattern)."""

from __future__ import annotations

import atexit
import logging
import logging.config
import logging.handlers as lh
import multiprocessing as mp
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

import rich  # noqa
from pythonjsonlogger import jsonlogger
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

_LOG_FILE: Final[Path] = Path("logs/my_app.log.jsonl")
_LOG_LEVEL = "DEBUG"  # root level – override per logger if needed


# Single, process‑safe queue for all log records
_queue: lh.Queue = mp.Manager().Queue(-1)


# -- Formatters -------------------------------------------------------------------------
class JsonFormatter(jsonlogger.JsonFormatter):
    """Add milliseconds and process name."""

    def add_fields(self, log_record, record, message_dict):
        """Inject extra fields used by the JSON logger."""
        super().add_fields(log_record, record, message_dict)
        log_record["msecs"] = record.msecs
        log_record["line"] = record.lineno
        log_record["module"] = record.module


json_fmt = JsonFormatter("%(levelname)s %(name)s %(message)s %(asctime)s")


# -- Handlers attached to the QueueListener ------------------------------------------------------
# Keep ~5 MB per file, plus three backups (≈ 20 MB total)
file_handler = RotatingFileHandler(
    _LOG_FILE,
    maxBytes=2 * 1024 * 1024,  # 2 MB
    backupCount=1,
    encoding="utf-8",
)
file_handler.setFormatter(json_fmt)
file_handler.setLevel(logging.DEBUG)


# Console formatter is handled inside RichHandler
# setup rich_tracebacks
_ = install()
rich_handler = RichHandler(
    console=Console(stderr=True, log_time_format="%H:%M:%S"),
    show_level=True,
    show_time=True,
    tracebacks_word_wrap=False,
    rich_tracebacks=True,
    locals_max_length=1,
    locals_max_string=20,
    tracebacks_code_width=10,
    tracebacks_extra_lines=1,
    tracebacks_max_frames=1,
    tracebacks_show_locals=True,
    # tracebacks_suppress=["click", "typer"],
    show_path=True,
)
rich_handler.setLevel(logging.INFO)

_listener = lh.QueueListener(
    _queue,
    file_handler,
    rich_handler,
    respect_handler_level=True,
)

# ---------------------------------------------------------------------------


def init_logging() -> None:
    """Initialise root logger exactly once (idempotent)."""
    if getattr(init_logging, "_configured", False):  # type: ignore[attr-defined]
        return

    queue_handler = lh.QueueHandler(_queue)

    logging.basicConfig(
        level=_LOG_LEVEL,
        format="%(message)s",  # prevents double logging info for console
        handlers=[queue_handler],
        force=True,  # override anything Typer / other libs did
    )

    # Quiet noisy libraries
    # logging.getLogger("typer").setLevel(logging.WARNING)
    # logging.getLogger("click").setLevel(logging.WARNING)

    _listener.start()
    atexit.register(_listener.stop)

    init_logging._configured = True  # type: ignore[attr-defined]


logger = logging.getLogger(__name__)

# console shared by rich progress & logging
console_ = Console()  # reuse elsewhere


@contextmanager
def log_timing(label: str):
    """Wrap calls to time blocks of code.

    Example:
        with log_timing("Epoch validation"):
            validate(...)

    """
    start = time.perf_counter()
    yield
    dur = time.perf_counter() - start
    logger.info("[bold cyan]%s[/] took %.3fs", label, dur)
