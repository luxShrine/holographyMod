import logging
import pathlib
import time
from contextlib import contextmanager

import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import install as install_rich_pretty
from rich.traceback import install as install_rich_traceback

###############################################################################
# console shared by rich progress & logging
logger = logging.getLogger(__name__)
console_ = Console()  # reuse elsewhere
shell_handler = RichHandler(
    console=console_,
    rich_tracebacks=True,  # colourised trace backs
    markup=True,  # allows embed [bold red]
    show_time=True,
    show_level=True,
    show_path=False,  # keep the line number but drop long paths
    keywords=["loss", "accuracy", "MAE", "train", "test"],  # highlight custom words
)


class TensorSummaryFilter(logging.Filter):
    def filter(self, record):
        record.msg = (
            f"<Tensor shape={tuple(record.msg.shape)}>"
            if isinstance(record.msg, torch.Tensor)
            else record.msg
        )
        record.args = tuple(
            f"<Tensor shape={tuple(x.shape)}>" if isinstance(x, torch.Tensor) else x
            for x in record.args
        )
        return True


###############################################################################
# File handler keeps the long format
logfile = pathlib.Path("debug.log")
file_handler = logging.FileHandler(logfile, encoding="utf-8")

###############################################################################
# Assemble logger

logger.propagate = False  # avoid double-printing if libraries use root

# what level to use
logger.setLevel(logging.DEBUG)
shell_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)

# the formatter determines what our logs will look like
fmt_shell = "%(message)s"
fmt_file = "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"

shell_formatter = logging.Formatter(fmt_shell)
file_formatter = logging.Formatter(fmt_file)

# sets the logs to use the format defined above
shell_handler.addFilter(TensorSummaryFilter())
shell_handler.setFormatter(shell_formatter)
file_handler.setFormatter(file_formatter)
logger.addHandler(shell_handler)
logger.addHandler(file_handler)


logging.basicConfig(
    # level=logging.DEBUG,  # root level
    handlers=[shell_handler, file_handler],
    force=True,  # overwrite any prior config
)

###############################################################################
# rich.pretty & rich.traceback

install_rich_pretty()  # pretty-print dicts, lists, numpy arrays,
install_rich_traceback(  # color + locals when uncaught exception
    console=console_,
    show_locals=True,
    width=120,
    extra_lines=3,
)


def set_verbosity(debug: bool = False) -> None:
    """Raise terminal log level from INFO to DEBUG at runtime."""
    new_level = logging.DEBUG if debug else logging.INFO
    shell_handler.setLevel(new_level)
    logger.debug("Switched console log level to %s", logging.getLevelName(new_level))


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
