from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from holo.infra.util.types import AnalysisType


class RateColumn(ProgressColumn):
    """Custom class for creating rate column."""

    def render(self, task: Task) -> Text:
        """Render the speed of batch processing."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("", style="progress.percentage")
        return Text(f"{speed:.2f} batch/s", style="progress.data")


class MetricColumn(ProgressColumn):
    """Render any numeric field kept in task.fields (e.g. 'loss', 'acc', 'lr')."""

    def __init__(self, name: str, fmt: str = "{:.4f}", style: str = "cyan"):
        super().__init__()
        self.name, self.fmt, self.style = name, fmt, style

    def render(self, task: Task):
        val = task.fields.get(self.name)
        if val is None:
            return Text("–")
        return Text(self.fmt.format(val), style=self.style)


def setup_training_progress(train_type: AnalysisType) -> Progress:
    """Create and configure a Rich Progress bar for training monitoring."""
    metric_col_name = "val_mae" if train_type == AnalysisType.REG else "val_acc"
    metric_col_fmt = "{:.4f}" if train_type == AnalysisType.REG else "{:.2%}"

    return Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",  # Separator
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        RateColumn(),
        "•",
        MetricColumn("train_loss", fmt="{:.4f}", style="magenta"),
        "•",
        MetricColumn("val_loss", fmt="{:.4f}", style="yellow"),
        "•",
        MetricColumn(metric_col_name, fmt=metric_col_fmt, style="green"),
        "•",
        MetricColumn("lr", fmt="{:.1e}", style="dim cyan"),  # Shorter LR format
        SpinnerColumn(),
        transient=False,  # Keep finished tasks visible
    )
