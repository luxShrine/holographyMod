from rich.progress import ProgressColumn, Task
from rich.text import Text


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
