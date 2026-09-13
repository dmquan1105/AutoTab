"""Viewport planning compatibility module."""

from .utils.ranges import viewport


class WindowPlanner:
    """Plan one clipped viewport per cell finding."""

    def __init__(self, size: int = 9) -> None:
        self.size = size

    def plan(self, anchor: str, max_row: int, max_column: int) -> str:
        return viewport(anchor, max_row, max_column, self.size)
