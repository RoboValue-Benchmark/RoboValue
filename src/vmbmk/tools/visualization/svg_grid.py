"""Shared SVG grid geometry; each curve keeps its own history and score overlays."""
from __future__ import annotations


def grid_elements(
    width: int, height: int, margins: tuple[int, int, int, int],
    limits: tuple[float, float], maximum: int,
) -> list[str]:
    left, right, top, bottom = margins
    y_min, y_max = limits
    plot_width = width - left - right
    plot_height = height - top - bottom
    elements = []
    for index in range(6):
        fraction = index / 5
        score = y_max - fraction * (y_max - y_min)
        y_pos = top + fraction * plot_height
        elements.extend((
            f'<line class="grid" x1="{left}" y1="{y_pos:.2f}" x2="{width - right}" y2="{y_pos:.2f}"/>',
            f'<text x="{left - 10}" y="{y_pos + 4:.2f}" font-size="12" text-anchor="end">{score:.4g}</text>',
        ))
    for index in range(6):
        fraction = index / 5
        position = round(fraction * maximum)
        x_pos = left + fraction * plot_width
        elements.extend((
            f'<line class="grid" x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" y2="{height - bottom}"/>',
            f'<text x="{x_pos:.2f}" y="{height - bottom + 22}" font-size="12" text-anchor="middle">{position}</text>',
        ))
    return elements
