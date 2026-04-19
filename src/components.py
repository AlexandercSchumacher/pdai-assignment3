"""Custom HTML/SVG metric card components.

These replace `st.metric` on the highest-visibility views (Forecast top row,
Drivers model metrics) with a branded card that pairs the number with a trend
indicator and optionally a tiny inline-SVG sparkline.
"""

from __future__ import annotations

import html
from typing import Sequence

import streamlit as st


PALETTE_PRIMARY = "#0f3460"
PALETTE_MUTED = "#9aa4b3"
PALETTE_GREEN = "#2e7d32"
PALETTE_RED = "#c62828"
PALETTE_AMBER = "#ed6c02"
PALETTE_NEUTRAL = "#5a6577"


def _normalize(values: Sequence[float], height: int) -> list[float]:
    floats = [float(v) for v in values]
    if not floats:
        return []
    lo, hi = min(floats), max(floats)
    if hi - lo < 1e-9:
        mid = height / 2
        return [mid for _ in floats]
    span = hi - lo
    pad = 2
    return [height - pad - (v - lo) / span * (height - 2 * pad) for v in floats]


def sparkline(
    values: Sequence[float],
    *,
    baseline_values: Sequence[float] | None = None,
    width: int = 100,
    height: int = 28,
    color: str = PALETTE_PRIMARY,
    baseline_color: str = PALETTE_MUTED,
) -> str:
    """Return an inline SVG polyline sparkline. Empty input → empty string."""
    if not values:
        return ""
    points = _normalize(values, height)
    xs = [i / max(len(points) - 1, 1) * (width - 4) + 2 for i in range(len(points))]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, points))
    last_x, last_y = xs[-1], points[-1]

    layers: list[str] = []
    if baseline_values and len(baseline_values) == len(values):
        bpoints = _normalize(list(baseline_values) + list(values), height)[: len(baseline_values)]
        bpath = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, bpoints))
        layers.append(
            f'<polyline points="{bpath}" fill="none" stroke="{baseline_color}" '
            f'stroke-width="1.5" stroke-dasharray="3,2"/>'
        )

    layers.append(
        f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
    )
    layers.append(f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2.5" fill="{color}"/>')

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">'
        f'{"".join(layers)}</svg>'
    )


def residual_histogram_svg(
    residuals: Sequence[float],
    *,
    bins: int = 16,
    width: int = 100,
    height: int = 28,
    color: str = PALETTE_PRIMARY,
) -> str:
    if not residuals:
        return ""
    floats = [float(v) for v in residuals]
    lo, hi = min(floats), max(floats)
    if hi - lo < 1e-9:
        return sparkline([1.0] * bins, width=width, height=height, color=color)
    span = hi - lo
    counts = [0] * bins
    for v in floats:
        idx = min(int((v - lo) / span * bins), bins - 1)
        counts[idx] += 1
    cmax = max(counts) or 1
    bar_w = (width - 2) / bins
    bars: list[str] = []
    for i, c in enumerate(counts):
        bar_h = (c / cmax) * (height - 3)
        x = 1 + i * bar_w
        y = height - 1 - bar_h
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_w - 0.6, 0.4):.2f}" '
            f'height="{bar_h:.2f}" fill="{color}" opacity="0.85"/>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">'
        f'{"".join(bars)}</svg>'
    )


def risk_chip(level: str) -> str:
    normalized = (level or "").strip().lower()
    cls = "risk-medium"
    if normalized in {"low", "niedrig"}:
        cls = "risk-low"
    elif normalized in {"high", "hoch"}:
        cls = "risk-high"
    safe = html.escape(level or "—")
    return f'<span class="risk-chip {cls}">{safe}</span>'


def r2_chip(r2: float) -> str:
    if r2 >= 0.5:
        cls = "risk-low"
    elif r2 >= 0:
        cls = "risk-medium"
    else:
        cls = "risk-high"
    return f'<span class="risk-chip {cls}">{r2:.3f}</span>'


def metric_card(
    label: str,
    value: str,
    *,
    delta: float | None = None,
    delta_label: str | None = None,
    sparkline_svg: str | None = None,
    extra_note: str | None = None,
    value_is_html: bool = False,
) -> str:
    safe_label = html.escape(label)
    safe_value = value if value_is_html else html.escape(value)
    delta_html = ""
    if delta is not None:
        if delta > 0.05:
            arrow, cls = "↑", "metric-delta-up"
        elif delta < -0.05:
            arrow, cls = "↓", "metric-delta-down"
        else:
            arrow, cls = "→", "metric-delta-neutral"
        text_bits = [f"{arrow} {delta:+.1f}"]
        if delta_label:
            text_bits.append(html.escape(delta_label))
        delta_html = f'<div class="metric-delta {cls}">{" ".join(text_bits)}</div>'
    elif delta_label:
        delta_html = (
            f'<div class="metric-delta metric-delta-neutral">{html.escape(delta_label)}</div>'
        )
    spark_html = (
        f'<div class="metric-sparkline">{sparkline_svg}</div>' if sparkline_svg else ""
    )
    note_html = (
        f'<div class="metric-note">{html.escape(extra_note)}</div>' if extra_note else ""
    )
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{safe_label}</div>'
        f'<div class="metric-value">{safe_value}</div>'
        f"{spark_html}"
        f"{delta_html}"
        f"{note_html}"
        f"</div>"
    )


def render_card(column, card_html: str) -> None:
    column.markdown(card_html, unsafe_allow_html=True)
