"""Inline SVG charts for the PDF report.

Plotted from the same payload the analysis workspace draws, so a figure in the
report and the same figure on screen cannot drift apart. SVG rather than a
plotting library because the output is print: vector strokes stay crisp at any
zoom, the page carries no script, and the renderer has nothing to wait for.

Colours are the design tokens from ``web/src/styles/tokens.css``. They are
repeated here as literals because a stylesheet the PDF never loads cannot
supply them, and a chart that invents its own risk colours would contradict the
screen the reader just came from.
"""

from __future__ import annotations

import html
import math
from typing import Any

from markupsafe import Markup

# ---- Status: risk bands only (tokens.css) ----
STATUS_FILL = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
STATUS_SOFT = {
    "good": "#e8f6e8",
    "warning": "#fef4e0",
    "serious": "#fdeee8",
    "critical": "#fbe9e9",
}
STATUS_INK = {
    "good": "#0a5c0a",
    "warning": "#7a5300",
    "serious": "#8f3d1c",
    "critical": "#8f1f1f",
}

SERIES_1 = "#2a78d6"  # typical
SERIES_2 = "#eb6834"  # peak
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#1a1a2e"
INK_MUTED = "#6c7a89"

#: A4 at 15 mm side margins is 180 mm of content, about 680 px at 96 dpi.
WIDTH = 680


def _text(value: Any) -> str:
    """Escape a label. Chart text is data, and data can contain markup."""

    return html.escape(str(value), quote=True)


def _round(value: float, digits: int = 2) -> str:
    """Trim a trailing decimal zero without eating the zeros in a whole number.

    Stripping unconditionally turns 100 into 1, which is not a rounding error a
    reader would catch: both are plausible percentages.
    """

    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _svg(body: str, height: int, label: str) -> Markup:
    return Markup(
        f'<svg class="chart" viewBox="0 0 {WIDTH} {height}" width="100%" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_text(label)}">'
        f"{body}</svg>"
    )


# ---------------------------------------------------------------------------
# Score over time
# ---------------------------------------------------------------------------


def score_timeline(timeline: dict[str, Any]) -> Markup | None:
    """Score against time, over the method's own risk bands.

    The bands are the background rather than a legend entry because the
    question a reader asks of this figure is "how long was it up there", and
    that is answered by where the line sits, not by matching colours.
    """

    points = (timeline.get("series") or {}).get("points") or []
    if len(points) < 2:
        return None

    method = str(timeline.get("method") or "")
    bands = timeline.get("bands") or []
    scale_max = float(timeline.get("scaleMaximum") or 0) or 1.0
    threshold = timeline.get("actionThreshold")

    left, right, top, bottom = 32, WIDTH - 96, 12, 168
    span_x, span_y = right - left, bottom - top
    max_t = max(float(p["t"]) for p in points) or 1.0
    low_y, high_y = 0.5, scale_max + 0.5

    def x_of(seconds: float) -> float:
        return left + (seconds / max_t) * span_x

    def y_of(score: float) -> float:
        return bottom - ((score - low_y) / (high_y - low_y)) * span_y

    parts: list[str] = []

    # Band backgrounds, each labelled once in the right gutter.
    for band in bands:
        lower, upper = float(band["lower"]), min(float(band["upper"]), scale_max)
        if upper < lower:
            continue
        y_top, y_bottom = y_of(upper + 0.5), y_of(lower - 0.5)
        fill = STATUS_SOFT.get(str(band.get("status")), "#f4f4f2")
        ink = STATUS_INK.get(str(band.get("status")), INK_MUTED)
        parts.append(
            f'<rect x="{left}" y="{y_top:.1f}" width="{span_x}" '
            f'height="{max(0.0, y_bottom - y_top):.1f}" fill="{fill}"/>'
        )
        parts.append(
            f'<text x="{right + 8}" y="{(y_top + y_bottom) / 2 + 3:.1f}" '
            f'font-size="8" fill="{ink}">{_text(band.get("label"))}</text>'
        )

    # Score gridlines, one per whole score the scale actually reaches.
    step = 1 if scale_max <= 8 else 2
    for score in range(1, int(scale_max) + 1, step):
        y = y_of(score)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{left - 6}" y="{y + 3:.1f}" font-size="8" fill="{INK_MUTED}" '
            f'text-anchor="end">{score}</text>'
        )

    # The action threshold, the one line a reader is asked to care about.
    if isinstance(threshold, int | float):
        y = y_of(float(threshold))
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{INK}" '
            f'stroke-width="1" stroke-dasharray="4 3" opacity="0.55"/>'
        )

    coordinates = " ".join(f"{x_of(float(p['t'])):.1f},{y_of(float(p['s'])):.1f}" for p in points)
    parts.append(
        f'<polyline points="{coordinates}" fill="none" stroke="{SERIES_1}" '
        f'stroke-width="0.9" stroke-linejoin="round"/>'
    )

    # Peak marker, so the number quoted in the text is findable on the plot.
    verdict = timeline.get("verdict") or {}
    peak_sec, peak_score = verdict.get("peakSec"), verdict.get("peakScore")
    if isinstance(peak_sec, int | float) and isinstance(peak_score, int | float):
        px, py = x_of(float(peak_sec)), y_of(float(peak_score))
        status = str((verdict.get("peakBand") or {}).get("status") or "critical")
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{STATUS_FILL.get(status, SERIES_2)}" '
            f'stroke="#fff" stroke-width="1"/>'
        )
        anchor = "end" if px > right - 60 else "start"
        offset = -6 if anchor == "end" else 6
        parts.append(
            f'<text x="{px + offset:.1f}" y="{py - 7:.1f}" font-size="8.5" font-weight="600" '
            f'fill="{INK}" text-anchor="{anchor}">peak {int(peak_score)}</text>'
        )

    # Time axis, labelled in minutes and seconds as the workspace does.
    parts.append(
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{AXIS}" stroke-width="1"/>'
    )
    ticks = 6
    for index in range(ticks + 1):
        seconds = max_t * index / ticks
        x = x_of(seconds)
        parts.append(
            f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom + 4}" '
            f'stroke="{AXIS}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{bottom + 15}" font-size="8" fill="{INK_MUTED}" '
            f'text-anchor="middle">{int(seconds) // 60}:{int(seconds) % 60:02d}</text>'
        )
    parts.append(
        f'<text x="{left}" y="{bottom + 30}" font-size="8" fill="{INK_MUTED}">'
        f"Elapsed time · dashed line marks the action threshold</text>"
    )

    return _svg("".join(parts), bottom + 36, f"{method} score over time")


# ---------------------------------------------------------------------------
# Exposure by action level
# ---------------------------------------------------------------------------


def _arc(cx: float, cy: float, outer: float, inner: float, start: float, end: float) -> str:
    """One donut segment as a filled path."""

    large = 1 if (end - start) % 360 > 180 else 0
    s, e = math.radians(start - 90), math.radians(end - 90)
    x1, y1 = cx + outer * math.cos(s), cy + outer * math.sin(s)
    x2, y2 = cx + outer * math.cos(e), cy + outer * math.sin(e)
    x3, y3 = cx + inner * math.cos(e), cy + inner * math.sin(e)
    x4, y4 = cx + inner * math.cos(s), cy + inner * math.sin(s)
    return (
        f"M{x1:.2f},{y1:.2f} A{outer},{outer} 0 {large} 1 {x2:.2f},{y2:.2f} "
        f"L{x3:.2f},{y3:.2f} A{inner},{inner} 0 {large} 0 {x4:.2f},{y4:.2f} Z"
    )


def exposure_donut(timeline: dict[str, Any]) -> Markup | None:
    """Share of scored time spent in each risk band."""

    shares = [s for s in (timeline.get("shares") or []) if float(s.get("percent") or 0) > 0]
    if not shares:
        return None

    verdict = timeline.get("verdict") or {}
    percent_at_action = verdict.get("percentAtAction")

    cx, cy, outer, inner = 96, 100, 74, 48
    parts: list[str] = []
    cursor = 0.0

    for share in shares:
        percent = float(share["percent"])
        sweep = percent / 100 * 360
        fill = STATUS_FILL.get(str(share.get("status")), AXIS)
        # A full ring has no arc endpoints; draw it as two rings instead.
        if sweep >= 359.9:
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{(outer + inner) / 2}" fill="none" '
                         f'stroke="{fill}" stroke-width="{outer - inner}"/>')
        else:
            parts.append(f'<path d="{_arc(cx, cy, outer, inner, cursor, cursor + sweep)}" '
                         f'fill="{fill}"/>')
        cursor += sweep

    if isinstance(percent_at_action, int | float):
        parts.append(
            f'<text x="{cx}" y="{cy - 2}" font-size="22" font-weight="700" fill="{INK}" '
            f'text-anchor="middle">{_round(float(percent_at_action), 1)}%</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{cy + 14}" font-size="8" fill="{INK_MUTED}" '
            f'text-anchor="middle">at action level</text>'
        )

    # Legend: band, share of time, and how long that actually was.
    legend_x, row_y = 210, 34
    percent_x, clock_x = WIDTH - 92, WIDTH
    parts.append(
        f'<text x="{percent_x}" y="{row_y - 16}" font-size="8" fill="{INK_MUTED}" '
        f'text-anchor="end">share</text>'
    )
    parts.append(
        f'<text x="{clock_x}" y="{row_y - 16}" font-size="8" fill="{INK_MUTED}" '
        f'text-anchor="end">time</text>'
    )
    for share in shares:
        fill = STATUS_FILL.get(str(share.get("status")), AXIS)
        seconds = float(share.get("durationSec") or 0)
        clock = f"{int(seconds) // 60}:{int(seconds) % 60:02d}"
        parts.append(f'<rect x="{legend_x}" y="{row_y - 8}" width="9" height="9" rx="2" fill="{fill}"/>')
        parts.append(
            f'<text x="{legend_x + 16}" y="{row_y}" font-size="9.5" fill="{INK}">'
            f'{_text(share.get("label"))} <tspan fill="{INK_MUTED}">'
            f'({_text(share.get("range"))})</tspan></text>'
        )
        parts.append(
            f'<text x="{percent_x}" y="{row_y}" font-size="9.5" font-weight="600" '
            f'fill="{INK}" text-anchor="end">{_round(float(share["percent"]), 1)}%</text>'
        )
        parts.append(
            f'<text x="{clock_x}" y="{row_y}" font-size="9" fill="{INK_MUTED}" '
            f'text-anchor="end">{clock}</text>'
        )
        row_y += 22

    height = max(196, row_y + 8)
    return _svg("".join(parts), height, "Share of scored time by risk band")


# ---------------------------------------------------------------------------
# What drove the score
# ---------------------------------------------------------------------------


def driver_bars(timeline: dict[str, Any], limit: int = 7) -> Markup | None:
    """Points each body region contributed, typically and at its worst.

    Two bars per region rather than one: a region that is mildly raised for the
    whole task and one that spikes once are different problems with different
    fixes, and a single mean hides which of the two is on the page.
    """

    loads = [
        load
        for load in (timeline.get("posture") or [])
        if float(load.get("worstPoints") or 0) > 0 or float(load.get("meanPoints") or 0) > 0
    ]
    if not loads:
        return None

    loads = sorted(loads, key=lambda item: float(item.get("worstPoints") or 0), reverse=True)[:limit]
    maximum = max(
        [float(load.get("worstPoints") or 0) for load in loads]
        + [float(load.get("meanPoints") or 0) for load in loads]
        + [0.001]
    )

    label_w, value_w = 108, 92
    track_x = label_w + 8
    track_w = WIDTH - track_x - value_w
    row_h, bar_h, gap = 30, 9, 3
    top = 26

    parts = [
        f'<rect x="{track_x}" y="6" width="9" height="9" rx="2" fill="{SERIES_1}"/>',
        f'<text x="{track_x + 14}" y="14" font-size="8.5" fill="{INK_MUTED}">Typical</text>',
        f'<rect x="{track_x + 62}" y="6" width="9" height="9" rx="2" fill="{SERIES_2}"/>',
        f'<text x="{track_x + 76}" y="14" font-size="8.5" fill="{INK_MUTED}">Peak</text>',
        f'<text x="{WIDTH}" y="14" font-size="8" fill="{INK_MUTED}" text-anchor="end">'
        f"points of score removed if the region were neutral</text>",
    ]

    for index, load in enumerate(loads):
        y = top + index * row_h
        mean_points = float(load.get("meanPoints") or 0)
        worst_points = float(load.get("worstPoints") or 0)

        parts.append(
            f'<text x="0" y="{y + 11}" font-size="9.5" fill="{INK}">{_text(load.get("label"))}</text>'
        )
        for offset, value, colour in (
            (0, mean_points, SERIES_1),
            (bar_h + gap, worst_points, SERIES_2),
        ):
            parts.append(
                f'<rect x="{track_x}" y="{y + offset}" width="{track_w}" height="{bar_h}" '
                f'rx="2" fill="#f2f2ef"/>'
            )
            width = max(0.0, value / maximum * track_w)
            if width > 0:
                parts.append(
                    f'<rect x="{track_x}" y="{y + offset}" width="{width:.1f}" height="{bar_h}" '
                    f'rx="2" fill="{colour}"/>'
                )

        percent_raised = load.get("percentRaised")
        raised = (
            f"{_round(float(percent_raised), 0)}% of task"
            if isinstance(percent_raised, int | float)
            else ""
        )
        parts.append(
            f'<text x="{WIDTH}" y="{y + 9}" font-size="9.5" font-weight="600" fill="{INK}" '
            f'text-anchor="end">+{_round(worst_points, 1)}</text>'
        )
        parts.append(
            f'<text x="{WIDTH}" y="{y + 21}" font-size="8" fill="{INK_MUTED}" '
            f'text-anchor="end">{_text(raised)}</text>'
        )

    return _svg("".join(parts), top + len(loads) * row_h + 4, "Body region contribution to the score")


# ---------------------------------------------------------------------------
# NIOSH multipliers
# ---------------------------------------------------------------------------


def multiplier_bars(multipliers: dict[str, Any]) -> Markup | None:
    """Each RNLE multiplier against its ceiling of 1.0.

    A multiplier is a penalty: the shortfall below 1.0 is the part of the load
    limit that task geometry took away, so the bar is read from the top down.
    """

    named = (
        ("HM", "Horizontal"),
        ("VM", "Vertical"),
        ("DM", "Distance"),
        ("AM", "Asymmetry"),
        ("FM", "Frequency"),
        ("CM", "Coupling"),
    )
    values = [
        (key, label, float(multipliers[key]))
        for key, label in named
        if isinstance(multipliers.get(key), int | float)
    ]
    if not values:
        return None

    top, plot_h = 16, 96
    slot = WIDTH / len(values)
    bar_w = min(58.0, slot - 26)
    parts = [
        f'<line x1="0" y1="{top}" x2="{WIDTH}" y2="{top}" stroke="{GRID}" stroke-width="1" '
        f'stroke-dasharray="3 3"/>',
        f'<text x="0" y="{top - 5}" font-size="8" fill="{INK_MUTED}">1.0 (no penalty)</text>',
    ]

    for index, (key, label, value) in enumerate(values):
        centre = slot * (index + 0.5)
        x = centre - bar_w / 2
        height = max(1.0, min(1.0, value) * plot_h)
        y = top + plot_h - height
        # Below 0.6 the multiplier alone has more than halved the limit.
        colour = SERIES_1 if value >= 0.75 else (STATUS_FILL["warning"] if value >= 0.6 else STATUS_FILL["serious"])
        parts.append(
            f'<rect x="{x:.1f}" y="{top}" width="{bar_w:.1f}" height="{plot_h}" rx="3" fill="#f2f2ef"/>'
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height:.1f}" rx="3" fill="{colour}"/>'
        )
        parts.append(
            f'<text x="{centre:.1f}" y="{y - 5:.1f}" font-size="9.5" font-weight="600" fill="{INK}" '
            f'text-anchor="middle">{value:.3f}</text>'
        )
        parts.append(
            f'<text x="{centre:.1f}" y="{top + plot_h + 14}" font-size="9" font-weight="600" '
            f'fill="{INK}" text-anchor="middle">{key}</text>'
        )
        parts.append(
            f'<text x="{centre:.1f}" y="{top + plot_h + 25}" font-size="8" fill="{INK_MUTED}" '
            f'text-anchor="middle">{_text(label)}</text>'
        )

    return _svg("".join(parts), top + plot_h + 32, "NIOSH multipliers against their ceiling")


__all__ = ["driver_bars", "exposure_donut", "multiplier_bars", "score_timeline"]
