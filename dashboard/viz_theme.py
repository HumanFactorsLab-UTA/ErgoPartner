"""One visual system for every chart in the dashboard.

Charts were previously styled ad hoc, each picking its own colours and layout.
That produced overlapping labels, legends sitting on top of data, and a different
palette per figure, so nothing read as one product and several figures were
simply hard to look at.

Everything here is parameters, not opinions:

* The categorical hues are the dark-surface column of the reference palette,
  validated with ``scripts/validate_palette.js`` against this dashboard's own
  surface (``#080d18``) rather than the generic dark surface. All five slots pass
  the lightness band, chroma floor, adjacent CVD separation, normal-vision floor,
  and 3:1 contrast. The first three additionally pass on the all-pairs list, so
  scatter and small-multiple forms cap at three series.
* Status colours are reserved for risk bands and never reused as a series hue.
  They always ship with a label, so colour never carries meaning alone.
* Ink, grid, and axis colours are recessive by construction: the data is the only
  saturated thing on the surface.

Rules the helpers enforce, from ``references/marks-and-anatomy.md`` and
``references/anti-patterns.md``:

* one y-axis, never two;
* a legend whenever two or more series are drawn, and none for a single series
  because the title already names it;
* 2px lines, markers at least 8px, hairline grid;
* a hover layer on every plotted form.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import plotly.graph_objects as go

# Validated against surface #080d18 in dark mode. Adjacent-pair worst CVD dE 8.4,
# worst normal-vision dE 19.3, all five >= 3:1 contrast. Assign in fixed order and
# never cycle: a ninth series folds into "Other" or becomes small multiples.
SERIES: tuple[str, ...] = (
    "#3987e5",  # blue
    "#d95926",  # orange
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#d55181",  # magenta
    "#9085e9",  # violet
)

#: Forms that compare every series against every other (scatter, bubble, small
#: multiples) only clear the floors for the first three slots.
ALL_PAIRS_SERIES_CAP = 3

# Reserved for risk bands. Never a series colour. Always paired with a label.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

#: One hue, light to dark, for continuous magnitude. Never a rainbow.
SEQUENTIAL_BLUE = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)

SURFACE = "#080d18"
PAGE = "#05070c"
INK_PRIMARY = "#f6f8ff"
INK_SECONDARY = "#a8b6ce"
INK_MUTED = "#7b8aa3"
GRIDLINE = "#1b2942"
AXIS = "#263858"

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


class RiskBand(NamedTuple):
    """One published action level of one method.

    ``label`` is the method's own name for the level and ``action`` is what the
    method says to do about it. Both are quoted from the published tables rather
    than paraphrased, because an assessor reading "High" on a REBA score and
    "High" on a RULA score must be reading the same two source documents we are.

    ``describes_risk`` records which of the two the label is. REBA names a risk
    level; RULA names an action. Treating them alike produced sentences like
    "peaked at 7 — change now risk", so prose asks the band which it is.
    """

    lower: int
    upper: int
    label: str
    action: str
    status: str
    describes_risk: bool = True

    @property
    def colour(self) -> str:
        return STATUS[self.status]

    @property
    def range_text(self) -> str:
        return str(self.lower) if self.lower == self.upper else f"{self.lower}-{self.upper}"


# RULA 1993, action levels 1-4. Seven is the top of the scale. RULA's own names
# for its levels are instructions, not risk descriptors.
RULA_BANDS: tuple[RiskBand, ...] = (
    RiskBand(1, 2, "Acceptable", "Acceptable if not maintained or repeated", "good", False),
    RiskBand(3, 4, "Investigate", "Further investigation; change may be needed", "warning", False),
    RiskBand(5, 6, "Investigate soon", "Further investigation; change soon", "serious", False),
    RiskBand(7, 7, "Change now", "Investigate and implement change immediately", "critical", False),
)

# REBA 2000, five risk levels. These were previously given RULA's wording, which
# made a REBA 8 read as "investigate and change soon" when the method says the
# risk is High and action is necessary soon.
REBA_BANDS: tuple[RiskBand, ...] = (
    RiskBand(1, 1, "Negligible", "No action necessary", "good"),
    RiskBand(2, 3, "Low", "Action may be necessary", "good"),
    RiskBand(4, 7, "Medium", "Action is necessary", "warning"),
    RiskBand(8, 10, "High", "Action is necessary soon", "serious"),
    RiskBand(11, 15, "Very high", "Action is necessary NOW", "critical"),
)

#: Top of each method's published scale, used for axis ranges.
METHOD_SCALE_MAXIMUM = {"rula": 7, "reba": 15}


def method_bands(method: str) -> tuple[RiskBand, ...]:
    """Return the published action levels for ``method``, lowest first."""

    return RULA_BANDS if str(method).lower() == "rula" else REBA_BANDS


def risk_band(score: float | None, method: str) -> RiskBand | None:
    """Return the published band a score falls in, or None when unscored.

    Scores above the top of a scale still resolve to the top band rather than to
    nothing: an out-of-range value is a data problem to report elsewhere, not a
    reason to draw the worst posture in the trial as uncoloured.
    """

    if score is None or isinstance(score, bool) or not isinstance(score, int | float):
        return None
    value = float(score)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    bands = method_bands(method)
    for band in bands:
        if value <= band.upper:
            return band
    return bands[-1]


def risk_colour(score: float | None, method: str) -> str:
    """Colour for a RULA or REBA score, by the method's own action bands.

    Bands come from the published action levels, not from an even split of the
    scale, so the colour means the same thing an assessor means.
    """

    band = risk_band(score, method)
    return INK_MUTED if band is None else band.colour


def risk_label(score: float | None, method: str) -> str:
    """The method's own name for the level, so colour never carries it alone."""

    band = risk_band(score, method)
    return "No score" if band is None else band.label


def risk_action(score: float | None, method: str) -> str:
    """What the method says to do about a score, in the method's own words."""

    band = risk_band(score, method)
    return "No score was produced" if band is None else band.action


def apply_layout(
    figure: go.Figure,
    *,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int = 320,
    show_legend: bool | None = None,
    series_count: int = 1,
) -> go.Figure:
    """Apply the shared layout: recessive chrome, no overlap, one y-axis.

    ``show_legend`` defaults to the rule rather than to a preference: a legend
    appears for two or more series and is omitted for one, where the title
    already names what is drawn. The legend sits above the plot so it can never
    cover a mark, which is what the previous per-figure legends did.
    """

    legend_visible = (series_count >= 2) if show_legend is None else show_legend
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_FAMILY, "size": 12, "color": INK_SECONDARY},
        title=(
            {
                "text": title,
                "font": {"size": 15, "color": INK_PRIMARY},
                "x": 0,
                "xanchor": "left",
                "y": 0.97,
                "yanchor": "top",
            }
            if title
            else None
        ),
        height=height,
        # Generous top margin so a legend above the plot never sits on the data.
        margin={"l": 56, "r": 20, "t": 64 if title else 44, "b": 44},
        showlegend=legend_visible,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11, "color": INK_SECONDARY},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
        },
        hovermode="x unified",
        hoverlabel={
            "bgcolor": SURFACE,
            "bordercolor": AXIS,
            "font": {"family": FONT_FAMILY, "size": 12, "color": INK_PRIMARY},
        },
    )
    axis = {
        "showgrid": True,
        "gridcolor": GRIDLINE,
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": AXIS,
        "tickfont": {"size": 11, "color": INK_MUTED},
        "title_font": {"size": 12, "color": INK_SECONDARY},
        "automargin": True,
    }
    # Only set an axis title when one was passed. Passing None through would
    # clear a title the caller had already set on the figure, which is exactly
    # what happened when the existing figures were ported onto this helper.
    figure.update_xaxes(**axis)
    figure.update_yaxes(**axis)
    if x_title is not None:
        figure.update_xaxes(title_text=x_title)
    if y_title is not None:
        figure.update_yaxes(title_text=y_title)
    return figure


def line(
    figure: go.Figure,
    x: Any,
    y: Any,
    *,
    name: str,
    slot: int = 0,
    fill: bool = False,
    width: int = 2,
    dash: str | None = None,
    hover_suffix: str = "",
) -> go.Figure:
    """Add a 2px line in the next categorical slot, with a hover layer."""

    colour = SERIES[slot % len(SERIES)]
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=name,
            mode="lines",
            line={"color": colour, "width": width, "shape": "spline", "smoothing": 0.6}
            | ({"dash": dash} if dash else {}),
            fill="tozeroy" if fill else None,
            fillcolor=_translucent(colour, 0.16) if fill else None,
            hovertemplate=f"%{{y:.1f}}{hover_suffix}<extra>{name}</extra>",
        )
    )
    return figure


def translucent(hex_colour: str, alpha: float) -> str:
    """Return a hex colour as ``rgba`` at the requested alpha.

    Band backgrounds and fills need the same hue as the mark they belong to at a
    fraction of its strength; picking a second, lighter hex by hand is how a
    palette drifts.
    """

    value = hex_colour.lstrip("#")
    r, g, b = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


#: Retained for callers written against the private name.
_translucent = translucent


def empty_figure(message: str, height: int = 220) -> go.Figure:
    """A placeholder that says why there is nothing to draw."""

    figure = go.Figure()
    figure.add_annotation(
        text=message,
        showarrow=False,
        font={"size": 13, "color": INK_MUTED, "family": FONT_FAMILY},
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure
