"""
Chart Builders - Plotly chart functions for the dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from constants import (
    CAMERA_ROLE_COLORS,
    REBA_RISK_BANDS,
    RULA_RISK_BANDS,
)


def score_timeline(
    df: pd.DataFrame, score_col: str, tool: str = "REBA", time_col: str = "timestamp_sec"
) -> go.Figure:
    """
    Line chart of score over time with risk-level background bands.
    """
    if time_col not in df.columns or score_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Data not available", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5
        )
        return fig

    bands = REBA_RISK_BANDS if tool.upper() == "REBA" else RULA_RISK_BANDS
    max_time = df[time_col].max()

    fig = go.Figure()

    # Add risk-level background bands
    for low, high, label, color in bands:
        fig.add_shape(
            type="rect",
            x0=0,
            x1=max_time,
            y0=low - 0.5,
            y1=high + 0.5,
            fillcolor=color,
            opacity=0.15,
            line_width=0,
            layer="below",
        )
        fig.add_annotation(
            x=max_time * 1.02,
            y=(low + high) / 2,
            text=label,
            showarrow=False,
            font=dict(size=9, color=color),
            xanchor="left",
        )

    # Score line
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[score_col],
            mode="lines",
            line=dict(color="#3498db", width=1.5),
            name=f"{tool} Score",
            hovertemplate=f"Time: %{{x:.1f}}s<br>{tool}: %{{y}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{tool} Score Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title=f"{tool} Score",
        yaxis=dict(range=[0, max(bands, key=lambda b: b[1])[1] + 1]),
        margin=dict(r=100),
        height=400,
        showlegend=False,
    )

    return fig


def risk_distribution_pie(scores: pd.Series, tool: str = "REBA") -> go.Figure:
    """
    Donut chart showing percentage of frames in each risk category.
    """
    bands = REBA_RISK_BANDS if tool.upper() == "REBA" else RULA_RISK_BANDS
    labels = []
    values = []
    colors = []

    for low, high, label, color in bands:
        count = ((scores >= low) & (scores <= high)).sum()
        if count > 0:
            labels.append(label)
            values.append(count)
            colors.append(color)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(colors=colors),
                textinfo="percent+label",
                hovertemplate="%{label}: %{value} frames (%{percent})<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=f"{tool} Risk Distribution",
        showlegend=True,
        height=350,
    )

    return fig


def body_segment_radar(
    segment_means: dict[str, float], title: str = "Mean Body Segment Scores"
) -> go.Figure:
    """
    Radar chart showing mean scores for each body segment.
    segment_means: dict with keys like Neck, Trunk, Legs, Upper Arm, Lower Arm, Wrist.
    """
    categories = list(segment_means.keys())
    values = list(segment_means.values())
    # Close the radar
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="rgba(52, 152, 219, 0.3)",
            line=dict(color="#3498db", width=2),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(values) + 1]),
        ),
        title=title,
        showlegend=False,
        height=350,
    )

    return fig


def niosh_multiplier_bar(multipliers: dict[str, float]) -> go.Figure:
    """
    Horizontal bar chart showing NIOSH multiplier values (HM, VM, DM, AM, FM, CM).
    Values range 0-1, with 1 being ideal.
    """
    names = list(multipliers.keys())
    values = list(multipliers.values())

    # Color: green if >=0.8, yellow if >=0.5, red if <0.5
    colors = []
    for v in values:
        if v >= 0.8:
            colors.append("#2ecc71")
        elif v >= 0.5:
            colors.append("#f1c40f")
        else:
            colors.append("#e74c3c")

    full_names = {
        "HM": "Horizontal (HM)",
        "VM": "Vertical (VM)",
        "DM": "Distance (DM)",
        "AM": "Asymmetry (AM)",
        "FM": "Frequency (FM)",
        "CM": "Coupling (CM)",
    }

    display_names = [full_names.get(n, n) for n in names]

    fig = go.Figure(
        data=go.Bar(
            x=values,
            y=display_names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="NIOSH Multiplier Factors",
        xaxis_title="Multiplier Value",
        xaxis=dict(range=[0, 1.15]),
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=120),
    )

    return fig


def comparison_box_plot(
    df: pd.DataFrame, score_col: str, group_col: str = "participant_id", title: str | None = None
) -> go.Figure:
    """
    Box plots comparing score distributions across participants.
    """
    fig = px.box(
        df,
        x=group_col,
        y=score_col,
        color=group_col,
        title=title or f"{score_col} Distribution by {group_col}",
    )

    fig.update_layout(
        xaxis_title="Participant",
        yaxis_title=score_col,
        showlegend=False,
        height=400,
    )

    return fig


def comparison_grouped_bar(
    summary_df: pd.DataFrame,
    score_col_mean: str,
    score_col_max: str,
    group_col: str = "participant_id",
    title: str | None = None,
) -> go.Figure:
    """
    Grouped bar chart comparing mean and max scores across participants.
    """
    participants = summary_df[group_col].tolist()

    fig = go.Figure(
        data=[
            go.Bar(
                name="Mean",
                x=participants,
                y=summary_df[score_col_mean],
                marker_color="#1abc9c",
                text=[f"{v:.1f}" if pd.notna(v) else "" for v in summary_df[score_col_mean]],
                textposition="outside",
            ),
            go.Bar(
                name="Max",
                x=participants,
                y=summary_df[score_col_max],
                marker_color="#9b59b6",
                text=[f"{v:.0f}" if pd.notna(v) else "" for v in summary_df[score_col_max]],
                textposition="outside",
            ),
        ]
    )

    fig.update_layout(
        barmode="group",
        title=title or "Score Comparison",
        xaxis_title="Participant",
        yaxis_title="Score",
        height=400,
    )

    return fig


def demographic_scatter(
    summary_df: pd.DataFrame,
    x_col: str,
    score_col: str,
    color_col: str = "participant_id",
    title: str | None = None,
) -> go.Figure:
    """
    Scatter plot of a demographic variable vs score.
    """
    filtered = summary_df.dropna(subset=[x_col, score_col])
    if filtered.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5
        )
        return fig

    fig = px.scatter(
        filtered,
        x=x_col,
        y=score_col,
        color=color_col,
        title=title or f"{score_col} vs {x_col}",
        hover_data=["participant_id", "task_name"],
    )

    fig.update_layout(height=400)
    return fig


def per_camera_angle_comparison(
    per_cam_df: pd.DataFrame, angle_col: str = "trunk_flexion_deg", time_col: str = "timestamp_sec"
) -> go.Figure:
    """
    Overlay line chart showing the same angle from each camera view.
    per_cam_df must have columns: timestamp_sec, camera_role, and the angle column.
    """
    fig = go.Figure()

    for role in sorted(per_cam_df["camera_role"].unique()):
        cam_data = per_cam_df[per_cam_df["camera_role"] == role]
        color = CAMERA_ROLE_COLORS.get(role, "#95a5a6")
        fig.add_trace(
            go.Scatter(
                x=cam_data[time_col],
                y=cam_data[angle_col],
                mode="lines",
                name=f"{role.capitalize()} Camera",
                line=dict(color=color, width=1.5),
                hovertemplate=f"{role}: %{{y:.1f}} at %{{x:.1f}}s<extra></extra>",
            )
        )

    angle_label = angle_col.replace("_deg", "").replace("_", " ").title()
    fig.update_layout(
        title=f"{angle_label} - Per Camera View",
        xaxis_title="Time (seconds)",
        yaxis_title=f"{angle_label} (degrees)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def camera_visibility_timeline(
    per_cam_df: pd.DataFrame, time_col: str = "timestamp_sec"
) -> go.Figure:
    """
    Line chart showing average landmark visibility per camera over time.
    """
    fig = go.Figure()

    for role in sorted(per_cam_df["camera_role"].unique()):
        cam_data = per_cam_df[per_cam_df["camera_role"] == role]
        color = CAMERA_ROLE_COLORS.get(role, "#95a5a6")
        fig.add_trace(
            go.Scatter(
                x=cam_data[time_col],
                y=cam_data["avg_visibility"],
                mode="lines",
                name=f"{role.capitalize()}",
                line=dict(color=color, width=1.5),
            )
        )

    fig.update_layout(
        title="Landmark Visibility by Camera",
        xaxis_title="Time (seconds)",
        yaxis_title="Average Visibility",
        yaxis=dict(range=[0, 1.05]),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def best_view_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Stacked bar showing which camera was selected as best view
    for each angle group across all frames.
    """
    group_cols = [c for c in df.columns if c.startswith("best_view_")]
    if not group_cols:
        fig = go.Figure()
        fig.add_annotation(
            text="No multi-view data", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5
        )
        return fig

    # Count occurrences per group per camera
    data = {}
    for col in group_cols:
        counts = df[col].value_counts()
        label = col.replace("best_view_", "").replace("_", " ").title()
        data[label] = counts.to_dict()

    all_cameras = set()
    for counts in data.values():
        all_cameras.update(counts.keys())
    all_cameras = sorted(all_cameras)

    fig = go.Figure()
    # Build a color map that covers both role names ("front"/"side"/"iso")
    # and generic camera indices ("cam_0"/"cam_1"/"cam_2").
    _color_palette = [
        "#e74c3c",
        "#3498db",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
        "#d35400",
    ]
    _cam_color_map = dict(CAMERA_ROLE_COLORS)
    for i, color in enumerate(_color_palette):
        key = f"cam_{i}"
        if key not in _cam_color_map:
            _cam_color_map[key] = color

    for cam in all_cameras:
        values = [data[g].get(cam, 0) for g in data.keys()]
        fig.add_trace(
            go.Bar(
                name=cam.capitalize(),
                x=list(data.keys()),
                y=values,
                marker_color=_cam_color_map.get(cam, "#95a5a6"),
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Camera Selection per Angle Group",
        xaxis_title="Angle Group",
        yaxis_title="Frames Selected",
        height=350,
    )

    return fig
