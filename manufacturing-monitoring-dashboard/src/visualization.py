"""Reusable Plotly chart builders for the manufacturing dashboard."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.logger import get_logger

logger = get_logger(__name__)

_MACHINE_COLORS = {
    "M-001": "#2ecc71",
    "M-002": "#3498db",
    "M-003": "#e74c3c",
    "M-004": "#e67e22",
    "M-005": "#9b59b6",
    "M-006": "#1abc9c",
    "M-007": "#e91e63",
}

_OEE_PALETTE = {"oee": "#2196F3", "availability": "#4CAF50", "performance": "#FF9800", "quality": "#9C27B0"}


def create_oee_factor_chart(oee_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart showing OEE and its three factors per machine.

    Args:
        oee_df: DataFrame with machine_id, oee, availability, performance, quality.

    Returns:
        Plotly Figure.
    """
    if oee_df.empty:
        return _empty_figure("No OEE data available.")

    df = oee_df.copy()
    for col in ["oee", "availability", "performance", "quality"]:
        if col in df.columns:
            df[col] = (df[col] * 100).round(1)

    fig = go.Figure()
    for col, color in [
        ("oee", _OEE_PALETTE["oee"]),
        ("availability", _OEE_PALETTE["availability"]),
        ("performance", _OEE_PALETTE["performance"]),
        ("quality", _OEE_PALETTE["quality"]),
    ]:
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Bar(
                name=col.capitalize(),
                x=df["machine_id"],
                y=df[col],
                marker_color=color,
                text=df[col].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            )
        )

    fig.update_layout(
        barmode="group",
        title="OEE Factor Decomposition by Machine",
        xaxis_title="Machine",
        yaxis_title="Percentage (%)",
        yaxis=dict(range=[0, 115]),
        legend_title="Factor",
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_oee_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    """Line chart of daily OEE per machine over the trend window.

    Args:
        trend_df: DataFrame with date, machine_id, oee columns.

    Returns:
        Plotly Figure.
    """
    if trend_df.empty:
        return _empty_figure("No OEE trend data available.")

    df = trend_df.copy()
    df["oee_pct"] = (df["oee"] * 100).round(1)

    fig = px.line(
        df,
        x="date",
        y="oee_pct",
        color="machine_id",
        color_discrete_map=_MACHINE_COLORS,
        markers=True,
        labels={"oee_pct": "OEE (%)", "date": "Date", "machine_id": "Machine"},
        title="7-Day OEE Trend",
    )
    fig.add_hline(y=85, line_dash="dash", line_color="green", annotation_text="Target 85%")
    fig.add_hline(y=70, line_dash="dot", line_color="orange", annotation_text="Watch 70%")
    fig.update_layout(
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 110]),
    )
    return fig


def create_sensor_anomaly_chart(
    df: pd.DataFrame,
    sensor: str,
    machine_id: Optional[str] = None,
) -> go.Figure:
    """Time-series chart of a sensor signal with anomaly points highlighted.

    Args:
        df: Sensor DataFrame with timestamp, the sensor column, and is_anomaly.
        sensor: Column name of the sensor to plot.
        machine_id: Optional machine label for the title.

    Returns:
        Plotly Figure.
    """
    if df.empty or sensor not in df.columns:
        return _empty_figure(f"No data available for sensor '{sensor}'.")

    normal = df[~df.get("is_anomaly", pd.Series(False, index=df.index))]
    anomaly = df[df.get("is_anomaly", pd.Series(False, index=df.index))]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=normal["timestamp"],
            y=normal[sensor],
            mode="lines",
            name="Normal",
            line=dict(color="#2196F3", width=1.5),
        )
    )
    if not anomaly.empty:
        fig.add_trace(
            go.Scatter(
                x=anomaly["timestamp"],
                y=anomaly[sensor],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#e74c3c", size=7, symbol="x"),
            )
        )

    label = machine_id or "Machine"
    fig.update_layout(
        title=f"{label} — {sensor.replace('_', ' ').title()} (24h)",
        xaxis_title="Time",
        yaxis_title=sensor.replace("_", " ").title(),
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_failure_probability_chart(summary_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of 7-day failure probability per machine.

    Args:
        summary_df: DataFrame with machine_id and failure_probability_7d.

    Returns:
        Plotly Figure.
    """
    if summary_df.empty or "failure_probability_7d" not in summary_df.columns:
        return _empty_figure("No failure probability data available.")

    df = summary_df.copy().sort_values("failure_probability_7d", ascending=True)
    df["prob_pct"] = (df["failure_probability_7d"] * 100).round(1)

    colors = [
        "#e74c3c" if p >= 70 else "#e67e22" if p >= 40 else "#2ecc71"
        for p in df["prob_pct"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df["prob_pct"],
            y=df["machine_id"],
            orientation="h",
            marker_color=colors,
            text=df["prob_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        )
    )
    fig.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="High Risk 70%")
    fig.add_vline(x=40, line_dash="dot", line_color="orange", annotation_text="Medium Risk 40%")
    fig.update_layout(
        title="7-Day Failure Probability by Machine",
        xaxis_title="Failure Probability (%)",
        xaxis=dict(range=[0, 110]),
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_health_score_chart(summary_df: pd.DataFrame) -> go.Figure:
    """Radial/bar chart of machine health scores.

    Args:
        summary_df: DataFrame with machine_id and health_score.

    Returns:
        Plotly Figure.
    """
    if summary_df.empty or "health_score" not in summary_df.columns:
        return _empty_figure("No health score data available.")

    df = summary_df.copy().sort_values("health_score", ascending=False)

    colors = [
        "#2ecc71" if s >= 80 else "#e67e22" if s >= 60 else "#e74c3c"
        for s in df["health_score"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df["machine_id"],
            y=df["health_score"],
            marker_color=colors,
            text=df["health_score"].apply(lambda v: f"{v:.1f}"),
            textposition="outside",
        )
    )
    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Healthy 80")
    fig.add_hline(y=60, line_dash="dot", line_color="orange", annotation_text="Warning 60")
    fig.update_layout(
        title="Machine Health Score",
        xaxis_title="Machine",
        yaxis_title="Health Score (0–100)",
        yaxis=dict(range=[0, 115]),
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    """Return a placeholder figure with a centered annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#888"),
    )
    fig.update_layout(
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
