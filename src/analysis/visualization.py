"""Plotly figure builders shared by the dashboard pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def landmarks_3d_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    fig = px.scatter_3d(frame, x="x", y="y", z="z", title=title, color="z", color_continuous_scale="Viridis")
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=50, b=0), scene_aspectmode="data")
    return fig


def line_figure(df: pd.DataFrame, x: str, ys: list[str], title: str, y_label: str = "") -> go.Figure:
    melted = df[[x] + ys].melt(id_vars=x, var_name="Metric", value_name="Value")
    fig = px.line(melted, x=x, y="Value", color="Metric", title=title)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), yaxis_title=y_label or None)
    return fig


def blink_timeline_figure(table: pd.DataFrame, blinks: pd.DataFrame, title: str = "Eye blink score") -> go.Figure:
    fig = line_figure(table, "time_s", ["eye_blink"], title, "Blink score")
    for _, event in blinks.iterrows():
        start = float(event["start_s"])
        end = start + float(event["duration_ms"]) / 1000.0
        fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.2, line_width=0)
    return fig


def distribution_figure(series_by_name: dict[str, pd.Series], title: str, x_label: str) -> go.Figure:
    fig = go.Figure()
    for name, values in series_by_name.items():
        clean = pd.Series(values).dropna()
        fig.add_trace(go.Histogram(x=clean, name=name, opacity=0.55, histnorm="probability"))
    fig.update_layout(
        barmode="overlay",
        title=title,
        xaxis_title=x_label,
        yaxis_title="Share of frames",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def box_figure(series_by_name: dict[str, pd.Series], title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    for name, values in series_by_name.items():
        fig.add_trace(go.Box(y=pd.Series(values).dropna(), name=name, boxmean=True))
    fig.update_layout(title=title, yaxis_title=y_label, height=360, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def bar_figure(values: pd.Series, title: str, x_label: str = "") -> go.Figure:
    fig = px.bar(x=values.index, y=values.values, title=title, labels={"x": x_label, "y": ""})
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    return fig
