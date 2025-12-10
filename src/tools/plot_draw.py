"""Plotting utilities exposed as LangChain tools."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI thread issues

from langchain_core.tools import tool


PLOTS_DIR = Path(__file__).resolve().parents[1] / "files"

# Global matplotlib configuration for Chinese character support
MATPLOTLIB_CHINESE_CONFIG = {
    'font.family': ['sans-serif', 'DejaVu Sans', 'PingFang HK', 'Hiragino Sans GB', 'Arial Unicode MS'],
    'font.sans-serif': ['PingFang HK', 'Hiragino Sans GB', 'Heiti TC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans'],
    'axes.unicode_minus': False,  # Use ASCII minus sign
    'svg.fonttype': 'none'  # Ensure text is rendered as text in SVG
}


def _ensure_output_path(output_path: Optional[str], prefix: str) -> Path:
    """Resolve an output path, generating one inside the repo's files/ directory when omitted."""
    if output_path:
        path = Path(output_path).expanduser()
    else:
        filename = f"{prefix}-{uuid.uuid4().hex}.png"
        path = PLOTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_mapping_from_file(data_file: str) -> Dict[str, float]:
    """Load a label->numeric mapping from a JSON file."""
    path = Path(data_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"data_file must contain a JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("data_file must contain a JSON object mapping labels to numeric values")

    mapping: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            mapping[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"data_file values must be numeric; failed on key {key!r}") from exc
    if not mapping:
        raise ValueError("data_file contains no data")
    return mapping


def _normalize_series(
    data_file: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    values: Optional[Sequence[float]] = None,
) -> Tuple[List[str], List[float]]:
    """Normalize mapping or parallel sequences into label/value lists."""
    if data_file is None and (labels is None or values is None):
        raise ValueError("provide either `data_file` path or both `labels` and `values`")
    if data_file is not None:
        data = _load_mapping_from_file(data_file)
        labels_list = list(data.keys())
        values_list = [float(v) for v in data.values()]
    else:
        labels_list = list(labels or [])
        values_list = [float(v) for v in values or []]
        if len(labels_list) != len(values_list):
            raise ValueError("`labels` and `values` must have the same length")
    if not labels_list:
        raise ValueError("no data provided for plotting")
    return labels_list, values_list


@tool
def heap_map(
    data_file: str,
    *,
    title: Optional[str] = None,
    cmap: str = "Blues",
    figsize: Tuple[float, float] = (10.0, 1.6),
    output_path: Optional[str] = None,
    **imshow_kwargs,
) -> str:
    """
    Plot a simple heat map for term frequencies and return the saved image path.

    Args:
        data_file: Path to a JSON file containing a mapping of label -> count (Dict[str, int]).
        title: Optional chart title.
        cmap: Matplotlib colormap name for the heat map, default is "Blues".
        figsize: Figure size passed to matplotlib, default is (10.0, 1.6).
        output_path: Optional path to save the image; when omitted, saves to files/.
        **imshow_kwargs: Forwarded to `Axes.imshow` for flexibility.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError("matplotlib is required for heap_map") from exc

    # Configure matplotlib for Chinese character support
    plt.rcParams.update(MATPLOTLIB_CHINESE_CONFIG)

    labels, values = _normalize_series(data_file=data_file)
    matrix = np.array([values])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, **imshow_kwargs)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks([])

    for x, val in enumerate(values):
        ax.text(x, 0, f"{val:g}", ha="center", va="center", color="black")

    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)

    path = _ensure_output_path(output_path, prefix="heatmap")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


@tool
def bar_chart(
    data_file: Optional[str] = None,
    *,
    labels: Optional[Iterable[str]] = None,
    values: Optional[Iterable[float]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    color: Optional[Iterable[str]] = None,
    figsize: Tuple[float, float] = (8.0, 4.0),
    rotation: float = 30.0,
    output_path: Optional[str] = None,
    **bar_kwargs,
) -> str:
    """
    Draw a bar chart and return the saved image path.

    Args mirror matplotlib's bar-related options where possible, forwarding extras
    via **bar_kwargs to `Axes.bar`. Provide either `data_file` (JSON object mapping
    labels to numbers) or both `labels` and `values`.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for bar_chart") from exc

    # Configure matplotlib for Chinese character support
    plt.rcParams.update(MATPLOTLIB_CHINESE_CONFIG)

    x_labels, y_values = _normalize_series(data_file=data_file, labels=labels, values=values)

    fig, ax = plt.subplots(figsize=figsize)

    # Handle color parameter - ensure it's a proper matplotlib color value
    if color is None:
        bar_color = None
    elif isinstance(color, str):
        bar_color = color
    elif hasattr(color, '__iter__') and not isinstance(color, (str, bytes)):
        try:
            bar_color = list(color)
        except TypeError:
            bar_color = None
    else:
        # Handle invalid color types (like ValidatorIterator)
        bar_color = None

    ax.bar(x_labels, y_values, color=bar_color, **bar_kwargs)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.tick_params(axis="x", labelrotation=rotation)

    path = _ensure_output_path(output_path, prefix="bar")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


@tool
def pie_chart(
    data_file: Optional[str] = None,
    *,
    labels: Optional[Iterable[str]] = None,
    values: Optional[Iterable[float]] = None,
    title: Optional[str] = None,
    autopct: str = "%.1f%%",
    startangle: float = 90.0,
    figsize: Tuple[float, float] = (6.0, 6.0),
    output_path: Optional[str] = None,
    **pie_kwargs,
) -> str:
    """
    Draw a pie chart and return the saved image path.

    Args mirror matplotlib's pie options where possible, forwarding extras via
    **pie_kwargs to `Axes.pie`. Provide either `data_file` (JSON object mapping
    labels to numbers) or both `labels` and `values`.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for pie_chart") from exc

    # Configure matplotlib for Chinese character support
    plt.rcParams.update(MATPLOTLIB_CHINESE_CONFIG)

    pie_labels, pie_values = _normalize_series(data_file=data_file, labels=labels, values=values)

    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(
        pie_values,
        labels=pie_labels,
        autopct=autopct,
        startangle=startangle,
        **pie_kwargs,
    )

    if title:
        ax.set_title(title)
    ax.axis("equal")

    # Improve legibility for tight layouts.
    plt.setp(autotexts, size=9)
    plt.setp(texts, size=10)

    path = _ensure_output_path(output_path, prefix="pie")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)
