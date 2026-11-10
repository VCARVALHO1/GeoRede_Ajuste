"""Shared helpers for analyzing standardized processing outputs."""

from __future__ import annotations

import logging
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except ModuleNotFoundError:
    plt = None
    mpl = None
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None
try:
    from scipy.linalg import svd
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
except ModuleNotFoundError:
    svd = np.linalg.svd
    dendrogram = None
    linkage = None
    squareform = None

# Optional 3D interactive plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None

# Optional statistical tests
try:
    from scipy import stats
except ModuleNotFoundError:
    stats = None

logger = logging.getLogger(__name__)


def setup_matplotlib_style():
    """Apply a consistent publication-friendly style to static plots."""
    if plt is None:
        return
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    rc_params = {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 14,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 1.35,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
    try:
        plt.rcParams.update(rc_params)
    except Exception:
        logger.debug("Could not apply matplotlib style", exc_info=True)
    if sns is not None:
        try:
            sns.set_palette("colorblind")
        except Exception:
            logger.debug("Could not apply seaborn palette", exc_info=True)


setup_matplotlib_style()

# ----------------------------------------------------------------------
# Method metadata
# ----------------------------------------------------------------------
METHOD_METADATA = {
    "SVD-P": {
        "article_name": "SVD-based Orthogonal Procrustes transformation",
        "reference": "Arun, Huang and Blostein (1987); Umeyama (1991).",
    },
    "SCH-Q": {
        "article_name": "Schut quaternion-based spatial similarity transformation",
        "reference": "Schut (1973); Horn (1987).",
    },
    "Q-ST": {
        "article_name": "Quasi-Similarity Transformation",
        "reference": "Stepien et al. (2020).",
    },
    "CMTLS": {
        "article_name": "Constrained Multivariate Total Least Squares for 3D similarity transformation",
        "reference": "Tao, Hua and Feng (2020).",
    },
    "TFS": {
        "article_name": "Total Free Station fixed-scale free-station transformation",
        "reference": "Stepien (2019).",
    },
    "N-CUQ-LM": {
        "article_name": "Network-constrained unit-quaternion Levenberg-Marquardt adjustment",
        "reference": "Horn (1987); Levenberg (1944); Marquardt (1963); fully coupled network extension.",
    },
}


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def method_code(method_or_code) -> str:
    """Extract method code string from object or string."""
    return str(getattr(method_or_code, "method_code", method_or_code))


def method_metadata(method_or_code) -> Dict:
    """Return metadata dict for a given method."""
    code = method_code(method_or_code)
    meta = dict(METHOD_METADATA.get(code, {}))
    if not meta and not isinstance(method_or_code, str):
        meta = {
            "article_name": getattr(method_or_code, "article_name", code),
            "reference": getattr(method_or_code, "reference", ""),
        }
    meta.setdefault("code", code)
    meta.setdefault("article_name", code)
    meta.setdefault("reference", "")
    return meta


# Standard output fields template
STANDARD_OUTPUT_FIELDS = {
    "method_code": "",
    "article_name": "",
    "scale": np.nan,
    "rotation_matrix": None,
    "translation": None,
    "transformed_control_points": None,
    "residuals_xyz": None,
    "residual_norms": None,
    "observation_residuals": None,
    "rmse": np.nan,
    "mean_residual": np.nan,
    "std_residual": np.nan,
    "max_residual": np.nan,
    "sigma0": np.nan,
    "sigma0_sq": np.nan,
    "degrees_of_freedom": np.nan,
    "dof": np.nan,
    "covariance": None,
    "correlation_matrix": None,
    "normal_matrix": None,
    "weight_matrix": None,
    "design_matrix": None,
    "elapsed_time_solver": np.nan,
    "elapsed_time_total": np.nan,
    "converged": None,
    "iterations": np.nan,
    "warnings": [],
}


def standardize_adjustment_outputs(
    method_or_code, outputs: Optional[Dict] = None
) -> Dict:
    """
    Standardize adjustment outputs to a consistent dictionary format.

    Parameters
    ----------
    method_or_code : str or object
        Method identifier.
    outputs : dict, optional
        Raw outputs to merge.

    Returns
    -------
    dict
        Standardized output dictionary.
    """
    standardized = dict(STANDARD_OUTPUT_FIELDS)
    meta = method_metadata(method_or_code)
    standardized["method_code"] = meta["code"]
    standardized["article_name"] = meta["article_name"]
    if outputs:
        standardized.update(outputs)

    # Ensure dof and degrees_of_freedom are consistent
    if np.isnan(standardized.get("degrees_of_freedom")) and not np.isnan(
        standardized.get("dof")
    ):
        standardized["degrees_of_freedom"] = standardized["dof"]
    if np.isnan(standardized.get("dof")) and not np.isnan(
        standardized.get("degrees_of_freedom")
    ):
        standardized["dof"] = standardized["degrees_of_freedom"]
    return standardized


def plot_error_ellipsoid(
    ax,
    center: Union[list, np.ndarray],
    covariance: np.ndarray,
    scale_factor: float = 1.0,
    color: str = "r",
    alpha: float = 0.3,
):
    """
    Plot a 3D covariance ellipsoid on a Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes3D
        The 3D axis to plot on.
    center : array-like of shape (3,)
        Center coordinates.
    covariance : ndarray of shape (3,3)
        Covariance matrix.
    scale_factor : float, optional
        Scaling factor for the ellipsoid radii (e.g., for confidence intervals).
    color : str, optional
        Color of the ellipsoid.
    alpha : float, optional
        Transparency level.
    """
    cov = np.asarray(covariance, dtype=float)
    u_matrix, s_vals, _ = svd(cov)
    radii = np.sqrt(np.clip(s_vals, 0.0, None)) * scale_factor

    # Parametric surface
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    center = np.asarray(center, dtype=float)
    for i in range(len(x)):
        for j in range(len(x[0])):
            rotated = np.dot([x[i, j], y[i, j], z[i, j]], u_matrix)
            x[i, j], y[i, j], z[i, j] = rotated + center

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def run_label_order(label) -> int:
    """Return sort order for run labels (outlier removal runs after initial)."""
    text = str(label)
    if text == "outliers_removed":
        return 1
    if text.startswith("outliers_removed_iter"):
        try:
            return int(text.replace("outliers_removed_iter", ""))
        except ValueError:
            return 1
    return 0


def select_method_summary(
    summary: pd.DataFrame, method_key: str, prefer_removed: bool = False
) -> Dict:
    """
    Return the display summary row for a method, respecting outlier runs.

    Parameters
    ----------
    summary : pd.DataFrame
        DataFrame containing method summaries.
    method_key : str
        Method identifier.
    prefer_removed : bool, optional
        If True, prefer outlier‑removed runs; otherwise prefer initial runs.

    Returns
    -------
    dict
        Selected summary row as dict, or empty dict if none found.
    """
    if summary is None or summary.empty or "method" not in summary.columns:
        return {}

    rows = summary[summary["method"].astype(str) == str(method_key)].copy()
    if rows.empty:
        return {}

    if prefer_removed and "run_label" in rows.columns:
        filtered = rows[
            rows["run_label"].astype(str).str.startswith("outliers_removed")
        ].copy()
        if not filtered.empty:
            filtered["__run_order"] = filtered["run_label"].map(run_label_order)
            rows = filtered.sort_values("__run_order")
    elif "run_label" in rows.columns:
        initial = rows[rows["run_label"].astype(str).isin({"", "initial", "network"})].copy()
        if not initial.empty:
            rows = initial

    return rows.iloc[-1].to_dict()


def load_method_summary(
    output_dir: Union[str, Path], method_key: str, prefer_removed: bool = False
) -> Dict:
    """
    Load method summary from CSV file and select appropriate row.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing method_summary.csv.
    method_key : str
        Method identifier.
    prefer_removed : bool, optional
        Passed to select_method_summary.

    Returns
    -------
    dict
        Selected summary row or empty dict.
    """
    summary_path = Path(output_dir) / "method_summary.csv"
    if not summary_path.exists():
        return {}
    try:
        summary = pd.read_csv(summary_path)
    except Exception as e:
        logger.warning(f"Failed to read summary CSV: {e}")
        return {}
    return select_method_summary(summary, method_key, prefer_removed=prefer_removed)


# ----------------------------------------------------------------------
# 3D HTML visualizations using Plotly
# ----------------------------------------------------------------------
def _apply_plotly_layout(
    fig,
    title: str,
    xlabel: str = "X (m)",
    ylabel: str = "Y (m)",
    zlabel: str = "Z (m)",
):
    """Apply a shared clean layout to interactive 3D outputs."""
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=14, family="Arial")),
        scene=dict(
            xaxis_title=dict(text=xlabel, font=dict(size=10)),
            yaxis_title=dict(text=ylabel, font=dict(size=10)),
            zaxis_title=dict(text=zlabel, font=dict(size=10)),
            aspectmode="data",
            bgcolor="white",
        ),
        legend=dict(
            title=dict(text="Legend", font=dict(size=10)),
            font=dict(size=9),
            x=1.02,
            xanchor="left",
            y=0.98,
            yanchor="top",
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(17,24,39,0.35)",
            borderwidth=0.5,
        ),
        margin=dict(l=135, r=220, b=0, t=44),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _plotly_left_colorbar(title: str):
    return dict(
        title=dict(text=title, font=dict(size=9), side="right"),
        x=-0.12,
        xanchor="right",
        y=0.5,
        yanchor="middle",
        len=0.72,
        thickness=18,
        outlinewidth=0.5,
        tickfont=dict(size=8),
    )


def _set_marker_colorbar(trace, title: str):
    marker = getattr(trace, "marker", None)
    colorbar = getattr(marker, "colorbar", None)
    if colorbar is not None:
        colorbar.update(_plotly_left_colorbar(title))


def create_3d_scatter_html(
    points: np.ndarray,
    residuals: Optional[np.ndarray] = None,
    colors: Optional[Union[str, List]] = None,
    title: str = "3D Point Cloud",
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """
    Create an interactive 3D scatter plot as HTML using Plotly.

    Parameters
    ----------
    points : ndarray of shape (n,3)
        3D coordinates.
    residuals : ndarray of shape (n,), optional
        Residual values for color mapping.
    colors : str or list, optional
        Fixed color or list of colors per point. Overrides residuals.
    title : str, optional
        Plot title.
    output_path : str or Path, optional
        If provided, save HTML to file. Otherwise return HTML string.

    Returns
    -------
    str or None
        Path to saved HTML file if output_path given, else None.
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not installed. Install with 'pip install plotly'")
        return None

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be (n,3) array")

    # Prepare color data
    color_data = None
    colorbar_title = ""
    if residuals is not None:
        residuals = np.asarray(residuals)
        color_data = residuals
        colorbar_title = "Residual (m)"
    elif colors is not None and isinstance(colors, str):
        # Use fixed color
        pass  # handled by marker_color

    # Create figure
    fig = go.Figure()

    # Scatter3d trace
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=5,
            color=color_data,
            colorscale="Viridis",
            colorbar=_plotly_left_colorbar(colorbar_title) if color_data is not None else None,
            opacity=0.8,
        ),
        text=[f"Point {i}" for i in range(len(points))] if residuals is None else [f"Resid: {r:.3f}" for r in residuals],
        hoverinfo="text+x+y+z",
    )
    if isinstance(colors, str):
        trace.marker.color = colors
    fig.add_trace(trace)

    _apply_plotly_layout(fig, title)
    if color_data is not None:
        _set_marker_colorbar(fig.data[0], colorbar_title)

    return _save_or_return_html(fig, output_path)


def create_3d_ellipsoid_html(
    center: np.ndarray,
    covariance: np.ndarray,
    scale_factor: float = 1.0,
    color: str = "rgba(255,0,0,0.5)",
    title: str = "Covariance Ellipsoid",
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """
    Create an interactive 3D ellipsoid from covariance matrix using Plotly mesh.

    Parameters
    ----------
    center : array-like of shape (3,)
        Center coordinates.
    covariance : ndarray of shape (3,3)
        Covariance matrix.
    scale_factor : float, optional
        Scaling factor for radii (e.g., 2 for 95% confidence).
    color : str, optional
        Color of the ellipsoid (CSS color or rgba).
    title : str, optional
        Plot title.
    output_path : str or Path, optional
        If provided, save HTML to file.

    Returns
    -------
    str or None
        Path to saved HTML file if output_path given, else None.
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not installed. Install with 'pip install plotly'")
        return None

    cov = np.asarray(covariance, dtype=float)
    center = np.asarray(center, dtype=float)

    # SVD to get ellipsoid axes
    U, S, _ = svd(cov)
    radii = np.sqrt(np.clip(S, 0.0, None)) * scale_factor

    # Parametric surface
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_ell = radii[0] * np.outer(np.cos(u), np.sin(v))
    y_ell = radii[1] * np.outer(np.sin(u), np.sin(v))
    z_ell = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # Rotate and translate
    pts = np.stack([x_ell.ravel(), y_ell.ravel(), z_ell.ravel()], axis=1)
    pts_rot = pts @ U.T  # (U * pts.T).T
    pts_rot += center

    # Reshape back for mesh
    n_u, n_v = len(u), len(v)
    X = pts_rot[:, 0].reshape(n_u, n_v)
    Y = pts_rot[:, 1].reshape(n_u, n_v)
    Z = pts_rot[:, 2].reshape(n_u, n_v)

    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, color], [1, color]], showscale=False, opacity=0.5))
    _apply_plotly_layout(fig, title)
    return _save_or_return_html(fig, output_path)


def create_3d_network_html(
    points: Dict[str, np.ndarray],
    connections: List[Tuple[str, str, Optional[float]]],
    residuals: Optional[Dict[Tuple[str, str], float]] = None,
    point_colors: Optional[Dict[str, str]] = None,
    title: str = "3D Survey Network",
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """
    Create an interactive 3D network plot (points + lines) with optional residual coloring.

    Parameters
    ----------
    points : dict
        Mapping from point ID to (x, y, z) array.
    connections : list of tuples (from_id, to_id, [distance])
        Connections between points. Distance may be used for coloring.
    residuals : dict, optional
        Mapping (from_id, to_id) -> residual value for color coding lines.
    point_colors : dict, optional
        Mapping point_id -> color (CSS color). Overrides default.
    title : str
        Plot title.
    output_path : str or Path, optional
        Save HTML to file.

    Returns
    -------
    str or None
        Path to HTML file if output_path given.
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not installed. Install with 'pip install plotly'")
        return None

    fig = go.Figure()

    # --- Points trace ---
    point_ids = list(points.keys())
    coords = np.array([points[pid] for pid in point_ids])
    point_colors_list = []
    if point_colors:
        point_colors_list = [point_colors.get(pid, "blue") for pid in point_ids]
    else:
        point_colors_list = "blue"

    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers+text",
        marker=dict(size=6, color=point_colors_list),
        text=point_ids,
        textposition="top center",
        name="Points",
        hoverinfo="text+x+y+z",
    ))

    # --- Lines (connections) ---
    for from_id, to_id, dist in connections:
        if from_id not in points or to_id not in points:
            continue
        p1 = points[from_id]
        p2 = points[to_id]
        # Line color based on residual if provided
        resid = None
        if residuals is not None:
            resid = residuals.get((from_id, to_id), None)
        line_color = "gray"
        if resid is not None:
            # Color map from green (small resid) to red (large)
            intensity = min(1.0, max(0.0, abs(resid) / 0.1))  # 0.1 m as max
            line_color = f"rgb({int(255*intensity)}, {int(255*(1-intensity))}, 0)"
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode="lines",
            line=dict(color=line_color, width=3),
            showlegend=False,
            hoverinfo="none",
        ))

    _apply_plotly_layout(fig, title)
    return _save_or_return_html(fig, output_path)


def _save_or_return_html(fig, output_path: Optional[Union[str, Path]]) -> Optional[str]:
    """Helper: save figure to HTML file or return its string representation."""
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        return str(out_path)
    else:
        # Return HTML string if needed (but usually we want file)
        return fig.to_html()


# ----------------------------------------------------------------------
# Academic-quality visualization methods
# ----------------------------------------------------------------------

def _ensure_matplotlib_seaborn():
    """Ensure matplotlib and seaborn are available, raise warning if not."""
    if plt is None:
        raise ImportError("Matplotlib is required for this plot. Install with 'pip install matplotlib'")
    if sns is None:
        # Fallback: use matplotlib only
        logger.warning("Seaborn not installed. Using matplotlib defaults.")


def plot_covariance_matrix_organized(
    covariance_matrix: np.ndarray,
    param_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "RdBu_r",
    show_values: bool = False,
    show_dendrogram: bool = True,
    title: str = "Covariance Matrix",
) -> Optional[str]:
    """
    Create an academically organized covariance matrix visualization.

    Features:
    - Heatmap with dendrogram showing parameter clustering
    - Automatic parameter naming if not provided
    - Symmetry emphasis with triangular mask option
    - Correlation or covariance display

    Parameters
    ----------
    covariance_matrix : ndarray of shape (p, p)
        Covariance or correlation matrix.
    param_names : list of str, optional
        Parameter names. Defaults to auto-generated labels (X1, X2, ... or X/Y/Z groups).
    output_path : str or Path, optional
        If provided, save figure to this path (PNG/PDF recommended for publication).
    figsize : tuple, optional
        Figure size (width, height) in inches.
    cmap : str, optional
        Colormap (diverging colormap like 'RdBu_r' is good for correlation).
    show_values : bool, optional
        If True, annotate cells with numeric values (use only for small matrices).
    show_dendrogram : bool, optional
        If True, add row/column clustering dendrograms.
    title : str, optional
        Plot title.

    Returns
    -------
    str or None
        Path to saved file if output_path given, else None.
    """
    _ensure_matplotlib_seaborn()
    cov = np.asarray(covariance_matrix, dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    p = cov.shape[0]
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square")

    # Generate parameter names if not provided
    if param_names is None:
        # Try to infer from shape (assume 3D point parameters: X,Y,Z repeated)
        if p % 3 == 0:
            n_points = p // 3
            param_names = []
            for i in range(n_points):
                param_names.extend([f"P{i+1}_X", f"P{i+1}_Y", f"P{i+1}_Z"])
        else:
            param_names = [f"θ_{i+1}" for i in range(p)]
    else:
        if len(param_names) != p:
            logger.warning(f"param_names length ({len(param_names)}) != matrix size ({p}). Using auto-generated.")
            param_names = [f"θ_{i+1}" for i in range(p)]

    # Convert to correlation matrix for better interpretability in heatmap
    diag = np.diag(cov).astype(float)
    looks_like_correlation = (
        np.nanmax(np.abs(cov)) <= 1.000001
        and np.allclose(diag, np.ones_like(diag), atol=1e-3)
    )
    if looks_like_correlation:
        corr_display = np.clip(cov, -1.0, 1.0)
        cbar_label = "Correlation coefficient"
    else:
        std_devs = np.sqrt(np.clip(np.abs(diag), 0.0, None))
        if np.all(std_devs > 0):
            corr = cov / np.outer(std_devs, std_devs)
            corr_display = np.clip(np.nan_to_num(corr), -1.0, 1.0)
            cbar_label = "Correlation coefficient"
        else:
            corr_display = cov
            cbar_label = "Covariance"

    if corr_display.shape != cov.shape:
        corr_display = cov
        cbar_label = "Correlation coefficient"

    if show_dendrogram and linkage is not None and squareform is not None:
        # Compute distance matrix (1 - |correlation|)
        dist_matrix = 1 - np.abs(np.nan_to_num(corr_display))
        dist_matrix = np.clip(dist_matrix, 0.0, 2.0)
        # Ensure symmetric and zero diagonal
        np.fill_diagonal(dist_matrix, 0)
        # Condensed distance vector
        condensed_dist = squareform(dist_matrix, checks=False)
        # Hierarchical clustering
        link = linkage(condensed_dist, method='average')

        # Set up figure with GridSpec for dendrograms
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, width_ratios=[0.2, 1], height_ratios=[0.2, 1], hspace=0.05, wspace=0.05)

        # Dendrogram on top (column clustering)
        ax_top = fig.add_subplot(gs[0, 1])
        dendrogram(link, ax=ax_top, orientation='top', labels=param_names, above_threshold_color='gray')
        ax_top.set_xticklabels([])
        ax_top.set_ylabel('Distance')
        ax_top.set_title('Column Clustering')

        # Dendrogram on left (row clustering)
        ax_left = fig.add_subplot(gs[1, 0])
        dendrogram(link, ax=ax_left, orientation='left', labels=param_names, above_threshold_color='gray')
        ax_left.set_yticklabels([])
        ax_left.set_xlabel('Distance')
        ax_left.set_title('Row Clustering')

        # Main heatmap
        ax_main = fig.add_subplot(gs[1, 1])
        # Reorder matrix according to clustering order
        idx = dendrogram(link, no_plot=True)['leaves']
        reordered_corr = corr_display[np.ix_(idx, idx)]
        reordered_labels = [param_names[i] for i in idx]

        im = ax_main.imshow(reordered_corr, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        ax_main.set_xticks(np.arange(p))
        ax_main.set_yticks(np.arange(p))
        ax_main.set_xticklabels(reordered_labels, rotation=90, fontsize=8)
        ax_main.set_yticklabels(reordered_labels, fontsize=8)
        ax_main.set_title(title)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_main, shrink=0.8)
        cbar.set_label(cbar_label)

        # Add value annotations if requested (only for small matrices)
        if show_values and p <= 20:
            for i in range(p):
                for j in range(p):
                    text = ax_main.text(j, i, f"{reordered_corr[i, j]:.2f}",
                                        ha="center", va="center", color="black" if abs(reordered_corr[i, j]) < 0.7 else "white", fontsize=6)
    else:
        # Simple heatmap without dendrogram
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr_display, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(np.arange(p))
        ax.set_yticks(np.arange(p))
        ax.set_xticklabels(param_names, rotation=90, fontsize=8)
        ax.set_yticklabels(param_names, fontsize=8)
        ax.set_title(title)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label)

        if show_values and p <= 15:
            for i in range(p):
                for j in range(p):
                    ax.text(j, i, f"{corr_display[i, j]:.2f}",
                            ha="center", va="center", color="black" if abs(corr_display[i, j]) < 0.7 else "white", fontsize=6)

    plt.tight_layout()
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
    else:
        plt.show()
        return None


def plot_normalized_residuals(
    residuals: np.ndarray,
    sigma0: float = 1.0,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Normalized Residuals Analysis",
) -> Optional[str]:
    """
    Create academic-quality plot for normalized residuals analysis.

    Features:
    - Histogram with theoretical normal curve overlay
    - Q-Q plot (quantile-quantile) for normality assessment
    - Standardized residual indices with control limits

    Parameters
    ----------
    residuals : ndarray
        Residual values (raw or standardized).
    sigma0 : float, optional
        A priori standard deviation (if residuals are raw, divide by sigma0).
    output_path : str or Path, optional
        Path to save the figure.
    figsize : tuple, optional
        Figure size.
    title : str, optional
        Main title.

    Returns
    -------
    str or None
        Path to saved file if output_path given.
    """
    _ensure_matplotlib_seaborn()
    res = np.asarray(residuals).flatten()
    # Standardize if sigma0 provided
    if sigma0 > 0:
        std_res = res / sigma0
    else:
        std_res = res

    n = len(std_res)
    mean_res = np.mean(std_res)
    std_res_actual = np.std(std_res, ddof=1)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Histogram + normal curve
    ax1.hist(std_res, bins='auto', density=True, alpha=0.7, color='steelblue', edgecolor='black', label='Standardized residuals')
    # Overlay theoretical normal PDF
    x = np.linspace(min(-3, std_res.min()), max(3, std_res.max()), 100)
    if stats is not None:
        pdf_theor = stats.norm.pdf(x, 0, 1)
        pdf_emp = stats.norm.pdf(x, mean_res, std_res_actual)
        ax1.plot(x, pdf_theor, 'r--', linewidth=2, label='Theoretical N(0,1)')
        ax1.plot(x, pdf_emp, 'g-', linewidth=1.5, label=f'Empirical N({mean_res:.2f},{std_res_actual:.2f})')
    else:
        # Simple normal approximation
        ax1.plot(x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), 'r--', linewidth=2, label='Theoretical N(0,1)')

    ax1.set_xlabel('Standardized Residual')
    ax1.set_ylabel('Density')
    ax1.set_title('Histogram with Normal Overlay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Q-Q plot
    if stats is not None:
        stats.probplot(std_res, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Check)')
        ax2.grid(True, alpha=0.3)
        # Add reference line annotations
        slope, intercept, r = stats.linregress(np.sort(stats.norm.ppf((np.arange(n) + 0.5) / n)), np.sort(std_res))[:3]
        ax2.annotate(f'R² = {r**2:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round", alpha=0.5))
    else:
        ax2.text(0.5, 0.5, "Q-Q plot requires scipy.stats", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Normality Check (scipy not available)')

    fig.suptitle(f"{title}\n(n={n}, mean={mean_res:.3f}, std={std_res_actual:.3f})", fontsize=12)
    plt.tight_layout()

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
    else:
        plt.show()
        return None


def plot_influence_diagnostics(
    leverage: np.ndarray,
    residuals_studentized: np.ndarray,
    cooks_d: Optional[np.ndarray] = None,
    param_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> Optional[str]:
    """
    Create influence diagnostics plots for adjustment/regression.

    Features:
    - Leverage vs. Studentized residuals (with Cook's distance contours)
    - Index plot of Cook's distance (identify influential observations)

    Parameters
    ----------
    leverage : ndarray
        Hat matrix diagonal (leverage values) for each observation.
    residuals_studentized : ndarray
        Internally or externally studentized residuals.
    cooks_d : ndarray, optional
        Cook's distance values (if not provided, estimated from leverage and residuals).
    param_names : list, optional
        Observation labels.
    output_path : str or Path, optional
        Path to save figure.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    str or None
        Path to saved file if output_path given.
    """
    _ensure_matplotlib_seaborn()
    lev = np.asarray(leverage).flatten()
    rstud = np.asarray(residuals_studentized).flatten()
    n = len(lev)
    p = None  # number of parameters, estimate from leverage? Not trivial.

    # Estimate number of parameters from average leverage (mean leverage = p/n)
    mean_lev = np.mean(lev)
    p_est = int(round(mean_lev * n))
    if p_est < 1:
        p_est = 2

    if cooks_d is None:
        # Approximate Cook's distance: D_i = (rstud_i^2 / p) * (lev_i / (1 - lev_i))
        cooks_d_est = (rstud**2 / p_est) * (lev / (1 - lev + 1e-10))
        cooks = cooks_d_est
    else:
        cooks = np.asarray(cooks_d).flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Leverage vs Studentized residuals with Cook's contours
    ax1.scatter(lev, rstud, alpha=0.7, c='steelblue', edgecolors='black')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='|r| = 2')
    ax1.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
    # Cook's distance contours (for typical p=2..10, show levels 0.5 and 1)
    lev_grid = np.linspace(0, max(lev)*1.1, 50)
    # For different p values, but we'll show approximate
    levels = [0.5, 1.0]
    colors = ['orange', 'darkred']
    for level, color in zip(levels, colors):
        # Solve for r given lev: D = (r^2/p)*(lev/(1-lev)) => r = sqrt(D * p * (1-lev)/lev)
        r_contour = np.sqrt(np.clip(level * p_est * (1 - lev_grid) / (lev_grid + 1e-10), 0, None))
        ax1.plot(lev_grid, r_contour, color=color, linestyle=':', linewidth=1.5, label=f"Cook's D = {level}")
        ax1.plot(lev_grid, -r_contour, color=color, linestyle=':')

    ax1.set_xlabel('Leverage (h_ii)')
    ax1.set_ylabel('Studentized Residual')
    ax1.set_title('Influence Plot (Leverage vs Residual)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Highlight points with high influence (Cook's D > 1 or high leverage)
    high_influence = (cooks > 1) | (lev > 2 * mean_lev) | (np.abs(rstud) > 2)
    if np.any(high_influence):
        ax1.scatter(lev[high_influence], rstud[high_influence], s=100, facecolors='none', edgecolors='red', linewidth=2, label='High influence')

    # Right: Index plot of Cook's distance
    indices = np.arange(1, n + 1)
    ax2.bar(indices, cooks, width=0.8, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axhline(y=1, color='red', linestyle='--', label='Cook\'s D = 1 (high influence)')
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='Cook\'s D = 0.5 (moderate)')
    ax2.set_xlabel('Observation Index')
    ax2.set_ylabel("Cook's Distance")
    ax2.set_title("Cook's Distance per Observation")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    if param_names is not None and len(param_names) == n:
        # Label high influence points
        for i, (idx, d_val) in enumerate(zip(indices, cooks)):
            if d_val > 0.5 or (param_names and high_influence[i]):
                ax2.annotate(param_names[i], (idx, d_val), xytext=(5, 3), textcoords='offset points', fontsize=7, alpha=0.7)

    fig.suptitle("Influence Diagnostics for Adjustment", fontsize=14)
    plt.tight_layout()

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
    else:
        plt.show()
        return None


def plot_residual_autocorrelation(
    residuals: np.ndarray,
    lags: int = 20,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 4),
) -> Optional[str]:
    """
    Plot residual autocorrelation function (ACF) for checking independence.

    Useful for time-series or ordered observations (e.g., sequential measurements).

    Parameters
    ----------
    residuals : ndarray
        Residual values in order of observation.
    lags : int, optional
        Maximum number of lags to display.
    output_path : str or Path, optional
        Path to save figure.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    str or None
        Path to saved file if output_path given.
    """
    _ensure_matplotlib_seaborn()
    res = np.asarray(residuals).flatten()
    n = len(res)

    if stats is None:
        logger.warning("scipy.stats not available for ACF calculation.")
        return None

    # Compute autocorrelation
    from scipy import stats as sp_stats
    acf = np.zeros(lags + 1)
    acf[0] = 1.0
    res_centered = res - np.mean(res)
    var = np.sum(res_centered**2)
    if var == 0:
        logger.warning("Residuals have zero variance; cannot compute ACF.")
        return None

    for k in range(1, lags + 1):
        acf[k] = np.sum(res_centered[k:] * res_centered[:-k]) / var

    # Approximate 95% confidence bounds (~ ±2/√n)
    conf_interval = 1.96 / np.sqrt(n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(lags + 1), acf, width=0.8, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=conf_interval, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='95% confidence bound')
    ax.axhline(y=-conf_interval, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.fill_between([0, lags], -conf_interval, conf_interval, color='red', alpha=0.1)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Residual Autocorrelation Function (ACF)')
    ax.set_xticks(np.arange(0, lags+1, max(1, lags//10)))
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    plt.tight_layout()
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
    else:
        plt.show()
        return None


def plot_residuals_vs_fitted(
    residuals: np.ndarray,
    fitted_values: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Optional[str]:
    """
    Residuals vs fitted values plot for heteroscedasticity detection.

    Parameters
    ----------
    residuals : ndarray
        Residual values.
    fitted_values : ndarray
        Predicted/transformed values.
    output_path : str or Path, optional
        Path to save figure.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    str or None
        Path to saved file if output_path given.
    """
    _ensure_matplotlib_seaborn()
    res = np.asarray(residuals).flatten()
    fitted = np.asarray(fitted_values).flatten()

    if len(res) != len(fitted):
        raise ValueError("residuals and fitted values must have same length")

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(fitted, res, alpha=0.6, c='steelblue', edgecolors='black')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    # Add lowess/loess smooth if statsmodels available? Not required, but add a moving average
    # Simple: bin average for trend
    if len(fitted) > 10:
        order = np.argsort(fitted)
        fitted_sorted = fitted[order]
        res_sorted = res[order]
        # Rolling mean
        window = max(5, len(fitted) // 20)
        rolling_mean = np.convolve(res_sorted, np.ones(window)/window, mode='valid')
        ax.plot(fitted_sorted[window//2: -window//2 + 1], rolling_mean, 'g-', linewidth=2, label='Moving average trend')
        ax.legend()

    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted Values')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
    else:
        plt.show()
        return None


def _save_matplotlib_figure(fig, output_path: Union[str, Path]) -> str:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _method_filtered_frame(df: pd.DataFrame, method_key: Optional[str]) -> pd.DataFrame:
    if df.empty or not method_key or "method" not in df.columns:
        return df.copy()
    filtered = df[df["method"].astype(str) == str(method_key)].copy()
    return filtered if not filtered.empty else df.copy()


def _numeric_residual_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "adjusted_x", "adjusted_y", "adjusted_z",
        "observed_x", "observed_y", "observed_z",
        "vx", "vy", "vz",
    ]
    out = df.copy()
    for col in required + ["v_norm"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    present = [col for col in required if col in out.columns]
    if present:
        out = out.dropna(subset=present)
    return out


def plot_observation_residuals(
    residual_df: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "Post-adjustment residuals",
    tolerance: Optional[float] = None,
) -> Optional[str]:
    r"""Plot residual components $v_x$, $v_y$, $v_z$ and $\|v\|$.

    When ``tolerance`` is provided in metres, the chart highlights the
    rejection bands $|v_i| > T$ used in dimensional-control inspection.
    """
    _ensure_matplotlib_seaborn()
    df = _numeric_residual_frame(residual_df)
    component_cols = [col for col in ("vx", "vy", "vz") if col in df.columns]
    if not component_cols:
        return None
    values = df[component_cols].to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=1)
    x = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col, color in zip(component_cols, ("#2563eb", "#16a34a", "#dc2626")):
        ax.plot(x, df[col].to_numpy(dtype=float), linewidth=0.9,
                alpha=0.85, label=col.upper(), color=color)
    ax.plot(x, norms, color="#111827", linewidth=1.3,
            label=r"$\|v\|$", alpha=0.9)
    ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--")
    _draw_tolerance_bands(ax, tolerance)
    ax.set_title(title)
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Residual (m)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(ncol=4, fontsize=8)
    return _save_matplotlib_figure(fig, output_path)


def _draw_tolerance_bands(ax, tolerance: Optional[float]):
    r"""Draw dimensional tolerance limits $\pm T$ on a residual axis."""
    if tolerance is None:
        return
    try:
        tol = abs(float(tolerance))
    except (TypeError, ValueError):
        return
    if not np.isfinite(tol) or tol <= 0:
        return
    ax.axhline(tol, color="#dc2626", linestyle="--",
               linewidth=1.05, label=rf"$+T={tol:.4g}\,\mathrm{{m}}$")
    ax.axhline(-tol, color="#dc2626", linestyle="--",
               linewidth=1.05, label=rf"$-T={tol:.4g}\,\mathrm{{m}}$")
    ymin, ymax = ax.get_ylim()
    ax.axhspan(tol, ymax, facecolor="#dc2626", alpha=0.10)
    ax.axhspan(ymin, -tol, facecolor="#dc2626", alpha=0.10)


def plot_observation_correction_vectors(
    residual_df: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "Observation correction vectors",
) -> Optional[str]:
    """Static correction-vector plot equivalent to the former GUI quiver view."""
    _ensure_matplotlib_seaborn()
    df = _numeric_residual_frame(residual_df)
    component_cols = [col for col in ("vx", "vy", "vz") if col in df.columns]
    if not component_cols:
        return None
    values = df[component_cols].to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=1)
    x = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.quiver(
        x,
        np.zeros_like(x, dtype=float),
        np.zeros_like(x, dtype=float),
        norms,
        norms,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        cmap="viridis",
        width=0.0035,
    )
    ax.axhline(0, color="#6b7280", linewidth=0.8)
    ymax = float(np.nanmax(norms)) if norms.size else 1.0
    ax.set_ylim(min(0.0, -0.05 * ymax), ymax * 1.12 if ymax > 0 else 1.0)
    ax.set_title(title)
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Correction norm (m)")
    ax.grid(True, linestyle="--", alpha=0.35)
    return _save_matplotlib_figure(fig, output_path)


def plot_network_geometry(
    residual_df: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "Network geometry (XY)",
) -> Optional[str]:
    """Static XY network geometry from analyzer residual outputs."""
    _ensure_matplotlib_seaborn()
    df = _numeric_residual_frame(residual_df)
    required = {"adjusted_x", "adjusted_y", "observed_x", "observed_y"}
    if not required.issubset(df.columns):
        return None

    fig, ax = plt.subplots(figsize=(8.5, 8))
    station_col = "station" if "station" in df.columns else None
    point_col = "point" if "point" in df.columns else None
    if station_col:
        grouped = df.groupby(station_col, dropna=False)
        for station, group in grouped:
            ax.scatter(
                group["adjusted_x"], group["adjusted_y"],
                s=18, alpha=0.75, label=str(station)
            )
    else:
        ax.scatter(df["adjusted_x"], df["adjusted_y"],
                   s=18, alpha=0.75, label="Adjusted")
    ax.scatter(
        df["observed_x"], df["observed_y"],
        s=22, marker="x", color="#111827", linewidths=0.8,
        alpha=0.65, label="Observed/reference"
    )
    if len(df) <= 80 and point_col:
        for _, row in df.iterrows():
            ax.annotate(str(row[point_col]), (row["adjusted_x"], row["adjusted_y"]),
                        fontsize=6, alpha=0.65, xytext=(2, 2),
                        textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel("Easting / X (m)")
    ax.set_ylabel("Northing / Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="best")
    return _save_matplotlib_figure(fig, output_path)


def _ellipse_xy_parts(center, covariance, visual_scale=1.0, confidence_sigma=np.sqrt(5.991464547), points=160):
    cov = np.asarray(covariance, dtype=float)
    if cov.shape[0] < 2 or cov.shape[1] < 2:
        return None
    cov_xy = 0.5 * (cov[:2, :2] + cov[:2, :2].T)
    if not np.all(np.isfinite(cov_xy)) or np.allclose(cov_xy, 0):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(cov_xy)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    axes = np.sqrt(eigenvalues) * confidence_sigma * visual_scale
    if not np.any(axes > 0):
        return None
    theta = np.linspace(0.0, 2.0 * np.pi, points)
    circle = np.vstack((np.cos(theta), np.sin(theta)))
    ellipse = eigenvectors @ np.diag(axes) @ circle
    center_xy = np.asarray(center, dtype=float)[:2]
    xy = center_xy[:, None] + ellipse
    major = eigenvectors[:, 0] * axes[0]
    minor = eigenvectors[:, 1] * axes[1]
    return xy[0], xy[1], center_xy, major, minor, axes / max(visual_scale, 1e-12)


def plot_xy_error_ellipses(
    residual_df: pd.DataFrame,
    covariance_by_station: Dict[str, np.ndarray],
    output_path: Union[str, Path],
    title: str = "XY error ellipses",
) -> Optional[str]:
    """Plot translucent XY error ellipses with light principal-axis strokes."""
    _ensure_matplotlib_seaborn()
    df = _numeric_residual_frame(residual_df)
    required = {"adjusted_x", "adjusted_y", "station"}
    if df.empty or not required.issubset(df.columns) or not covariance_by_station:
        return None

    ellipse_items = []
    for _, row in df.iterrows():
        station = str(row["station"])
        cov = covariance_by_station.get(station)
        if cov is None:
            continue
        center = np.asarray([row["adjusted_x"], row["adjusted_y"]], dtype=float)
        parts = _ellipse_xy_parts(center, cov, visual_scale=1.0)
        if parts is not None:
            ellipse_items.append((row, cov, parts))
    if not ellipse_items:
        return None

    coords = df[["adjusted_x", "adjusted_y"]].to_numpy(dtype=float)
    span = float(np.linalg.norm(np.nanmax(coords, axis=0) - np.nanmin(coords, axis=0)))
    span = span if np.isfinite(span) and span > 0 else 1.0
    axes_ref = []
    for _, _, parts in ellipse_items:
        axes_ref.append(float(parts[-1][0]))
    representative_axis = float(np.nanpercentile(axes_ref, 75)) if axes_ref else 0.0
    visual_scale = max(span * 0.025, 0.05) / representative_axis if representative_axis > 0 else 1.0
    visual_scale = float(np.clip(visual_scale, 1.0, 5000.0))

    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.scatter(df["adjusted_x"], df["adjusted_y"], s=12,
               color="#111827", alpha=0.45, label="Adjusted points")
    major_values = []
    drawn = 0
    for row, cov, _ in ellipse_items[:350]:
        center = np.asarray([row["adjusted_x"], row["adjusted_y"]], dtype=float)
        parts = _ellipse_xy_parts(center, cov, visual_scale=visual_scale)
        if parts is None:
            continue
        ex, ey, center_xy, major, minor, true_axes = parts
        major_values.append(float(true_axes[0] * 1000.0))
        color = "#2563eb"
        ax.fill(ex, ey, facecolor=color, edgecolor=color,
                linewidth=0.75, alpha=0.11)
        ax.plot(ex, ey, color=color, linewidth=0.75, alpha=0.58)
        ax.plot(
            [center_xy[0] - major[0], center_xy[0] + major[0]],
            [center_xy[1] - major[1], center_xy[1] + major[1]],
            color="#111827", linewidth=0.45, alpha=0.42,
        )
        ax.plot(
            [center_xy[0] - minor[0], center_xy[0] + minor[0]],
            [center_xy[1] - minor[1], center_xy[1] + minor[1]],
            color="#111827", linewidth=0.35, alpha=0.30,
        )
        drawn += 1

    if drawn == 0:
        plt.close(fig)
        return None

    if major_values:
        note = f"95% XY ellipses, scaled {visual_scale:.0f}x for visibility; median semi-major axis {np.nanmedian(major_values):.2f} mm."
        ax.text(0.01, 0.99, note, transform=ax.transAxes, va="top",
                ha="left", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="#d1d5db", alpha=0.82))
    ax.set_title(title)
    ax.set_xlabel("Easting / X (m)")
    ax.set_ylabel("Northing / Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(fontsize=8, loc="best")
    return _save_matplotlib_figure(fig, output_path)


def plot_control_point_residuals(
    residuals_gcp: np.ndarray,
    point_names: Optional[List[str]] = None,
    sigma0: float = 1.0,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Control Point Residuals",
    tolerance: Optional[float] = None,
) -> Optional[str]:
    r"""Histogram and ranked residual chart for control/check points.

    The optional engineering tolerance $T$ is interpreted in metres and is
    shown as acceptance limits $\pm T$ in the ranked residual panel.
    """
    _ensure_matplotlib_seaborn()
    res = np.asarray(residuals_gcp, dtype=float).ravel()
    res = res[np.isfinite(res)]
    if res.size < 3:
        return None
    sigma = float(sigma0) if np.isfinite(sigma0) and sigma0 > 0 else 1.0
    res_norm = res / sigma
    mean_res = float(np.mean(res_norm))
    std_res = float(np.std(res_norm, ddof=1)) if res_norm.size > 1 else 0.0
    rmse = float(np.sqrt(np.mean(res ** 2)))
    max_res = float(np.max(np.abs(res)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.hist(res_norm, bins="auto", density=True, alpha=0.72,
             color="#60a5fa", edgecolor="#111827", label="Residuals")
    x = np.linspace(min(-3.0, float(np.nanmin(res_norm))),
                    max(3.0, float(np.nanmax(res_norm))), 160)
    if stats is not None:
        ax1.plot(x, stats.norm.pdf(x, 0, 1), "r--",
                 linewidth=1.6, label="N(0,1)")
        ax1.plot(x, stats.norm.pdf(x, mean_res, max(std_res, 1e-9)),
                 color="#16a34a", linewidth=1.4,
                 label=f"Empirical N({mean_res:.2f},{std_res:.2f})")
    ax1.set_xlabel("Normalized residual")
    ax1.set_ylabel("Density")
    ax1.set_title("Distribution")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    order = np.argsort(np.abs(res))[::-1]
    ordered_res = res[order]
    x_idx = np.arange(1, ordered_res.size + 1)
    colors = np.where(np.abs(ordered_res) > 3.0 * sigma, "#dc2626", "#2563eb")
    ax2.bar(x_idx, ordered_res, width=0.82, color=colors,
            alpha=0.72, edgecolor="#111827")
    ax2.axhline(0, color="#6b7280", linestyle="--", linewidth=0.8)
    ax2.axhline(3.0 * sigma, color="#dc2626", linestyle="--",
                linewidth=1.0, label="+/- 3 sigma")
    ax2.axhline(-3.0 * sigma, color="#dc2626",
                linestyle="--", linewidth=1.0)
    _draw_tolerance_bands(ax2, tolerance)
    if point_names is not None and len(point_names) == res.size:
        ordered_names = [point_names[i] for i in order]
        for i, value in enumerate(ordered_res[:12]):
            ax2.annotate(str(ordered_names[i]), (x_idx[i], value),
                         xytext=(3, 3), textcoords="offset points",
                         fontsize=7, alpha=0.75)
    ax2.set_xlabel("Ranked point")
    ax2.set_ylabel("Residual (m)")
    ax2.set_title("Largest residuals")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle(f"{title} (n={res.size}, RMSE={rmse:.4f} m, max={max_res:.4f} m)")
    plt.tight_layout()
    if output_path:
        return _save_matplotlib_figure(fig, output_path)
    plt.show()
    return None


def plot_global_chi_square_test(
    sigma0_sq,
    dof,
    alpha: float = 0.05,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (9, 5.5),
    sigma0_apriori: Optional[float] = None,
) -> Optional[str]:
    r"""Plot the global variance-factor test based on $\chi^2_\nu$.

    The statistic is assessed through the dimensionless variance-factor
    interval $\chi^2_{\alpha/2,\nu}/\nu \leq
    \hat{\sigma}_0^2/\sigma_a^2 \leq
    \chi^2_{1-\alpha/2,\nu}/\nu$, where $\nu$ is the number of degrees of
    freedom and $\sigma_a$ is the a priori standard deviation in metres.
    """
    _ensure_matplotlib_seaborn()
    try:
        sigma0_sq = float(sigma0_sq)
        dof = int(dof)
        alpha = float(alpha)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(sigma0_sq) or sigma0_sq < 0 or dof <= 0 or not (0 < alpha < 1):
        return None
    variance_factor = sigma0_sq
    sigma0_apriori_note = "sem normalizacao a priori"
    try:
        sigma0_apriori_value = float(sigma0_apriori) if sigma0_apriori is not None else None
    except (TypeError, ValueError):
        sigma0_apriori_value = None
    if sigma0_apriori_value is not None and np.isfinite(sigma0_apriori_value) and sigma0_apriori_value > 0:
        variance_factor = sigma0_sq / (sigma0_apriori_value ** 2)
        sigma0_apriori_note = rf"$\sigma_a={sigma0_apriori_value * 1000.0:.2f}\,\mathrm{{mm}}$"

    lower = _chi_square_ppf(alpha / 2.0, dof) / dof
    upper = _chi_square_ppf(1.0 - alpha / 2.0, dof) / dof
    xmax = max(upper * 1.45, variance_factor * 1.25, 1.0)
    x = np.linspace(0.0, xmax, 700)
    pdf = dof * np.asarray([_chi_square_pdf(value * dof, dof) for value in x], dtype=float)
    accepted = lower <= variance_factor <= upper
    status = "Modelo Aceito" if accepted else "Modelo/Pesos Rejeitados"
    status_color = "#16a34a" if accepted else "#dc2626"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, pdf, color="#1f2937", linewidth=1.8,
            label=rf"$f(\hat{{\sigma}}_0^2/\sigma_a^2)$, $\nu={dof}$")
    ax.fill_between(x[x <= lower], pdf[x <= lower], color="#dc2626",
                    alpha=0.18, label=rf"Rejeição $\alpha/2$")
    ax.fill_between(x[x >= upper], pdf[x >= upper], color="#dc2626",
                    alpha=0.18, label=rf"Rejeição $\alpha/2$")
    ax.axvline(lower, color="#dc2626", linestyle="--", linewidth=1.0,
               label=rf"$\chi^2_{{\alpha/2,\nu}}/\nu={lower:.4g}$")
    ax.axvline(upper, color="#dc2626", linestyle="--", linewidth=1.0,
               label=rf"$\chi^2_{{1-\alpha/2,\nu}}/\nu={upper:.4g}$")
    ax.axvline(variance_factor, color=status_color, linewidth=2.4,
               label=rf"$\hat{{\sigma}}_0^2/\sigma_a^2={variance_factor:.4g}$")
    ax.text(
        0.98, 0.92,
        f"{status}\n"
        rf"$\hat{{\sigma}}_0={math.sqrt(sigma0_sq) * 1000.0:.2f}\,\mathrm{{mm}}$"
        f"\n{sigma0_apriori_note}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color=status_color,
        bbox=dict(facecolor="white", edgecolor=status_color, alpha=0.86),
    )
    ax.set_title(r"Teste Global do Fator de Variância $\hat{\sigma}_0^2$")
    ax.set_xlabel(r"Fator de variância $\hat{\sigma}_0^2 = \chi^2/\nu$")
    ax.set_ylabel(r"Densidade de probabilidade")
    ax.set_title(r"Teste Global do Fator de Variancia $\hat{\sigma}_0^2/\sigma_a^2$")
    ax.set_xlabel(r"Fator de variancia adimensional $\hat{\sigma}_0^2/\sigma_a^2 = \chi^2/\nu$")
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    if output_path:
        return _save_matplotlib_figure(fig, output_path)
    plt.show()
    return None


def _chi_square_pdf(x: float, dof: int) -> float:
    if x <= 0 or dof <= 0:
        return 0.0
    if stats is not None:
        return float(stats.chi2.pdf(x, dof))
    half = 0.5 * dof
    try:
        return float((x ** (half - 1.0) * math.exp(-0.5 * x)) / ((2.0 ** half) * math.gamma(half)))
    except (OverflowError, ValueError):
        return 0.0


def _chi_square_ppf(probability: float, dof: int) -> float:
    if stats is not None:
        return float(stats.chi2.ppf(probability, dof))
    probability = float(np.clip(probability, 1e-9, 1.0 - 1e-9))
    z = NormalDist().inv_cdf(probability)
    value = dof * (1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))) ** 3
    return float(max(value, 0.0))


def plot_spatial_quiver_2d(
    residual_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    scale_quiver: float = 1.0,
    figsize: Tuple[int, int] = (9, 8),
    title: str = "Spatial residual quiver map",
) -> Optional[str]:
    """Plot real-position residual vectors $(v_x,v_y)$ over $(X,Y)$."""
    _ensure_matplotlib_seaborn()
    df = _numeric_residual_frame(residual_df)
    required = {"adjusted_x", "adjusted_y", "vx", "vy"}
    if df.empty or not required.issubset(df.columns):
        return None
    df = df.dropna(subset=list(required)).copy()
    if df.empty:
        return None
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required))
    if df.empty:
        return None
    try:
        scale_quiver = float(scale_quiver)
    except (TypeError, ValueError):
        scale_quiver = 1.0
    scale_quiver = scale_quiver if np.isfinite(scale_quiver) and scale_quiver > 0 else 1.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df["adjusted_x"], df["adjusted_y"], s=12,
               color="#111827", alpha=0.42, label="Adjusted points")
    q = ax.quiver(
        df["adjusted_x"], df["adjusted_y"],
        df["vx"] * scale_quiver, df["vy"] * scale_quiver,
        angles="xy", scale_units="xy", scale=1.0,
        color="#2563eb", width=0.0032, alpha=0.82,
        label=rf"$({{v_x}},{{v_y}})\times {scale_quiver:g}$",
    )
    ax.quiverkey(q, X=0.08, Y=0.94, U=0.001 * scale_quiver,
                 label=rf"Seta $=1\,\mathrm{{mm}}$", labelpos="E",
                 coordinates="axes")
    ax.set_title(title)
    ax.set_xlabel(r"Easting / $X$ (m)")
    ax.set_ylabel(r"Northing / $Y$ (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    if output_path:
        return _save_matplotlib_figure(fig, output_path)
    plt.show()
    return None


def plot_local_redundancy_numbers(
    leverage,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> Optional[str]:
    """Plot Baarda local redundancy numbers $r_i = 1-h_{ii}$."""
    _ensure_matplotlib_seaborn()
    lev = np.asarray(leverage, dtype=float).ravel()
    lev = lev[np.isfinite(lev)]
    if lev.size == 0:
        return None
    redundancy = 1.0 - lev
    x = np.arange(1, redundancy.size + 1)
    colors = np.where(redundancy < 0.1, "#dc2626", "#2563eb")
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, redundancy, color=colors, alpha=0.78,
           edgecolor="#111827", linewidth=0.35)
    ax.axhline(0.1, color="#dc2626", linestyle="--", linewidth=1.2,
               label=rf"Limite crítico $r_i=0.1$")
    ax.set_title(r"Confiabilidade Local de Baarda: $r_i = 1-h_{ii}$")
    ax.set_xlabel("Observation index")
    ax.set_ylabel(r"Número de redundância local $r_i$")
    ax.set_ylim(min(0.0, float(np.nanmin(redundancy)) * 1.05),
                max(1.0, float(np.nanmax(redundancy)) * 1.08))
    ax.grid(True, axis="y", linestyle="--", alpha=0.32)
    ax.legend(fontsize=8)
    plt.tight_layout()
    if output_path:
        return _save_matplotlib_figure(fig, output_path)
    plt.show()
    return None


def plot_residual_spatial_trends(
    residual_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (13, 4.5),
) -> Optional[str]:
    """Plot trends $v_x(X)$, $v_y(Y)$ and $v_z(Z)$ for systematic effects."""
    _ensure_matplotlib_seaborn()
    df = _numeric_residual_frame(residual_df)
    pairs = [("adjusted_x", "vx", r"$v_x$ vs $X$"),
             ("adjusted_y", "vy", r"$v_y$ vs $Y$"),
             ("adjusted_z", "vz", r"$v_z$ vs $Z$")]
    if df.empty or not all({x, y}.issubset(df.columns) for x, y, _ in pairs):
        return None
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    any_drawn = False
    for ax, (x_col, y_col, subtitle) in zip(axes, pairs):
        sub = df[[x_col, y_col]].copy()
        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
        sub = sub.dropna().sort_values(x_col)
        if sub.empty:
            ax.set_visible(False)
            continue
        any_drawn = True
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        ax.scatter(x, y, s=18, color="#2563eb", alpha=0.68,
                   edgecolor="#111827", linewidth=0.25, label="Residuals")
        if x.size >= 2 and np.nanstd(x) > 0:
            coef = np.polyfit(x, y, 1)
            ax.plot(x, np.polyval(coef, x), color="#dc2626", linewidth=1.5,
                    label=rf"Tendência linear")
        if x.size >= 5:
            window = max(3, min(15, int(np.ceil(x.size * 0.12))))
            smooth = pd.Series(y).rolling(window, center=True, min_periods=2).mean().to_numpy()
            ax.plot(x, smooth, color="#16a34a", linewidth=1.3,
                    label=rf"Média móvel")
        ax.axhline(0, color="#6b7280", linestyle="--", linewidth=0.8)
        ax.set_title(subtitle)
        ax.set_xlabel(x_col.replace("adjusted_", "").upper() + " (m)")
        ax.set_ylabel("Residual (m)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=7)
    if not any_drawn:
        plt.close(fig)
        return None
    fig.suptitle(r"Tendências Espaciais dos Resíduos $(v_x,v_y,v_z)$")
    plt.tight_layout()
    if output_path:
        return _save_matplotlib_figure(fig, output_path)
    plt.show()
    return None


def plot_pareto_outliers(
    outlier_flags_df: pd.DataFrame,
    group_by: str = "point",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Outlier Candidate Pareto",
) -> Optional[str]:
    """Pareto chart of outlier flags grouped by point or station."""
    _ensure_matplotlib_seaborn()
    if outlier_flags_df is None or outlier_flags_df.empty:
        return None
    df = outlier_flags_df.copy()
    if "is_outlier_candidate" in df.columns:
        candidate = df["is_outlier_candidate"].astype(str).str.lower().isin(
            {"true", "1", "yes"})
        df = df[candidate].copy()
    if df.empty:
        return None
    group_col = group_by if group_by in df.columns else (
        "point" if "point" in df.columns else "station" if "station" in df.columns else None)
    if group_col:
        grouped = df.groupby(group_col).size().sort_values(ascending=False)
    else:
        grouped = pd.Series({"Outlier candidates": len(df)})
    grouped = grouped.head(30)
    if grouped.empty:
        return None

    fig, ax1 = plt.subplots(figsize=figsize)
    labels = grouped.index.astype(str).tolist()
    counts = grouped.to_numpy(dtype=float)
    ax1.bar(labels, counts, color="#2563eb", alpha=0.74,
            edgecolor="#111827")
    ax1.set_ylabel("Candidate count", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_title(title)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    cumulative = np.cumsum(counts) / np.sum(counts) * 100.0
    ax2.plot(labels, cumulative, "o-", color="#dc2626",
             linewidth=1.7, markersize=4)
    ax2.axhline(80, color="#f59e0b", linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Cumulative percent (%)", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    fig.tight_layout()
    if output_path:
        return _save_matplotlib_figure(fig, output_path)
    plt.show()
    return None


def plot_qq_with_envelope(
    residuals: np.ndarray,
    sigma0: float = 1.0,
    n_simulations: int = 500,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (6, 6),
    title: str = "Q-Q Plot with Simulation Envelope",
) -> Optional[str]:
    """Q-Q normality plot with a simulated 95% envelope."""
    _ensure_matplotlib_seaborn()
    if stats is None:
        return None
    res = np.asarray(residuals, dtype=float).ravel()
    res = res[np.isfinite(res)]
    if res.size < 3:
        return None
    sigma = float(sigma0) if np.isfinite(sigma0) and sigma0 > 0 else 1.0
    res = np.sort(res / sigma)
    n = res.size
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    rng = np.random.default_rng(20260610)
    simulations = np.sort(rng.normal(size=(int(n_simulations), n)), axis=1)
    lower = np.percentile(simulations, 2.5, axis=0)
    upper = np.percentile(simulations, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(theoretical, lower, upper,
                    color="#9ca3af", alpha=0.32, label="95% envelope")
    ax.plot(theoretical, res, "o", markersize=4,
            color="#2563eb", alpha=0.75, label="Residuals")
    ax.plot(theoretical, theoretical, "r--", linewidth=1.3, label="N(0,1)")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample normalized residuals")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    if output_path:
        return _save_matplotlib_figure(fig, output_path)
    plt.show()
    return None


def plot_residual_map_2d(
    points_xy: np.ndarray,
    residuals: np.ndarray,
    point_labels: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "2D Residual Map",
) -> Optional[str]:
    """Plan-view residual map colored by residual magnitude."""
    _ensure_matplotlib_seaborn()
    points = np.asarray(points_xy, dtype=float)
    res = np.asarray(residuals, dtype=float).ravel()
    valid = (
        points.ndim == 2 and points.shape[1] >= 2 and points.shape[0] == res.size
    )
    if not valid:
        return None
    mask = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) & np.isfinite(res)
    points = points[mask]
    res = res[mask]
    if point_labels is not None and len(point_labels) == mask.size:
        point_labels = [label for label, keep in zip(point_labels, mask) if keep]
    if res.size < 1:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(points[:, 0], points[:, 1], c=res, s=52,
                    cmap="RdYlBu_r", alpha=0.84,
                    edgecolors="#111827", linewidth=0.45)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Residual norm (m)")
    if point_labels is not None and len(point_labels) == res.size and res.size <= 120:
        for i, label in enumerate(point_labels):
            ax.annotate(str(label), (points[i, 0], points[i, 1]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Easting / X (m)")
    ax.set_ylabel("Northing / Y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    if output_path:
        return _save_matplotlib_figure(fig, output_path)
    plt.show()
    return None


def plot_station_transformation_parameters(
    station_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Station Transformation Parameters",
) -> Optional[str]:
    """Grouped bar chart for station translations and scale."""
    _ensure_matplotlib_seaborn()
    if station_df is None or station_df.empty:
        return None
    df = station_df.copy()
    required = [col for col in ("tx", "ty", "tz", "scale") if col in df.columns]
    if not required:
        return None
    if "station" not in df.columns:
        return None
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required, how="all")
    if df.empty:
        return None
    df = df.head(36)
    names = df["station"].astype(str).tolist()
    x = np.arange(len(df))
    components = [col for col in ("tx", "ty", "tz") if col in df.columns]
    fig, ax = plt.subplots(figsize=figsize)
    if components:
        width = 0.76 / len(components)
        for i, comp in enumerate(components):
            offset = (i - (len(components) - 1) / 2.0) * width
            ax.bar(x + offset, df[comp].to_numpy(dtype=float), width,
                   label=comp.upper(), alpha=0.74, edgecolor="#111827")
        ax.set_ylabel("Translation (m)")
    elif "scale" in df.columns:
        ax.bar(x, df["scale"].to_numpy(dtype=float),
               color="#2563eb", alpha=0.74, edgecolor="#111827")
        ax.set_ylabel("Scale")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    if output_path:
        return _save_matplotlib_figure(fig, output_path)
    plt.show()
    return None


# ----------------------------------------------------------------------
# SurveyVisualizer class (extended with academic plots)
# ----------------------------------------------------------------------
class SurveyVisualizer:
    """
    Creates visualizations for survey adjustment results.
    Supports both static Matplotlib and interactive Plotly HTML outputs,
    including academic-quality plots for publication.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Static Matplotlib methods (existing, with minor improvements)
    # ------------------------------------------------------------------
    def create_residual_area_plot(
        self,
        adjusted_coords: Dict,
        final_residuals: List[float],
        survey_data,
        output_path: Union[str, Path],
        grid_resolution: int = 20,
    ) -> str:
        """Create static residual distribution plots (matplotlib)."""
        if plt is None:
            self.logger.warning("Matplotlib not installed")
            return ""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect midpoints and residuals
        points = []
        residuals_list = []
        for i, measurement in enumerate(survey_data.measurements):
            from_point = measurement["from_point"]
            to_point = measurement["to_point"]
            if from_point in adjusted_coords and to_point in adjusted_coords and i < len(final_residuals):
                from_coord = adjusted_coords[from_point]
                to_coord = adjusted_coords[to_point]
                mid = ((from_coord[0] + to_coord[0]) / 2, (from_coord[1] + to_coord[1]) / 2)
                points.append(mid)
                residuals_list.append(final_residuals[i])

        if not points:
            self.logger.warning("No valid points for residual area plot")
            return ""

        points = np.asarray(points, dtype=float)
        residuals_arr = np.asarray(residuals_list, dtype=float)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Scatter plot
        scatter = ax1.scatter(points[:, 0], points[:, 1], c=residuals_arr, cmap="RdYlBu_r", s=50, alpha=0.7)
        ax1.set_title("Final Residuals Distribution")
        ax1.set_xlabel("X Coordinate (m)")
        ax1.set_ylabel("Y Coordinate (m)")
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label="Residual (m)")

        # Histogram
        ax2.hist(residuals_arr, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax2.set_title("Residual Distribution Histogram")
        ax2.set_xlabel("Residual (m)")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)

        # Residual vs distance
        distances = []
        for measurement in survey_data.measurements:
            from_point = measurement["from_point"]
            to_point = measurement["to_point"]
            if from_point in adjusted_coords and to_point in adjusted_coords:
                from_coord = adjusted_coords[from_point]
                to_coord = adjusted_coords[to_point]
                dist = np.hypot(to_coord[0] - from_coord[0], to_coord[1] - from_coord[1])
                distances.append(dist)

        if distances:
            # Align lengths: use the same subset as residuals_list
            # Actually residuals_list corresponds to the same order as survey_data.measurements (filtered)
            # But we need to ensure we use only those measurements that were actually used.
            # For simplicity, we assume distances list order matches residuals_list order.
            # This is a logical fix: filter distances similarly.
            valid_distances = []
            for i, meas in enumerate(survey_data.measurements):
                from_p = meas["from_point"]
                to_p = meas["to_point"]
                if from_p in adjusted_coords and to_p in adjusted_coords and i < len(final_residuals):
                    from_c = adjusted_coords[from_p]
                    to_c = adjusted_coords[to_p]
                    valid_distances.append(np.hypot(to_c[0] - from_c[0], to_c[1] - from_c[1]))
            if len(valid_distances) == len(residuals_list):
                ax3.scatter(valid_distances, residuals_list, alpha=0.6, color="green")
                ax3.set_title("Residual vs Distance")
                ax3.set_xlabel("Distance (m)")
                ax3.set_ylabel("Residual (m)")
                ax3.grid(True, alpha=0.3)

        # Network layout
        for point_id, coord in adjusted_coords.items():
            marker = "s" if str(point_id).startswith("STA") else "o"
            color = "red" if str(point_id).startswith("STA") else "blue"
            ax4.scatter(coord[0], coord[1], c=color, s=80 if marker == "s" else 45, marker=marker, alpha=0.7)
        ax4.set_title("Survey Network Layout")
        ax4.set_xlabel("X Coordinate (m)")
        ax4.set_ylabel("Y Coordinate (m)")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = output_path.with_suffix(".png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(plot_file)

    def create_station_transformation_plot(self, station_transformations: Dict, output_path: Union[str, Path]) -> str:
        """Plot station translation parameters as bar chart."""
        if plt is None:
            return ""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        station_ids, tx, ty, tz = [], [], [], []
        for station_id, transformation in station_transformations.items():
            station_ids.append(station_id)
            trans = transformation.get("effective_translation", {})
            tx.append(trans.get("delta_x", 0.0))
            ty.append(trans.get("delta_y", 0.0))
            tz.append(trans.get("delta_z", 0.0))

        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(station_ids))
        width = 0.25
        ax.bar(x_pos - width, tx, width, label="dX")
        ax.bar(x_pos, ty, width, label="dY")
        ax.bar(x_pos + width, tz, width, label="dZ")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(station_ids)
        ax.set_ylabel("Translation (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plot_file = output_path.with_suffix(".png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(plot_file)

    def create_precision_analysis_plot(self, adjusted_coords, variance_covariance, output_path):
        """Heatmap of covariance matrix (static)."""
        if plt is None:
            return ""
        variance_covariance = np.asarray(variance_covariance, dtype=float)
        if variance_covariance.size == 0 or variance_covariance.shape[0] < 3:
            return ""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        matrix = variance_covariance[: min(12, variance_covariance.shape[0]), : min(12, variance_covariance.shape[1])]
        if sns is not None:
            sns.heatmap(matrix, annot=False, cmap="RdYlBu_r", ax=ax)
        else:
            ax.imshow(matrix, cmap="RdYlBu_r")
        ax.set_title("Variance-Covariance Matrix")
        plt.tight_layout()
        plot_file = output_path.with_suffix(".png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(plot_file)

    def create_matrices_visualization(self, A_matrix, b_vector, residuals, weights, normal_matrix, variance_covariance, output_path):
        """Static matrix gallery."""
        if plt is None:
            return ""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        matrices = [
            (A_matrix, "Design Matrix A"),
            (np.asarray(b_vector).reshape(-1, 1), "Observation Vector b"),
            (np.asarray(residuals).reshape(-1, 1), "Residuals v"),
            (weights, "Weights P"),
            (normal_matrix, "Normal Matrix N"),
            (variance_covariance, "Covariance Q"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for ax, (matrix, title) in zip(axes.ravel(), matrices):
            arr = np.asarray(matrix, dtype=float)
            ax.imshow(arr, aspect="auto", cmap="viridis")
            ax.set_title(title)
        plt.tight_layout()
        plot_file = output_path.with_suffix(".png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(plot_file)

    def create_matrix_details_view(self, matrix, title, output_path):
        """Alias for create_precision_analysis_plot with a single matrix."""
        return self.create_precision_analysis_plot({}, matrix, output_path)

    def create_equations_flowchart(self, adjustment_results, output_file="equations_flowchart.png"):
        """Simple flowchart placeholder."""
        if plt is None:
            return ""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")
        ax.text(0.5, 0.5, "Input data -> adjustment method -> standardized outputs -> analysis", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(output_path)

    # ------------------------------------------------------------------
    # New HTML 3D methods (Plotly)
    # ------------------------------------------------------------------
    def create_3d_residual_plot_html(
        self,
        adjusted_coords: Dict[str, np.ndarray],
        residuals_xyz: Dict[str, np.ndarray],
        output_path: Union[str, Path],
        title: str = "3D Residuals at Points",
    ) -> Optional[str]:
        """
        Interactive 3D plot showing points with residual vectors.

        Parameters
        ----------
        adjusted_coords : dict
            Point ID -> (x,y,z) array.
        residuals_xyz : dict
            Point ID -> residual vector (dx,dy,dz) or scalar norm.
        output_path : str or Path
            Where to save the HTML file.
        title : str
            Plot title.

        Returns
        -------
        str or None
            Path to HTML file if successful.
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not installed")
            return None

        fig = go.Figure()
        # Points
        point_ids = list(adjusted_coords.keys())
        coords = np.array([adjusted_coords[pid] for pid in point_ids])
        # Residual norms for coloring
        norms = []
        for pid in point_ids:
            res = residuals_xyz.get(pid, np.zeros(3))
            norm = np.linalg.norm(res)
            norms.append(norm)

        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers+text",
            marker=dict(size=7, color=norms, colorscale="Viridis", colorbar=_plotly_left_colorbar("Residual norm (m)")),
            text=point_ids,
            textposition="top center",
            name="Points",
        ))

        # Residual vectors as lines from point to point+residual
        for pid, res in residuals_xyz.items():
            if pid not in adjusted_coords:
                continue
            p = adjusted_coords[pid]
            end = p + res
            fig.add_trace(go.Scatter3d(
                x=[p[0], end[0]],
                y=[p[1], end[1]],
                z=[p[2], end[2]],
                mode="lines",
                line=dict(color="red", width=2),
                showlegend=False,
                hoverinfo="none",
            ))

        _apply_plotly_layout(fig, title)
        if fig.data:
            _set_marker_colorbar(fig.data[0], "Residual norm (m)")
        out_path = Path(output_path).with_suffix(".html")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        return str(out_path)

    def create_3d_covariance_ellipsoids_html(
        self,
        centers: Dict[str, np.ndarray],
        covariances: Dict[str, np.ndarray],
        output_path: Union[str, Path],
        scale_factor: float = 1.0,
        title: str = "Covariance Ellipsoids",
    ) -> Optional[str]:
        """
        Create HTML with multiple 3D ellipsoids (one per point).

        Parameters
        ----------
        centers : dict
            Point ID -> (x,y,z) center.
        covariances : dict
            Point ID -> 3x3 covariance matrix.
        output_path : str or Path
            Output HTML file.
        scale_factor : float
            Scaling for ellipsoid radii.
        title : str
            Plot title.

        Returns
        -------
        str or None
            Path to HTML file.
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not installed")
            return None

        fig = go.Figure()
        colors = px.colors.qualitative.Plotly if px is not None else ["red", "green", "blue", "orange", "purple"]
        for i, (pid, center) in enumerate(centers.items()):
            cov = covariances.get(pid)
            if cov is None:
                continue
            # Generate ellipsoid mesh (same as create_3d_ellipsoid_html)
            U, S, _ = svd(np.asarray(cov))
            radii = np.sqrt(np.clip(S, 0.0, None)) * scale_factor
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x_ell = radii[0] * np.outer(np.cos(u), np.sin(v))
            y_ell = radii[1] * np.outer(np.sin(u), np.sin(v))
            z_ell = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            pts = np.stack([x_ell.ravel(), y_ell.ravel(), z_ell.ravel()], axis=1)
            pts_rot = pts @ U.T
            pts_rot += np.asarray(center)
            n_u, n_v = len(u), len(v)
            X = pts_rot[:, 0].reshape(n_u, n_v)
            Y = pts_rot[:, 1].reshape(n_u, n_v)
            Z = pts_rot[:, 2].reshape(n_u, n_v)
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                colorscale=[[0, colors[i % len(colors)]], [1, colors[i % len(colors)]]],
                showscale=False,
                opacity=0.4,
                name=f"{pid} ellipsoid",
            ))

        _apply_plotly_layout(fig, title)
        out_path = Path(output_path).with_suffix(".html")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        return str(out_path)

    # ------------------------------------------------------------------
    # New academic-quality plots (exposed as methods)
    # ------------------------------------------------------------------
    def plot_covariance_organized(
        self,
        covariance_matrix: np.ndarray,
        param_names: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = "RdBu_r",
        show_values: bool = False,
        show_dendrogram: bool = True,
        title: str = "Covariance Matrix",
    ) -> Optional[str]:
        """Public wrapper for plot_covariance_matrix_organized."""
        return plot_covariance_matrix_organized(
            covariance_matrix, param_names, output_path, figsize, cmap, show_values, show_dendrogram, title
        )

    def plot_normalized_residuals(
        self,
        residuals: np.ndarray,
        sigma0: float = 1.0,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Normalized Residuals Analysis",
    ) -> Optional[str]:
        """Public wrapper for plot_normalized_residuals."""
        return plot_normalized_residuals(residuals, sigma0, output_path, figsize, title)

    def plot_influence_diagnostics(
        self,
        leverage: np.ndarray,
        residuals_studentized: np.ndarray,
        cooks_d: Optional[np.ndarray] = None,
        param_names: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 5),
    ) -> Optional[str]:
        """Public wrapper for plot_influence_diagnostics."""
        return plot_influence_diagnostics(leverage, residuals_studentized, cooks_d, param_names, output_path, figsize)

    def plot_residual_autocorrelation(
        self,
        residuals: np.ndarray,
        lags: int = 20,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 4),
    ) -> Optional[str]:
        """Public wrapper for plot_residual_autocorrelation."""
        return plot_residual_autocorrelation(residuals, lags, output_path, figsize)

    def plot_residuals_vs_fitted(
        self,
        residuals: np.ndarray,
        fitted_values: np.ndarray,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> Optional[str]:
        """Public wrapper for plot_residuals_vs_fitted."""
        return plot_residuals_vs_fitted(residuals, fitted_values, output_path, figsize)

    def plot_observation_residuals(
        self,
        residual_df: pd.DataFrame,
        output_path: Union[str, Path],
        title: str = "Post-adjustment residuals",
        tolerance: Optional[float] = None,
    ) -> Optional[str]:
        return plot_observation_residuals(residual_df, output_path, title, tolerance)

    def plot_observation_correction_vectors(
        self,
        residual_df: pd.DataFrame,
        output_path: Union[str, Path],
        title: str = "Observation correction vectors",
    ) -> Optional[str]:
        return plot_observation_correction_vectors(residual_df, output_path, title)

    def plot_network_geometry(
        self,
        residual_df: pd.DataFrame,
        output_path: Union[str, Path],
        title: str = "Network geometry (XY)",
    ) -> Optional[str]:
        return plot_network_geometry(residual_df, output_path, title)

    def plot_xy_error_ellipses(
        self,
        residual_df: pd.DataFrame,
        covariance_by_station: Dict[str, np.ndarray],
        output_path: Union[str, Path],
        title: str = "XY error ellipses",
    ) -> Optional[str]:
        return plot_xy_error_ellipses(residual_df, covariance_by_station, output_path, title)

    def plot_control_point_residuals(
        self,
        residuals_gcp: np.ndarray,
        point_names: Optional[List[str]] = None,
        sigma0: float = 1.0,
        output_path: Optional[Union[str, Path]] = None,
        tolerance: Optional[float] = None,
        **kwargs,
    ) -> Optional[str]:
        return plot_control_point_residuals(residuals_gcp, point_names, sigma0, output_path, tolerance=tolerance, **kwargs)

    def plot_global_chi_square_test(
        self,
        sigma0_sq,
        dof,
        alpha: float = 0.05,
        output_path: Optional[Union[str, Path]] = None,
        sigma0_apriori: Optional[float] = None,
        **kwargs,
    ) -> Optional[str]:
        return plot_global_chi_square_test(
            sigma0_sq,
            dof,
            alpha,
            output_path,
            sigma0_apriori=sigma0_apriori,
            **kwargs,
        )

    def plot_spatial_quiver_2d(
        self,
        residual_df: pd.DataFrame,
        output_path: Optional[Union[str, Path]] = None,
        scale_quiver: float = 1.0,
        **kwargs,
    ) -> Optional[str]:
        return plot_spatial_quiver_2d(residual_df, output_path, scale_quiver, **kwargs)

    def plot_local_redundancy_numbers(
        self,
        leverage,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Optional[str]:
        return plot_local_redundancy_numbers(leverage, output_path, **kwargs)

    def plot_residual_spatial_trends(
        self,
        residual_df: pd.DataFrame,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Optional[str]:
        return plot_residual_spatial_trends(residual_df, output_path, **kwargs)

    def plot_pareto_outliers(
        self,
        outlier_flags_df: pd.DataFrame,
        group_by: str = "point",
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Optional[str]:
        return plot_pareto_outliers(outlier_flags_df, group_by, output_path, **kwargs)

    def plot_qq_with_envelope(
        self,
        residuals: np.ndarray,
        sigma0: float = 1.0,
        n_simulations: int = 500,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Optional[str]:
        return plot_qq_with_envelope(residuals, sigma0, n_simulations, output_path, **kwargs)

    def plot_residual_map_2d(
        self,
        points_xy: np.ndarray,
        residuals: np.ndarray,
        point_labels: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Optional[str]:
        return plot_residual_map_2d(points_xy, residuals, point_labels, output_path, **kwargs)

    def plot_station_transformation_parameters(
        self,
        station_df: pd.DataFrame,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Optional[str]:
        return plot_station_transformation_parameters(station_df, output_path, **kwargs)


def _diagnostic_residual_arrays(output_dir: Union[str, Path], method_key: Optional[str]):
    path = Path(output_dir) / "residuals.csv"
    if not path.exists():
        return None, None, pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, None, pd.DataFrame()
    if df.empty:
        return None, None, df
    if method_key and "method" in df.columns:
        filtered = df[df["method"].astype(str) == str(method_key)].copy()
        if not filtered.empty:
            df = filtered

    component_cols = [col for col in ("vx", "vy", "vz") if col in df.columns]
    if component_cols:
        residuals = df[component_cols].apply(
            pd.to_numeric, errors="coerce").to_numpy(dtype=float).ravel()
    elif "v_norm" in df.columns:
        residuals = pd.to_numeric(
            df["v_norm"], errors="coerce").to_numpy(dtype=float)
    else:
        residuals = np.asarray([], dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    fitted_cols = [
        col for col in ("adjusted_x", "adjusted_y", "adjusted_z")
        if col in df.columns
    ]
    if fitted_cols and component_cols:
        fitted = df[fitted_cols].apply(
            pd.to_numeric, errors="coerce").to_numpy(dtype=float).ravel()
    elif "adjusted_x" in df.columns:
        fitted = pd.to_numeric(
            df["adjusted_x"], errors="coerce").to_numpy(dtype=float)
    else:
        fitted = np.asarray([], dtype=float)
    fitted = fitted[np.isfinite(fitted)]
    if residuals.size and fitted.size:
        count = min(residuals.size, fitted.size)
        residuals = residuals[:count]
        fitted = fitted[:count]
    return residuals, fitted, df


def _diagnostic_covariance_array(output_dir: Union[str, Path], method_key: Optional[str]):
    arrays_path = Path(output_dir) / "outputs_full_arrays.npz"
    if not arrays_path.exists():
        return None, "", []
    try:
        arrays = np.load(arrays_path)
    except Exception:
        return None, "", []

    preferred_terms = (
        "network_coupled_covariance",
        "parameter_covariance",
        "numeric_covariance_approx",
        "covariance",
        "Q_xi",
        "correlation",
        "normal_matrix",
    )
    candidates = []
    for key in arrays.files:
        if method_key:
            key_method = str(key).split("__", 1)[0]
            if key_method != str(method_key):
                continue
        lower = key.lower()
        if any(term.lower() in lower for term in preferred_terms):
            if "cross_covariance" in lower:
                continue
            if "observation_covariance" in lower or "residual_covariance" in lower:
                continue
            try:
                arr = np.asarray(arrays[key], dtype=float)
            except Exception:
                continue
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] >= 2:
                candidates.append((key, arr))
    if not candidates:
        return None, "", []
    candidates.sort(key=lambda item: (
        _covariance_candidate_rank(item[0]),
        -item[1].shape[0],
    ))
    selected_name, selected_array = candidates[0]
    labels = _diagnostic_matrix_labels(Path(output_dir), selected_name)
    return selected_array, selected_name, labels


def _covariance_candidate_rank(name: str) -> int:
    lower = str(name).lower()
    if "network_coupled_covariance" in lower:
        return 0
    if "combined__parameter_covariance" in lower:
        return 1
    if lower.endswith("__covariance") or "__covariance" in lower:
        return 2
    if "numeric_covariance_approx" in lower:
        return 3
    if "q_xi" in lower:
        return 4
    if "correlation" in lower:
        return 5
    if "normal_matrix" in lower:
        return 6
    return 9


def _diagnostic_matrix_labels(output_dir: Union[str, Path], array_name: str) -> List[str]:
    labels_path = Path(output_dir) / "matrix_parameter_labels.csv"
    if not labels_path.exists():
        return []
    try:
        df = pd.read_csv(labels_path)
    except Exception:
        return []
    if df.empty or "array_name" not in df.columns or "parameter" not in df.columns:
        return []
    rows = df[df["array_name"].astype(str) == str(array_name)].copy()
    if rows.empty:
        return []
    if "index" in rows.columns:
        rows = rows.sort_values("index")
    return rows["parameter"].astype(str).tolist()


def _diagnostic_apriori_sigma(output_dir: Union[str, Path], default_mm: float = 3.0) -> float:
    """Return the a priori standard deviation in metres used only for diagnostics."""
    output_dir = Path(output_dir)
    for path in (output_dir / "run_configuration.json", output_dir / "run_configuration.csv"):
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".json":
                config = json.loads(path.read_text(encoding="utf-8"))
                value = config.get("apriori_sigma_m", None)
                if value is None and config.get("apriori_sigma_mm", None) is not None:
                    value = float(config["apriori_sigma_mm"]) / 1000.0
            else:
                config_df = pd.read_csv(path)
                rows = config_df[config_df["input_parameter"].astype(str).isin(["apriori_sigma_m", "apriori_sigma_mm"])]
                value = None
                if not rows.empty:
                    row_map = dict(zip(rows["input_parameter"].astype(str), rows["value"]))
                    if row_map.get("apriori_sigma_m") not in (None, ""):
                        value = row_map.get("apriori_sigma_m")
                    elif row_map.get("apriori_sigma_mm") not in (None, ""):
                        value = float(row_map.get("apriori_sigma_mm")) / 1000.0
            value = float(value)
            if np.isfinite(value) and value > 0:
                return value
        except Exception:
            continue
    return float(default_mm) / 1000.0


def _diagnostic_sigma0_dof(output_dir: Union[str, Path], method_key: Optional[str], covariance: Optional[np.ndarray] = None):
    station_path = Path(output_dir) / "station_outputs.csv"
    residual_path = Path(output_dir) / "residuals.csv"
    if not station_path.exists() or not residual_path.exists():
        return None, None
    try:
        station_df = _method_filtered_frame(pd.read_csv(station_path), method_key)
        residual_df = _method_filtered_frame(pd.read_csv(residual_path), method_key)
    except Exception:
        return None, None
    if station_df.empty or residual_df.empty or "sigma0_sq" not in station_df.columns:
        return None, None
    sigma_values = pd.to_numeric(station_df["sigma0_sq"], errors="coerce").dropna()
    if sigma_values.empty:
        scalar_path = Path(output_dir) / "outputs_scalar_values.csv"
        if scalar_path.exists():
            try:
                scalar_df = _method_filtered_frame(pd.read_csv(scalar_path), method_key)
                scalar_df = scalar_df[scalar_df["variable"].astype(str).isin(["sigma0_sq", "audit_sigma0_sq"])]
                sigma_values = pd.to_numeric(scalar_df["value"], errors="coerce").dropna()
            except Exception:
                sigma_values = pd.Series(dtype=float)
    if sigma_values.empty:
        return None, None
    dof_values = pd.Series(dtype=float)
    if "dof" in station_df.columns:
        dof_values = pd.to_numeric(station_df["dof"], errors="coerce").dropna()
    if not dof_values.empty and float(dof_values.sum()) > 0:
        dof = int(max(1, round(float(dof_values.sum()))))
    else:
        obs_components = len(residual_df) * 3
        n_params = int(covariance.shape[0]) if covariance is not None and covariance.ndim == 2 else 0
        dof = max(1, obs_components - n_params)
    return float(sigma_values.mean()), int(dof)


def _diagnostic_leverage(output_dir: Union[str, Path], method_key: Optional[str]):
    arrays_path = Path(output_dir) / "outputs_full_arrays.npz"
    if not arrays_path.exists():
        return None
    try:
        arrays = np.load(arrays_path)
    except Exception:
        return None
    candidates = []
    for key in arrays.files:
        if method_key:
            key_method = str(key).split("__", 1)[0]
            if key_method != str(method_key):
                continue
        lower = key.lower()
        if "design_matrix" not in lower and "numeric_design" not in lower:
            continue
        try:
            arr = np.asarray(arrays[key], dtype=float)
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape[0] >= arr.shape[1] and arr.shape[0] > 0:
            candidates.append((key, arr))
    if not candidates:
        return None
    candidates.sort(key=lambda item: -item[1].shape[0])
    design = candidates[0][1]
    try:
        hat = design @ np.linalg.pinv(design)
        leverage = np.diag(hat)
    except Exception:
        return None
    leverage = leverage[np.isfinite(leverage)]
    return leverage if leverage.size else None


def _diagnostic_station_covariances(output_dir: Union[str, Path], method_key: Optional[str]) -> Dict[str, np.ndarray]:
    arrays_path = Path(output_dir) / "outputs_full_arrays.npz"
    if not arrays_path.exists():
        return {}
    try:
        arrays = np.load(arrays_path)
    except Exception:
        return {}
    covariances: Dict[str, np.ndarray] = {}
    suffixes = (
        "residual_covariance_approx",
        "residual_covariance",
        "point_covariance",
        "covariance_xyz",
    )
    for key in arrays.files:
        if method_key:
            key_method = str(key).split("__", 1)[0]
            if key_method != str(method_key):
                continue
        lower = key.lower()
        if not any(lower.endswith(suffix) or suffix in lower for suffix in suffixes):
            continue
        try:
            arr = np.asarray(arrays[key], dtype=float)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            continue
        parts = key.split("__")
        station = parts[1] if len(parts) >= 3 else key
        if station not in covariances or "residual_covariance" in lower:
            covariances[str(station)] = arr[:3, :3]
    return covariances


def generate_gui_diagnostic_plots(
    output_dir: Union[str, Path],
    method_key: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Generate analyzer diagnostic plot files for GUI consumption.

    The GUI receives only file metadata from this function and remains
    responsible for listing/opening the generated outputs.
    """
    output_dir = Path(output_dir)
    diagnostic_dir = output_dir / "gui_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    visualizer = SurveyVisualizer()
    generated: List[Dict[str, str]] = []
    messages: List[str] = []
    prefix = method_key or "all"
    title_prefix = method_key or "All methods"

    def add_plot(label: str, path: Optional[Union[str, Path]], description: str):
        if path:
            plot_path = Path(path)
            generated.append({
                "label": label,
                "path": str(plot_path),
                "description": description,
            })

    residuals, fitted, residual_df = _diagnostic_residual_arrays(
        output_dir, method_key)
    if residual_df is not None and not residual_df.empty:
        try:
            add_plot(
                "Network geometry",
                visualizer.plot_network_geometry(
                    residual_df,
                    diagnostic_dir / f"{prefix}_network_geometry.png",
                    title=f"{title_prefix} network geometry (XY)",
                ),
                "XY geometry from observed/reference and adjusted points.",
            )
        except Exception as exc:
            messages.append(
                f"Network geometry plot was not generated: {exc}")

        try:
            add_plot(
                "Observation residuals",
                visualizer.plot_observation_residuals(
                    residual_df,
                    diagnostic_dir / f"{prefix}_observation_residuals.png",
                    title=f"{title_prefix} post-adjustment residuals",
                ),
                "Residual components and residual norm by observation.",
            )
        except Exception as exc:
            messages.append(
                f"Observation residual plot was not generated: {exc}")

        try:
            add_plot(
                "Observation correction vectors",
                visualizer.plot_observation_correction_vectors(
                    residual_df,
                    diagnostic_dir / f"{prefix}_observation_correction_vectors.png",
                    title=f"{title_prefix} observation correction vectors",
                ),
                "Correction vector norm by observation index.",
            )
        except Exception as exc:
            messages.append(
                f"Observation correction vectors plot was not generated: {exc}")

        station_covariances = _diagnostic_station_covariances(
            output_dir, method_key)
        if station_covariances:
            try:
                add_plot(
                    "XY error ellipses",
                    visualizer.plot_xy_error_ellipses(
                        residual_df,
                        station_covariances,
                        diagnostic_dir / f"{prefix}_xy_error_ellipses.png",
                        title=f"{title_prefix} XY error ellipses",
                    ),
                    "Transparent 95% XY ellipses with light principal axes.",
                )
            except Exception as exc:
                messages.append(
                    f"XY error ellipses plot was not generated: {exc}")

        if "v_norm" in residual_df.columns:
            try:
                residual_values = pd.to_numeric(
                    residual_df["v_norm"], errors="coerce").to_numpy(dtype=float)
                point_names = (
                    residual_df["point"].astype(str).tolist()
                    if "point" in residual_df.columns else None
                )
                sigma0 = float(np.nanstd(residual_values))
                sigma0 = sigma0 if np.isfinite(sigma0) and sigma0 > 0 else 1.0
                add_plot(
                    "Control point residuals",
                    visualizer.plot_control_point_residuals(
                        residual_values,
                        point_names=point_names,
                        sigma0=sigma0,
                        output_path=diagnostic_dir / f"{prefix}_control_point_residuals.png",
                        title=f"{title_prefix} control/check point residuals",
                    ),
                    "Histogram and ranked residual bars for CP/ICP checks.",
                )
            except Exception as exc:
                messages.append(
                    f"Control point residual plot was not generated: {exc}")

            try:
                add_plot(
                    "Q-Q envelope",
                    visualizer.plot_qq_with_envelope(
                        residual_values,
                        sigma0=sigma0,
                        n_simulations=500,
                        output_path=diagnostic_dir / f"{prefix}_qq_envelope.png",
                        title=f"{title_prefix} residual Q-Q envelope",
                    ),
                    "Normal Q-Q plot with a simulated 95% envelope.",
                )
            except Exception as exc:
                messages.append(
                    f"Q-Q envelope plot was not generated: {exc}")

        map_required = {"adjusted_x", "adjusted_y", "v_norm"}
        if map_required.issubset(residual_df.columns):
            try:
                map_df = residual_df.copy()
                for col in map_required:
                    map_df[col] = pd.to_numeric(map_df[col], errors="coerce")
                map_df = map_df.dropna(subset=list(map_required))
                labels = map_df["point"].astype(str).tolist() if "point" in map_df.columns else None
                add_plot(
                    "2D residual map",
                    visualizer.plot_residual_map_2d(
                        map_df[["adjusted_x", "adjusted_y"]].to_numpy(dtype=float),
                        map_df["v_norm"].to_numpy(dtype=float),
                        point_labels=labels,
                        output_path=diagnostic_dir / f"{prefix}_residual_map_2d.png",
                        title=f"{title_prefix} 2D residual map",
                    ),
                    "Plan-view residual map colored by residual magnitude.",
                )
            except Exception as exc:
                messages.append(
                    f"2D residual map was not generated: {exc}")

        quiver_required = {"adjusted_x", "adjusted_y", "vx", "vy"}
        if quiver_required.issubset(residual_df.columns):
            try:
                add_plot(
                    "Spatial quiver 2D",
                    visualizer.plot_spatial_quiver_2d(
                        residual_df,
                        diagnostic_dir / f"{prefix}_spatial_quiver_2d.png",
                        scale_quiver=1.0,
                        title=rf"{title_prefix} spatial quiver $(v_x,v_y)$",
                    ),
                    r"Real-position quiver map of planar residuals $(v_x,v_y)$.",
                )
            except Exception as exc:
                messages.append(
                    f"Spatial quiver 2D plot was not generated: {exc}")

        trend_required = {"adjusted_x", "adjusted_y", "adjusted_z", "vx", "vy", "vz"}
        if trend_required.issubset(residual_df.columns):
            try:
                add_plot(
                    "Residual spatial trends",
                    visualizer.plot_residual_spatial_trends(
                        residual_df,
                        diagnostic_dir / f"{prefix}_residual_spatial_trends.png",
                    ),
                    r"Trend diagnostics $v_x(X)$, $v_y(Y)$ and $v_z(Z)$.",
                )
            except Exception as exc:
                messages.append(
                    f"Residual spatial trend plot was not generated: {exc}")

    station_path = output_dir / "station_outputs.csv"
    if station_path.exists():
        try:
            station_df = pd.read_csv(station_path)
            station_df = _method_filtered_frame(station_df, method_key)
            add_plot(
                "Station transformation parameters",
                visualizer.plot_station_transformation_parameters(
                    station_df,
                    output_path=diagnostic_dir / f"{prefix}_station_parameters.png",
                    title=f"{title_prefix} station transformation parameters",
                ),
                "Station translation parameters from analyzer station outputs.",
            )
        except Exception as exc:
            messages.append(
                f"Station transformation parameters plot was not generated: {exc}")

    outlier_path = output_dir / "outlier_flags.csv"
    if outlier_path.exists() and outlier_path.stat().st_size > 0:
        try:
            outlier_df = pd.read_csv(outlier_path)
            if outlier_df.empty:
                outlier_df = pd.DataFrame()
            outlier_df = _method_filtered_frame(outlier_df, method_key)
            if not outlier_df.empty:
                add_plot(
                    "Outlier Pareto",
                    visualizer.plot_pareto_outliers(
                        outlier_df,
                        group_by="point",
                        output_path=diagnostic_dir / f"{prefix}_outlier_pareto.png",
                        title=f"{title_prefix} outlier candidate Pareto",
                    ),
                    "Pareto chart of outlier candidate frequency by point.",
                )
        except pd.errors.EmptyDataError:
            pass
        except Exception as exc:
            messages.append(
                f"Outlier Pareto plot was not generated: {exc}")

    if residuals is not None and residuals.size >= 3:
        sigma0 = float(np.nanstd(residuals))
        sigma0 = sigma0 if np.isfinite(sigma0) and sigma0 > 0 else 1.0
        try:
            add_plot(
                "Normalized residuals",
                visualizer.plot_normalized_residuals(
                    residuals,
                    sigma0=sigma0,
                    output_path=diagnostic_dir / f"{prefix}_normalized_residuals.png",
                    title=f"{title_prefix} normalized residuals",
                ),
                "Histogram and Q-Q plot for residual normality.",
            )
        except Exception as exc:
            messages.append(
                f"Normalized residuals plot was not generated: {exc}")

        if fitted is not None and fitted.size == residuals.size:
            try:
                add_plot(
                    "Residuals vs fitted",
                    visualizer.plot_residuals_vs_fitted(
                        residuals,
                        fitted,
                        output_path=diagnostic_dir / f"{prefix}_residuals_vs_fitted.png",
                    ),
                    "Residual spread against adjusted coordinates.",
                )
            except Exception as exc:
                messages.append(
                    f"Residuals vs fitted plot was not generated: {exc}")

        if residuals.size >= 8:
            try:
                add_plot(
                    "Residual autocorrelation",
                    visualizer.plot_residual_autocorrelation(
                        residuals,
                        lags=min(20, max(1, residuals.size // 4)),
                        output_path=diagnostic_dir / f"{prefix}_residual_autocorrelation.png",
                    ),
                    "ACF plot for ordered residual independence.",
                )
            except Exception as exc:
                messages.append(
                    f"Residual autocorrelation plot was not generated: {exc}")

        required = {"adjusted_x", "adjusted_y",
                    "adjusted_z", "vx", "vy", "vz", "point"}
        if not residual_df.empty and required.issubset(residual_df.columns):
            coords = {}
            vectors = {}
            numeric_cols = ["adjusted_x", "adjusted_y",
                            "adjusted_z", "vx", "vy", "vz"]
            valid_df = residual_df.copy()
            for col in numeric_cols:
                valid_df[col] = pd.to_numeric(
                    valid_df[col], errors="coerce")
            for _, row in valid_df.dropna(subset=numeric_cols + ["point"]).head(500).iterrows():
                point_id = f"{row.get('station', '')}_{row.get('point', '')}"
                coords[point_id] = np.asarray([
                    float(row["adjusted_x"]),
                    float(row["adjusted_y"]),
                    float(row["adjusted_z"]),
                ], dtype=float)
                vectors[point_id] = np.asarray([
                    float(row["vx"]),
                    float(row["vy"]),
                    float(row["vz"]),
                ], dtype=float)
            if coords:
                try:
                    add_plot(
                        "3D residual vectors",
                        visualizer.create_3d_residual_plot_html(
                            coords,
                            vectors,
                            diagnostic_dir / f"{prefix}_residual_vectors_3d.html",
                            title=f"{title_prefix} residual vectors",
                        ),
                        "Interactive residual vectors from adjusted points.",
                    )
                except Exception as exc:
                    messages.append(
                        f"3D residual vectors were not generated: {exc}")

    covariance, covariance_name, covariance_labels = _diagnostic_covariance_array(
        output_dir, method_key)
    if covariance is not None:
        sigma0_sq, dof = _diagnostic_sigma0_dof(output_dir, method_key, covariance)
        sigma0_apriori = _diagnostic_apriori_sigma(output_dir)
        if sigma0_sq is not None and dof is not None:
            try:
                add_plot(
                    "Global chi-square test",
                    visualizer.plot_global_chi_square_test(
                        sigma0_sq,
                        dof,
                        alpha=0.05,
                        sigma0_apriori=sigma0_apriori,
                        output_path=diagnostic_dir / f"{prefix}_global_chi_square_test.png",
                    ),
                    r"Global variance-factor test using $\chi^2_\nu$ and the configured a priori precision.",
                )
            except Exception as exc:
                messages.append(
                    f"Global chi-square test plot was not generated: {exc}")

        try:
            max_size = min(covariance.shape[0], 80)
            cov_plot = covariance[:max_size, :max_size]
            param_names = covariance_labels[:max_size] if len(covariance_labels) >= max_size else None
            add_plot(
                "Organized covariance",
                visualizer.plot_covariance_organized(
                    cov_plot,
                    param_names=param_names,
                    output_path=diagnostic_dir / f"{prefix}_organized_covariance.png",
                    show_values=max_size <= 15,
                    show_dendrogram=max_size <= 60,
                    title=f"{title_prefix} covariance/correlation ({covariance_name})",
                ),
                "Clustered covariance/correlation heatmap.",
            )
        except Exception as exc:
            messages.append(
                f"Organized covariance plot was not generated: {exc}")

    leverage = _diagnostic_leverage(output_dir, method_key)
    if leverage is not None:
        try:
            add_plot(
                "Local redundancy numbers",
                visualizer.plot_local_redundancy_numbers(
                    leverage,
                    output_path=diagnostic_dir / f"{prefix}_local_redundancy_numbers.png",
                ),
                r"Baarda local reliability numbers $r_i=1-h_{ii}$.",
            )
        except Exception as exc:
            messages.append(
                f"Local redundancy numbers plot was not generated: {exc}")

    return generated, messages


# ----------------------------------------------------------------------
# Module exports
# ----------------------------------------------------------------------
__all__ = [
    "SurveyVisualizer",
    "generate_gui_diagnostic_plots",
    "load_method_summary",
    "method_code",
    "method_metadata",
    "setup_matplotlib_style",
    "plot_error_ellipsoid",
    "run_label_order",
    "select_method_summary",
    "standardize_adjustment_outputs",
    "_apply_plotly_layout",
    "create_3d_scatter_html",
    "create_3d_ellipsoid_html",
    "create_3d_network_html",
    "PLOTLY_AVAILABLE",
    # Academic plots
    "plot_covariance_matrix_organized",
    "plot_network_geometry",
    "plot_observation_correction_vectors",
    "plot_observation_residuals",
    "plot_xy_error_ellipses",
    "plot_control_point_residuals",
    "plot_global_chi_square_test",
    "plot_spatial_quiver_2d",
    "plot_local_redundancy_numbers",
    "plot_residual_spatial_trends",
    "plot_pareto_outliers",
    "plot_qq_with_envelope",
    "plot_residual_map_2d",
    "plot_station_transformation_parameters",
    "plot_normalized_residuals",
    "plot_influence_diagnostics",
    "plot_residual_autocorrelation",
    "plot_residuals_vs_fitted",
]
