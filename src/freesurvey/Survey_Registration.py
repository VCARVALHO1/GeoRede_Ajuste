"""
SURVEY REGISTRATION WORKFLOWS

Single source for sequential registration and network adjustment workflows.
"""

# ===========================================================================
# SEQUENTIAL REGISTRATION WORKFLOW
# ===========================================================================
from freesurvey.Registration_Module import (
    Method1_SequentialSVD,
    Schut1973_SpaceTime,
    Stepien2020_QST,
    Method3_GlobalQuaternionsLM_ObsEquations,
    Method4_GlobalEulerLM,
    Tao2020_CMTLS_Rigorous,
    TotalFreeStation_TFS,
    evaluate_rmse,
    create_correlation_matrix,
    propagate_point_covariance,
    baarda_data_snooping_adjustment,
)
from freesurvey.data_manager import DataManager, extract_station_number
from freesurvey.results_analyzer import plot_error_ellipsoid
from collections import defaultdict
import sys
import os
import logging
import numpy as np
import pandas as pd
import json
import re
from datetime import datetime
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FOLDER = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "outputs")

GLOBAL_CP_MIN = 0
MIN_PAIRED_CPS = 4

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ===========================================================================
# SECTION 1 - READ DATA
# ===========================================================================

def load_station_data(data_folder: str):
    """Compatibility wrapper around the central DataManager."""
    survey_data = DataManager().load_station_folder(data_folder, delimiter="auto")
    stations = list(getattr(survey_data, "stations", {}).values())
    logger.info("Loaded %d station files in correct order.", len(stations))
    for st in stations:
        logger.info(f"  {st['name']} (number: {st['number']})")
    return stations


def get_coords_for_cps(station, cp_ids):
    """Extract coordinates for given CP IDs."""
    coords = []
    for pid in cp_ids:
        try:
            idx = station["all_ids"].index(pid)
            coords.append(station["coords"][idx])
        except ValueError:
            coords.append(np.full(3, np.nan))
    return np.array(coords)


def _numeric_design_matrix(source, scale, rotation, translation):
    """Numerically approximate the similarity-transformation design matrix."""
    base_params = np.concatenate([[scale], rotation.ravel(), translation])

    def transform(params):
        k = params[0]
        r = params[1:10].reshape(3, 3)
        t = params[10:13]
        return (k * (source @ r.T) + t).ravel()

    base = transform(base_params)
    design = np.zeros((base.size, base_params.size))
    eps = 1e-6
    for idx in range(base_params.size):
        step = np.zeros_like(base_params)
        step[idx] = eps
        design[:, idx] = (transform(base_params + step) - base) / eps
    return design


def _schut_quaternion_diagnostics(source_centered, target_centered, scale):
    scaled_source = source_centered * scale
    m = scaled_source.T @ target_centered
    n = np.zeros((4, 4), dtype=float)
    n[0, 0] = m[0, 0] + m[1, 1] + m[2, 2]
    n[1, 1] = m[0, 0] - m[1, 1] - m[2, 2]
    n[2, 2] = -m[0, 0] + m[1, 1] - m[2, 2]
    n[3, 3] = -m[0, 0] - m[1, 1] + m[2, 2]
    n[0, 1] = n[1, 0] = m[1, 2] - m[2, 1]
    n[0, 2] = n[2, 0] = m[2, 0] - m[0, 2]
    n[0, 3] = n[3, 0] = m[0, 1] - m[1, 0]
    n[1, 2] = n[2, 1] = m[0, 1] + m[1, 0]
    n[1, 3] = n[3, 1] = m[2, 0] + m[0, 2]
    n[2, 3] = n[3, 2] = m[1, 2] + m[2, 1]
    eigenvalues, eigenvectors = np.linalg.eigh(n)
    quaternion = eigenvectors[:, np.argmax(eigenvalues)]
    norm = np.linalg.norm(quaternion)
    if norm:
        quaternion = quaternion / norm
    return m, n, eigenvalues, eigenvectors, quaternion


def _distance_design(source, target):
    pair_i = []
    pair_j = []
    d_source = []
    d_target = []
    for i in range(len(source)):
        for j in range(i + 1, len(source)):
            pair_i.append(i)
            pair_j.append(j)
            d_source.append(np.linalg.norm(source[j] - source[i]))
            d_target.append(np.linalg.norm(target[j] - target[i]))
    return (
        np.asarray(pair_i, dtype=int),
        np.asarray(pair_j, dtype=int),
        np.asarray(d_source, dtype=float).reshape(-1, 1),
        np.asarray(d_target, dtype=float).reshape(-1, 1),
    )


def _build_coordinate_observation_metadata(point_ids, station_id="", method="", scenario=""):
    components = ("X", "Y", "Z")
    rows = []
    for point_index, point_id in enumerate(point_ids):
        for component in components:
            rows.append({
                "observation_index": len(rows),
                "point_id": str(point_id),
                "station_id": str(station_id),
                "component": component,
                "method": method,
                "scenario": scenario,
                "original_index": point_index,
            })
    return rows


def _run_baarda_diagnostics(
    design_matrix,
    residuals_xyz,
    point_ids,
    station_id,
    method_name,
    scenario,
    use_baarda_data_snooping=False,
    baarda_threshold=3.0,
    baarda_max_iterations=10,
    baarda_min_observations_per_point=4,
    baarda_remove_strategy="none",
    baarda_use_posterior_sigma=True,
):
    if not use_baarda_data_snooping:
        return {"enabled": False, "message": "Baarda Data Snooping disabled."}
    if design_matrix is None or np.asarray(design_matrix).size == 0:
        logger.warning("Baarda Data Snooping not available for this method because adjustment matrices were not returned.")
        return {
            "enabled": True,
            "threshold": baarda_threshold,
            "iterations": [],
            "removed_observations": [],
            "removed_points": [],
            "final_max_w": np.nan,
            "final_sigma0_hat": np.nan,
            "converged": False,
            "message": "Baarda Data Snooping not available for this method because adjustment matrices were not returned.",
        }

    residual_vector = np.asarray(residuals_xyz, dtype=float).reshape(-1)
    metadata = _build_coordinate_observation_metadata(
        point_ids,
        station_id=station_id,
        method=method_name,
        scenario=scenario,
    )
    if len(metadata) != residual_vector.size:
        logger.warning("Baarda metadata does not match residual vector size; diagnostics skipped.")
        return {
            "enabled": True,
            "threshold": baarda_threshold,
            "iterations": [],
            "removed_observations": [],
            "removed_points": [],
            "final_max_w": np.nan,
            "final_sigma0_hat": np.nan,
            "converged": False,
            "message": "Observation metadata did not match residual vector size.",
        }

    # The diagnostic adjustment solves A dX ~= -v, so the resulting residual
    # keeps the transformed-minus-target convention used by the registration code.
    _, diagnostics = baarda_data_snooping_adjustment(
        np.asarray(design_matrix, dtype=float),
        -residual_vector,
        P=None,
        observation_metadata=metadata,
        threshold=baarda_threshold,
        max_iterations=baarda_max_iterations,
        min_observations_per_point=baarda_min_observations_per_point,
        remove_strategy=baarda_remove_strategy,
        use_posterior_sigma=baarda_use_posterior_sigma,
    )
    diagnostics["residual_convention"] = "v = transformed - target"
    return diagnostics


def _collect_baarda_rows(result):
    report_rows = []
    final_rows = []
    if not result:
        return report_rows, final_rows
    for station in result.get("stations", []):
        qm = station.get("quality_metrics") or {}
        diag = qm.get("baarda_diagnostics") or {}
        report_rows.extend(diag.get("outlier_report_rows", []))
        final_rows.extend(diag.get("final_residual_rows", []))
    diag = result.get("baarda_diagnostics") or {}
    report_rows.extend(diag.get("outlier_report_rows", []))
    final_rows.extend(diag.get("final_residual_rows", []))
    return report_rows, final_rows


def export_baarda_outputs(result, output_folder, method_name):
    report_rows, final_rows = _collect_baarda_rows(result)
    if not report_rows and not final_rows:
        return {}
    os.makedirs(output_folder, exist_ok=True)
    paths = {}
    if report_rows:
        report_file = os.path.join(output_folder, "baarda_outlier_report.csv")
        pd.DataFrame(report_rows).to_csv(report_file, index=False)
        paths["outlier_report"] = report_file
    if final_rows:
        final_file = os.path.join(output_folder, "baarda_final_residuals.csv")
        pd.DataFrame(final_rows).to_csv(final_file, index=False)
        paths["final_residuals"] = final_file
    _plot_baarda_diagnostics(final_rows, report_rows, output_folder, method_name)
    return paths


def _save_current_figure(fig, output_folder, base_name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(output_folder, f"{base_name}.{ext}"), dpi=300, bbox_inches="tight")


def _plot_baarda_diagnostics(final_rows, report_rows, output_folder, method_name):
    if plt is None or not final_rows:
        return
    df = pd.DataFrame(final_rows)
    if df.empty:
        return

    threshold = np.nan
    if report_rows:
        threshold = pd.DataFrame(report_rows).get("threshold", pd.Series([np.nan])).iloc[0]

    fig, ax = plt.subplots(figsize=(12, 5))
    values = df["standardized_residual_w"].abs()
    ax.bar(np.arange(len(values)), values)
    if np.isfinite(threshold):
        ax.axhline(float(threshold), color="red", linestyle="--", label=f"threshold={threshold:g}")
        ax.legend()
    ax.set_title(f"Baarda |w| by observation - {method_name}")
    ax.set_xlabel("Observation")
    ax.set_ylabel("|w|")
    _save_current_figure(fig, output_folder, "baarda_w_values_by_observation")
    plt.close(fig)

    if "point_id" in df:
        by_point = df.assign(abs_w=values).groupby("point_id", dropna=False)["abs_w"].max().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 5))
        by_point.plot(kind="bar", ax=ax)
        if np.isfinite(threshold):
            ax.axhline(float(threshold), color="red", linestyle="--")
        ax.set_title(f"Baarda max |w| by point - {method_name}")
        ax.set_xlabel("Point")
        ax.set_ylabel("max |w|")
        _save_current_figure(fig, output_folder, "baarda_w_values_by_point")
        plt.close(fig)

    if report_rows:
        hist = pd.DataFrame(report_rows)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(hist["iteration"], hist["standardized_residual_w"].abs(), marker="o", label="max |w|")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("max |w|")
        ax2 = ax1.twinx()
        if "sigma0_hat" in hist:
            ax2.plot(hist["iteration"], hist["sigma0_hat"], color="tab:orange", marker="s", label="sigma0_hat")
        ax2.set_ylabel("sigma0_hat")
        ax1.set_title(f"Baarda iteration history - {method_name}")
        _save_current_figure(fig, output_folder, "baarda_iteration_history")
        plt.close(fig)

    if "redundancy_number" in df:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(np.arange(len(df)), df["redundancy_number"])
        ax.set_title(f"Baarda redundancy numbers - {method_name}")
        ax.set_xlabel("Observation")
        ax.set_ylabel("r_i")
        _save_current_figure(fig, output_folder, "baarda_redundancy_numbers")
        plt.close(fig)

    # Coordinates are not always present in the generic residual table. Create
    # an empty-but-valid diagnostic figure when a map cannot be drawn safely.
    fig, ax = plt.subplots(figsize=(7, 6))
    removed = pd.DataFrame(report_rows) if report_rows else pd.DataFrame()
    ax.set_title(f"Baarda removed observations map - {method_name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.text(0.5, 0.5, "XY coordinates not available", ha="center", va="center", transform=ax.transAxes)
    if not removed.empty:
        ax.text(0.5, 0.42, f"Removed/candidate observations: {int(removed['removed'].sum())}", ha="center", va="center", transform=ax.transAxes)
    _save_current_figure(fig, output_folder, "baarda_removed_observations_map")
    plt.close(fig)


def _build_quality_output_matrices(source, target, scale, rotation, translation, residuals_xyz, method_name):
    """Create common matrices and diagnostics for every sequential method."""
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    residual_norms = np.linalg.norm(residuals_xyz, axis=1)
    dof = max(1, 3 * len(source) - 7)
    sigma0_sq = float(np.sum(residual_norms ** 2) / dof)
    design = _numeric_design_matrix(source, scale, rotation, translation)
    normal = design.T @ design
    covariance = sigma0_sq * np.linalg.pinv(normal)
    observation_covariance = np.eye(3 * len(source)) * sigma0_sq
    residual_covariance = np.cov(residuals_xyz.T) if len(source) > 1 else np.zeros((3, 3))
    pair_i, pair_j, d_source, d_target = _distance_design(source, target)
    distance_residuals = scale * d_source - d_target
    quasi_rotation = target_centered.T @ source_centered @ np.linalg.pinv(source_centered.T @ source_centered)
    qr_u, qr_s, qr_vt = np.linalg.svd(quasi_rotation)
    schut_m, schut_n, schut_evals, schut_evecs, schut_q = _schut_quaternion_diagnostics(
        source_centered, target_centered, scale
    )
    return {
        "input_source_coordinates": source,
        "input_target_coordinates": target,
        "source_centroid": source_centroid,
        "target_centroid": target_centroid,
        "source_centered": source_centered,
        "target_centered": target_centered,
        "cross_covariance_h": source_centered.T @ target_centered,
        "design_matrix": design,
        "normal_matrix": normal,
        "covariance": covariance,
        "correlation_matrix": create_correlation_matrix(covariance),
        "observation_covariance": observation_covariance,
        "covariance_residuals": residual_covariance,
        "weight_matrix": np.linalg.pinv(observation_covariance),
        "Q_matrix": covariance,
        "numeric_design_matrix": design,
        "numeric_normal_matrix": normal,
        "numeric_covariance_approx": covariance,
        "numeric_correlation_approx": create_correlation_matrix(covariance),
        "distance_pair_i": pair_i,
        "distance_pair_j": pair_j,
        "distance_design_matrix": d_source,
        "distance_observation_vector": d_target,
        "distance_residuals": distance_residuals,
        "quasi_rotation_matrix": quasi_rotation,
        "quasi_rotation_svd_u": qr_u,
        "quasi_rotation_svd_singular_values": qr_s,
        "quasi_rotation_svd_vt": qr_vt,
        "schut_cross_covariance_m": schut_m,
        "schut_quaternion_matrix_n": schut_n,
        "schut_eigenvalues": schut_evals,
        "schut_eigenvectors": schut_evecs,
        "schut_quaternion": schut_q,
        "residual_sum_squares": float(np.sum(residual_norms ** 2)),
        "degrees_of_freedom": dof,
        "audit_sigma0_sq": sigma0_sq,
        "audit_sigma0": float(np.sqrt(sigma0_sq)),
        "method_diagnostic_name": method_name,
    }


# ===========================================================================
# SECTION 2 - SEQUENTIAL REGISTRATION
# ===========================================================================

def register_station_to_reference(source_station, target_station,
                                   common_cps, method_class, min_paired_cps=4,
                                   use_baarda_data_snooping=False,
                                   baarda_threshold=3.0,
                                   baarda_max_iterations=10,
                                   baarda_min_observations_per_point=4,
                                   baarda_remove_strategy="none",
                                   baarda_use_posterior_sigma=True):
    """
    Register source_station to target_station's coordinate system.
    """
    S = get_coords_for_cps(source_station, common_cps)
    T = get_coords_for_cps(target_station, common_cps)

    valid = ~(np.isnan(S).any(axis=1) | np.isnan(T).any(axis=1))
    S_valid = S[valid]
    T_valid = T[valid]

    if len(S_valid) < min_paired_cps:
        raise ValueError(
            f"Only {len(S_valid)} common CPs (< {min_paired_cps})")

    scale, R, trans, elapsed = None, None, None, None
    stats = {}
    point_uncertainties = {}

    if getattr(method_class, "is_global_lm", False):
        solve_res = method_class.solve([S_valid], T_valid)
        if len(solve_res) == 5:
            res_list, sig, cov, residuals, elapsed = solve_res
            matrices = {}
        elif len(solve_res) == 6:
            # cov here is for parameters
            res_list, sig, cov, residuals, elapsed, matrices = solve_res
        else:
            res_list, sig, cov, residuals, elapsed = solve_res[:5]
            matrices = {}

        scale, R, trans = res_list[0]
        stats = {'sigma0_sq': sig, 'sigma0': np.sqrt(
            sig), 'residuals': residuals, 'covariance': cov}
        stats.update(matrices)

        if getattr(method_class, "parameterization", "") == "quaternion":
            from freesurvey.Registration_Module import (
                Method3_GlobalQuaternionsLM_ObsEquations,
                Rotation,
                propagate_point_covariance,
            )
            r = Rotation.from_matrix(R)
            quat_xyzw = r.as_quat()
            q0, q1, q2, q3 = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
            if q0 < 0:
                q0, q1, q2, q3 = -q0, -q1, -q2, -q3
            p = np.array([scale, q0, q1, q2, q3, trans[0], trans[1], trans[2]])
            for j, pt_id in enumerate(source_station['all_ids']):
                try:
                    pt_cov = propagate_point_covariance(
                        source_station['coords'][j], p, cov,
                        Method3_GlobalQuaternionsLM_ObsEquations._quat_to_matrix)
                    point_uncertainties[pt_id] = pt_cov
                except Exception as e:
                    logger.warning(
                        f"Failed to propagate covariance for point {pt_id} in {source_station['name']}: {e}")
                    point_uncertainties[pt_id] = np.zeros((3, 3))

    elif method_class.__name__ == "Tao2020_CMTLS_Rigorous":
        scale, R, trans, Q, stats, elapsed = method_class.solve(
            S_valid, T_valid)
        stats = {'Q_matrix': Q, **stats}

        # If Tao2020, correlation matrix is in stats
        if 'correlation_matrix' in stats:
            stats['correlation_matrix'] = stats['correlation_matrix']

    elif (
        method_class.__name__ in {"Stepien2020_QST", "Q_ST", "TotalFreeStation_TFS", "TFS"}
        or getattr(method_class, "method_code", "") in {"Q-ST", "TFS"}
    ):
        scale, R, trans, stats, elapsed = method_class.solve(S_valid, T_valid)
    else:
        scale, R, trans, elapsed = method_class.solve(S_valid, T_valid)
        stats = {}

    transformed_all = (scale * source_station["coords"] @ R.T) + trans
    transformed_control = scale * (S_valid @ R.T) + trans
    residuals_fit = transformed_control - T_valid
    residual_norms = np.linalg.norm(residuals_fit, axis=1)
    rmse = evaluate_rmse(S_valid, T_valid, scale, R, trans)
    valid_control_ids = [cp for cp, is_valid in zip(common_cps, valid) if is_valid]

    matrix_outputs = _build_quality_output_matrices(
        S_valid, T_valid, scale, R, trans, residuals_fit, method_class.__name__
    )
    baarda_diagnostics = _run_baarda_diagnostics(
        matrix_outputs.get("design_matrix"),
        residuals_fit,
        valid_control_ids,
        source_station.get("name", ""),
        method_class.__name__,
        "sequential_registration",
        use_baarda_data_snooping=use_baarda_data_snooping,
        baarda_threshold=baarda_threshold,
        baarda_max_iterations=baarda_max_iterations,
        baarda_min_observations_per_point=baarda_min_observations_per_point,
        baarda_remove_strategy=baarda_remove_strategy,
        baarda_use_posterior_sigma=baarda_use_posterior_sigma,
    )
    stats = {**matrix_outputs, **stats}
    stats["baarda_diagnostics"] = baarda_diagnostics
    if stats.get("covariance") is not None and "correlation_matrix" not in stats:
        stats["correlation_matrix"] = create_correlation_matrix(stats["covariance"])

    # Calculate additional quality metrics from transformed-minus-target residuals.
    mean_residual = np.mean(residual_norms)
    std_residual = np.std(residual_norms)
    max_residual = np.max(residual_norms)

    quality_metrics = {
        'rmse': rmse,
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'max_residual': max_residual,
        'n_control_points': len(S_valid),
        'scale': scale,
        'translation': trans,
        'rotation_matrix': R,
        'control_point_ids': valid_control_ids,
        'source_control_points': S_valid,
        'target_control_points': T_valid,
        'transformed_control_points': transformed_control,
        'residuals_xyz': residuals_fit,
        'residual_norms': residual_norms,
        'observation_residuals': residuals_fit.ravel(),
        'observation_vector': T_valid.ravel(),
        'sigma0_sq': stats.get('sigma0_sq', stats.get('audit_sigma0_sq')),
        'sigma0': stats.get('sigma0', stats.get('audit_sigma0')),
        'rotation_determinant': float(np.linalg.det(R)),
        'orthogonality_error': float(np.linalg.norm(R.T @ R - np.eye(3))),
        **stats,
        'point_uncertainties': point_uncertainties  # Add point uncertainties here
    }

    return transformed_all, (scale, R, trans), rmse, elapsed, quality_metrics


def build_accumulated_reference(registered_dict):
    """Build one accumulated reference from all adjusted stations.

    Repeated point IDs are averaged so a later station is fitted against the
    whole adjusted network, not against the first station that happened to
    contain that point.
    """
    point_coords = defaultdict(list)
    ctrl_ids = set()
    id_order = []

    for ref_data in registered_dict.values():
        coords = np.asarray(ref_data.get('coords', []), dtype=float)
        ids = list(ref_data.get('all_ids', []))
        ctrl_ids.update(ref_data.get('ctrl_ids', []))
        for pid, coord in zip(ids, coords):
            if pid not in point_coords:
                id_order.append(pid)
            point_coords[pid].append(coord)

    if point_coords:
        all_ids = id_order
        coords = np.vstack([
            np.mean(np.asarray(point_coords[pid], dtype=float), axis=0)
            for pid in all_ids
        ])
        return {
            'coords': coords,
            'all_ids': all_ids,
            'ctrl_ids': sorted(ctrl_ids),
            'point_observation_counts': {pid: len(vals) for pid, vals in point_coords.items()}
        }
    return None


def _failed_station_record(station, error, common_count=0, attempt_count=0):
    """Create a consistent failed-station result entry."""
    return {
        'name': station['name'],
        'number': station['number'],
        'coords': None,
        'coords_original': station['coords'].copy(),
        'is_reference': False,
        'error': error,
        'all_ids': station['all_ids'].copy(),
        'ctrl_ids': station['ctrl_ids'].copy(),
        'registration_order': -1,
        'rmse': np.nan,
        'time': np.nan,
        'reference_used': None,
        'quality_metrics': None,
        'transform': None,
        'common_points_used': common_count,
        'attempt_count': attempt_count,
    }


def sequential_registration(stations, method_class, selected_control_points=None, min_paired_cps=4,
                            use_baarda_data_snooping=False,
                            baarda_threshold=3.0,
                            baarda_max_iterations=10,
                            baarda_min_observations_per_point=4,
                            baarda_remove_strategy="none",
                            baarda_use_posterior_sigma=True):
    """
    Sequentially register all stations in order E1 -> E2 -> E3 -> ...

    Strategy:
    1. Keep Station 1 (E1) as fixed reference
    2. Register Station 2 (E2) to Station 1
    3. Register Station 3 (E3) to accumulated reference (E1+E2)
    4. Continue for all stations in order

    Args:
        stations: List of station dictionaries
        method_class: Registration method class to use
        selected_control_points: Optional list of specific control point IDs to use
        min_paired_cps: Minimum number of common control points required (default: 4)
    """
    if len(stations) < 2:
        raise ValueError(
            "Need at least 2 stations for sequential registration")

    # Results storage
    registered_stations = []

    # Station 1 (E1) is the fixed reference
    reference_station = stations[0].copy()
    reference_quality_metrics = {
        'rmse': None,
        'mean_residual': None,
        'std_residual': None,
        'max_residual': None,
        'n_control_points': len(reference_station['ctrl_ids']),
        'scale': 1.0,
        'translation': np.zeros(3),
        'rotation_matrix': np.eye(3),
        'rotation_determinant': 1.0,
        'orthogonality_error': 0.0,
    }
    registered_stations.append({
        'name': reference_station['name'],
        'number': reference_station['number'],
        'coords': reference_station['coords'].copy(),
        'coords_original': reference_station['coords'].copy(),
        'is_reference': True,
        'original_coords': reference_station['coords'].copy(),
        'all_ids': reference_station['all_ids'].copy(),
        'ctrl_ids': reference_station['ctrl_ids'].copy(),
        'registration_order': 0,
        'rmse': None,
        'time': None,
        'reference_used': None,
        'quality_metrics': reference_quality_metrics,
        'transform': (1.0, np.eye(3), np.zeros(3))
    })

    # Dictionary of successfully registered stations
    registered_dict = {
        reference_station['name']: {
            'coords': reference_station['coords'].copy(),
            'all_ids': reference_station['all_ids'].copy(),
            'ctrl_ids': reference_station['ctrl_ids'].copy(),
            'number': reference_station['number']
        }
    }

    print(f"\n{'='*70}")
    print(f"SEQUENTIAL REGISTRATION using {method_class.__name__}")
    print(f"{'='*70}")
    print(f"Reference: {reference_station['name']} (fixed)")
    print(f"Processing order: {' -> '.join([s['name'] for s in stations])}")

    pending_stations = list(stations[1:])
    failed_attempts = {}
    attempt_counts = defaultdict(int)
    registration_order = 1
    pass_number = 1

    while pending_stations:
        print(f"\n--- Sequential pass {pass_number}: {len(pending_stations)} station(s) pending")
        progress = False
        next_pending = []

        for current_station in pending_stations:
            attempt_counts[current_station['name']] += 1
            current_number = current_station['number']

            print(f"\n--- Trying {current_station['name']} "
                  f"(station {current_number}, attempt {attempt_counts[current_station['name']]})")

            accumulated_ref = build_accumulated_reference(registered_dict)
            if accumulated_ref is None:
                error = 'No accumulated reference available'
                print(f"  WAIT: {error}")
                failed_attempts[current_station['name']] = _failed_station_record(
                    current_station, error, 0, attempt_counts[current_station['name']]
                )
                next_pending.append(current_station)
                continue

            common_points = [
                pid for pid in current_station['all_ids']
                if pid in accumulated_ref['all_ids']
            ]
            if selected_control_points:
                selected = set(selected_control_points)
                common_points = [pid for pid in common_points if pid in selected]

            print(
                f"  Found {len(common_points)} common adjusted point(s) with accumulated reference")

            if len(common_points) < min_paired_cps:
                error = f'Insufficient common points: {len(common_points)} < {min_paired_cps}'
                print(f"  WAIT: {error}; station will be retried after more stations are adjusted")
                failed_attempts[current_station['name']] = _failed_station_record(
                    current_station, error, len(common_points), attempt_counts[current_station['name']]
                )
                next_pending.append(current_station)
                continue

            try:
                transformed_coords, transform, rmse, elapsed, quality_metrics = register_station_to_reference(
                    current_station,
                    accumulated_ref,
                    common_points,
                    method_class,
                    min_paired_cps,
                    use_baarda_data_snooping=use_baarda_data_snooping,
                    baarda_threshold=baarda_threshold,
                    baarda_max_iterations=baarda_max_iterations,
                    baarda_min_observations_per_point=baarda_min_observations_per_point,
                    baarda_remove_strategy=baarda_remove_strategy,
                    baarda_use_posterior_sigma=baarda_use_posterior_sigma,
                )

                print("  OK: Registration successful!")
                print(f"    RMSE on common points: {rmse:.6f} m")
                print(
                    f"    Mean residual: {quality_metrics['mean_residual']:.6f} m")
                print(f"    Scale factor: {transform[0]:.6f}")
                print(f"    Time: {elapsed:.4f} s")

                registered_stations.append({
                    'name': current_station['name'],
                    'number': current_station['number'],
                    'coords': transformed_coords,
                    'coords_original': current_station['coords'].copy(),
                    'is_reference': False,
                    'original_coords': current_station['coords'].copy(),
                    'transform': transform,
                    'all_ids': current_station['all_ids'].copy(),
                    'ctrl_ids': current_station['ctrl_ids'].copy(),
                    'registration_order': registration_order,
                    'rmse': rmse,
                    'time': elapsed,
                    'reference_used': 'accumulated_all_adjusted_points',
                    'quality_metrics': quality_metrics,
                    'common_points_used': len(common_points),
                    'attempt_count': attempt_counts[current_station['name']],
                })
                registration_order += 1

                registered_dict[current_station['name']] = {
                    'coords': transformed_coords,
                    'all_ids': current_station['all_ids'].copy(),
                    'ctrl_ids': current_station['ctrl_ids'].copy(),
                    'number': current_station['number']
                }
                failed_attempts.pop(current_station['name'], None)
                progress = True

                print(
                    f"    Accumulated reference now has {len(registered_dict)} stations")

            except Exception as e:
                error = str(e)
                print(f"  WAIT: Registration failed: {error}; station will be retried")
                failed_attempts[current_station['name']] = _failed_station_record(
                    current_station, error, len(common_points), attempt_counts[current_station['name']]
                )
                next_pending.append(current_station)

        if not progress:
            print("\nNo additional station could be registered in this pass. Finalizing remaining failures.")
            for station in next_pending:
                registered_stations.append(failed_attempts.get(
                    station['name'],
                    _failed_station_record(
                        station,
                        'Registration did not converge after retry passes',
                        0,
                        attempt_counts[station['name']],
                    )
                ))
            break

        pending_stations = next_pending
        pass_number += 1

    last_success = next((s for s in reversed(
        registered_stations) if s.get('quality_metrics')), None)
    qm = last_success['quality_metrics'] if last_success else {}
    all_residuals = []
    for station in registered_stations:
        metrics = station.get('quality_metrics') or {}
        residuals_xyz = metrics.get('residuals_xyz')
        if residuals_xyz is not None:
            residuals_arr = np.asarray(residuals_xyz, dtype=float)
            if residuals_arr.size:
                all_residuals.extend(residuals_arr.ravel())
    if not all_residuals:
        all_residuals = list(np.asarray(qm.get('observation_residuals', np.array([]))).ravel())

    return {
        'stations': registered_stations,
        'method': method_class.__name__,
        'baarda_enabled': bool(use_baarda_data_snooping),
        'design_matrix': qm.get('design_matrix', np.array([])),
        'normal_matrix': qm.get('normal_matrix', np.array([])),
        'covariance': qm.get('covariance', np.array([])),
        'observation_residuals': np.asarray(all_residuals, dtype=float),
        'observation_vector': qm.get('observation_vector', np.array([])),
        'correlation_matrix': qm.get('correlation_matrix', np.array([]))
    }


def export_detailed_report(result, method_name):
    """Generate a detailed TXT report with all quality information."""

    report_file = os.path.join(
        OUTPUT_FOLDER, f'detailed_report_{method_name}.txt')

    with open(report_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write(f"DETAILED REGISTRATION REPORT\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        # Summary Statistics
        f.write("1. OVERALL STATISTICS\n")
        f.write("-"*40 + "\n")

        total_stations = len(result['stations'])
        successful = sum(
            1 for s in result['stations'][1:] if s['coords'] is not None)
        failed = total_stations - 1 - successful

        f.write(f"Total stations processed: {total_stations}\n")
        f.write(f"Reference station: {result['stations'][0]['name']}\n")

        # Show processing order
        f.write(
            f"Processing order: {' -> '.join([s['name'] for s in result['stations']])}\n")

        f.write(f"Successfully registered: {successful}\n")
        f.write(f"Failed registrations: {failed}\n")

        if total_stations > 1:
            f.write(
                f"Success rate: {successful/(total_stations-1)*100:.1f}%\n\n")
        else:
            f.write(f"Success rate: N/A\n\n")

        # Error Statistics
        valid_errors = [s['rmse'] for s in result['stations'][1:]
                        if s['rmse'] is not None and not np.isnan(s['rmse'])]
        if valid_errors:
            f.write("2. REGISTRATION ERROR STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Average RMSE: {np.mean(valid_errors):.6f} m\n")
            f.write(f"Std Deviation: {np.std(valid_errors):.6f} m\n")
            f.write(f"Minimum RMSE: {np.min(valid_errors):.6f} m\n")
            f.write(f"Maximum RMSE: {np.max(valid_errors):.6f} m\n")
            f.write(
                f"RMSE Range: {np.max(valid_errors)-np.min(valid_errors):.6f} m\n\n")

        # Time Statistics
        valid_times = [s['time'] for s in result['stations'][1:]
                       if s['time'] is not None and not np.isnan(s['time'])]
        if valid_times:
            f.write("3. COMPUTATION TIME STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total processing time: {np.sum(valid_times):.4f} s\n")
            f.write(f"Average per station: {np.mean(valid_times):.4f} s\n")
            f.write(f"Min time: {np.min(valid_times):.4f} s\n")
            f.write(f"Max time: {np.max(valid_times):.4f} s\n\n")

        # Per-Station Detailed Results
        f.write("4. PER-STATION DETAILED RESULTS\n")
        f.write("-"*40 + "\n")

        for station in result['stations']:
            f.write(
                f"\nStation: {station['name']} (Number: {station.get('number', 'N/A')})\n")
            if station.get('is_reference', False):
                f.write(f"  Status: REFERENCE STATION (fixed - E1)\n")
                f.write(
                    f"  Control points: {len(station.get('ctrl_ids', []))}\n")
                f.write(f"  Total points: {len(station.get('all_ids', []))}\n")
            else:
                if station.get('coords') is not None:
                    f.write(f"  Status: SUCCESS\n")
                    f.write(
                        f"  Registration order: {station.get('registration_order', 'N/A')}\n")
                    f.write(
                        f"  Reference used: {station.get('reference_used', 'N/A')}\n")

                    if station.get('rmse') is not None and not np.isnan(station['rmse']):
                        f.write(f"  RMSE: {station['rmse']:.6f} m\n")
                    if station.get('time') is not None and not np.isnan(station['time']):
                        f.write(
                            f"  Computation time: {station['time']:.4f} s\n")

                    qm = station.get('quality_metrics')
                    if qm:
                        f.write(f"  Quality Metrics:\n")
                        f.write(
                            f"    - Control points used: {qm.get('n_control_points', 'N/A')}\n")
                        f.write(
                            f"    - Mean residual: {qm.get('mean_residual', 'N/A'):.6f} m\n")
                        f.write(
                            f"    - Std residual: {qm.get('std_residual', 'N/A'):.6f} m\n")
                        f.write(
                            f"    - Max residual: {qm.get('max_residual', 'N/A'):.6f} m\n")
                        f.write(
                            f"    - Scale factor: {qm.get('scale', 'N/A'):.6f}\n")

                        if 'sigma0' in qm:
                            f.write(
                                f"    - Sigma0 (reference variance): {qm['sigma0']:.6e}\n")
                        if 'sigma0_sq' in qm:
                            f.write(f"    - Sigma0²: {qm['sigma0_sq']:.6e}\n")

                        if 'translation' in qm:
                            trans = qm['translation']
                            f.write(
                                f"    - Translation (ΔE, ΔN, ΔH): ({trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}) m\n")

                        if 'rotation_matrix' in qm:
                            R = qm['rotation_matrix']
                            f.write(f"    - Rotation Matrix:\n")
                            f.write(
                                f"      [{R[0,0]:.6f}, {R[0,1]:.6f}, {R[0,2]:.6f}]\n")
                            f.write(
                                f"      [{R[1,0]:.6f}, {R[1,1]:.6f}, {R[1,2]:.6f}]\n")
                            f.write(
                                f"      [{R[2,0]:.6f}, {R[2,1]:.6f}, {R[2,2]:.6f}]\n")

                    f.write(
                        f"  Control points: {len(station.get('ctrl_ids', []))}\n")
                    f.write(
                        f"  Total points: {len(station.get('all_ids', []))}\n")
                else:
                    f.write(f"  Status: FAILED\n")
                    f.write(
                        f"  Error: {station.get('error', 'Unknown error')}\n")

        # Quality Assessment
        f.write("\n5. QUALITY ASSESSMENT\n")
        f.write("-"*40 + "\n")

        quality_scores = []
        for station in result['stations'][1:]:
            qm = station.get('quality_metrics')
            if qm and 'rmse' in qm:
                if qm['rmse'] < 0.01:
                    quality = "EXCELLENT"
                    score = 5
                elif qm['rmse'] < 0.05:
                    quality = "GOOD"
                    score = 4
                elif qm['rmse'] < 0.10:
                    quality = "FAIR"
                    score = 3
                elif qm['rmse'] < 0.20:
                    quality = "POOR"
                    score = 2
                else:
                    quality = "VERY POOR"
                    score = 1

                quality_scores.append(score)
                f.write(
                    f"  Station {station['name']}: {quality} (RMSE={qm['rmse']:.4f}m)\n")

        if quality_scores:
            avg_quality = np.mean(quality_scores)
            f.write(f"\n  Overall Quality Score (1-5): {avg_quality:.2f}\n")
            if avg_quality >= 4.5:
                f.write(f"  Overall Assessment: EXCELLENT\n")
            elif avg_quality >= 3.5:
                f.write(f"  Overall Assessment: GOOD\n")
            elif avg_quality >= 2.5:
                f.write(f"  Overall Assessment: FAIR\n")
            elif avg_quality >= 1.5:
                f.write(f"  Overall Assessment: POOR\n")
            else:
                f.write(f"  Overall Assessment: VERY POOR\n")

        # Recommendations
        f.write("\n6. RECOMMENDATIONS\n")
        f.write("-"*40 + "\n")

        if failed > 0:
            f.write(f"  WARNING: {failed} station(s) failed to register.\n")
            f.write(
                f"  Consider checking data quality or reducing MIN_PAIRED_CPS threshold.\n")

        if valid_errors and np.mean(valid_errors) > 0.10:
            f.write(
                f"  WARNING: High average registration error ({np.mean(valid_errors):.4f}m).\n")
            f.write(
                f"  Consider using a different registration method or reviewing control points.\n")

        # Show best registrations
        best_stations = []
        for station in result['stations'][1:]:
            qm = station.get('quality_metrics')
            if qm and 'rmse' in qm:
                best_stations.append((qm['rmse'], station['name']))

        if best_stations:
            best_stations.sort()
            f.write(f"\n  Best registrations (lowest RMSE):\n")
            for rmse, name in best_stations[:3]:
                f.write(f"    - {name}: RMSE={rmse:.6f}m\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"  Detailed report saved to: {report_file}")
    return report_file


def export_registered_points(result, method_name):
    """Export all registered points to a CSV file with station origin information."""
    all_points = []

    for station in result['stations']:
        if station.get('coords') is not None:
            coords = station['coords']
            all_ids = station.get('all_ids', [])

            for i, point_id in enumerate(all_ids):
                if i < len(coords):
                    point_data = {
                        'point_id': point_id,
                        'station_origin': station['name'],
                        'station_number': station.get('number', 0),
                        'east': coords[i, 0],
                        'north': coords[i, 1],
                        'height': coords[i, 2],
                        'is_control_point': point_id.startswith('CP'),
                        'registration_order': station.get('registration_order', -1),
                    }

                    transform = station.get('transform')
                    if transform:
                        point_data['transform_scale'] = transform[0]
                        point_data['transform_trans_e'] = transform[2][0]
                        point_data['transform_trans_n'] = transform[2][1]
                        point_data['transform_trans_h'] = transform[2][2]

                    original_coords = station.get('coords_original')
                    if original_coords is not None and i < len(original_coords):
                        point_data['east_original'] = original_coords[i, 0]
                        point_data['north_original'] = original_coords[i, 1]
                        point_data['height_original'] = original_coords[i, 2]

                    qm = station.get('quality_metrics')
                    if qm:
                        point_data['registration_rmse'] = qm.get(
                            'rmse', np.nan)
                        point_data['scale_factor'] = qm.get('scale', np.nan)
                        point_data['mean_residual'] = qm.get(
                            'mean_residual', np.nan)

                    all_points.append(point_data)

    if all_points:
        df = pd.DataFrame(all_points)
        df = df.sort_values(['station_number', 'point_id'])

        output_file = os.path.join(
            OUTPUT_FOLDER, f'registered_points_{method_name}.csv')
        df.to_csv(output_file, index=False)
        print(f"\n  Exported {len(df)} points to: {output_file}")
        return df
    else:
        print(f"\n  Warning: No points to export for {method_name}")
        return pd.DataFrame()


def compare_methods_sequential(stations, method_classes):
    """Run sequential registration for multiple methods and compare."""
    results = {}
    exported_data = {}
    reports = {}

    for method_class in method_classes:
        print(f"\n{'#'*70}")
        print(
            f"# Running sequential registration with {method_class.__name__}")
        print(f"{'#'*70}")

        result = sequential_registration(stations, method_class)
        results[method_class.__name__] = result

        # Export registered points
        df = export_registered_points(result, method_class.__name__)
        exported_data[method_class.__name__] = df

        # Generate detailed report
        report = export_detailed_report(result, method_class.__name__)
        reports[method_class.__name__] = report
        export_baarda_outputs(result, OUTPUT_FOLDER, method_class.__name__)

    return results, exported_data, reports


# ===========================================================================
# SECTION 3 - VISUALIZATION AND REPORTING
# ===========================================================================

def plot_registration_results(results, stations):
    """Plot registration errors and point clouds."""
    if plt is None:
        logger.warning("matplotlib is not available; registration plots were not generated.")
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Error bar plot
    ax1 = axes[0, 0]
    methods = list(results.keys())
    n_methods = len(methods)

    # Get max number of stations with valid data
    max_stations = 0
    for method in methods:
        valid_stations = [s for s in results[method]
                          ['stations'][1:] if s.get('coords') is not None]
        max_stations = max(max_stations, len(valid_stations))

    if max_stations == 0:
        max_stations = 1

    x = np.arange(max_stations)
    width = 0.8 / max(n_methods, 1)

    for i, (method, result) in enumerate(results.items()):
        valid_stations = [s for s in result['stations']
                          [1:] if s.get('coords') is not None]
        errors = [s.get('rmse', np.nan)
                  for s in valid_stations if s.get('rmse') is not None]

        if errors:
            offset = (i - n_methods/2 + 0.5) * width
            bars = ax1.bar(x[:len(errors)] + offset, errors, width,
                           label=method, alpha=0.7)

            for bar, err in zip(bars, errors):
                if not np.isnan(err):
                    ax1.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.001,
                             f'{err:.4f}', ha='center', va='bottom', fontsize=7)

    ax1.set_xlabel('Station Sequence (E2, E3, E4, ...)')
    ax1.set_ylabel('Registration RMSE (m)')
    ax1.set_title('Sequential Registration Errors by Method')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Cumulative error
    ax2 = axes[0, 1]
    for method, result in results.items():
        valid_stations = [s for s in result['stations']
                          [1:] if s.get('coords') is not None]
        errors = [s.get('rmse', np.nan)
                  for s in valid_stations if s.get('rmse') is not None]
        if errors:
            cumulative = np.cumsum(errors)
            ax2.plot(range(1, len(cumulative)+1), cumulative, 'o-',
                     label=method, linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Registered Stations')
    ax2.set_ylabel('Cumulative RMSE (m)')
    ax2.set_title('Cumulative Registration Error')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Average error comparison
    ax3 = axes[0, 2]
    method_names = list(results.keys())
    avg_errors = []
    for method in method_names:
        valid_stations = [s for s in results[method]
                          ['stations'][1:] if s.get('coords') is not None]
        errors = [s.get('rmse', np.nan)
                  for s in valid_stations if s.get('rmse') is not None]
        if errors:
            avg_errors.append(np.nanmean(errors))
        else:
            avg_errors.append(0)

    success_rates = []
    for method in method_names:
        total = len(results[method]['stations']) - 1
        if total > 0:
            successful = sum(
                1 for s in results[method]['stations'][1:] if s.get('coords') is not None)
            success_rates.append(successful / total * 100)
        else:
            success_rates.append(0)

    x_pos = np.arange(len(method_names))
    bars = ax3.bar(x_pos, avg_errors, color='skyblue', alpha=0.7)
    for bar, err in zip(bars, avg_errors):
        if err > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{err:.5f}', ha='center', va='bottom', fontsize=8)

    ax3.set_ylabel('Average RMSE (m)')
    ax3.set_title('Method Comparison - Average Error')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # Success rate comparison
    ax4 = axes[1, 0]
    bars2 = ax4.bar(x_pos, success_rates, color='lightgreen', alpha=0.7)
    for bar, rate in zip(bars2, success_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)

    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Registration Success Rate by Method')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3, axis='y')

    # Time comparison
    ax5 = axes[1, 1]
    for method, result in results.items():
        valid_stations = [s for s in result['stations']
                          [1:] if s.get('coords') is not None]
        times = [s.get('time', np.nan) for s in valid_stations if s.get(
            'time') is not None and not np.isnan(s['time'])]
        if times:
            ax5.plot(range(1, len(times)+1), times, 'o-',
                     label=method, linewidth=2, markersize=6)
    ax5.set_xlabel('Station Number')
    ax5.set_ylabel('Computation Time (s)')
    ax5.set_title('Processing Time per Station')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Quality score comparison
    ax6 = axes[1, 2]
    quality_scores = []
    for method in method_names:
        scores = []
        for station in results[method]['stations'][1:]:
            qm = station.get('quality_metrics')
            if qm and 'rmse' in qm:
                if qm['rmse'] < 0.01:
                    scores.append(5)
                elif qm['rmse'] < 0.05:
                    scores.append(4)
                elif qm['rmse'] < 0.10:
                    scores.append(3)
                elif qm['rmse'] < 0.20:
                    scores.append(2)
                else:
                    scores.append(1)
        if scores:
            quality_scores.append(np.mean(scores))
        else:
            quality_scores.append(0)

    bars3 = ax6.bar(x_pos, quality_scores, color='coral', alpha=0.7)
    for bar, score in zip(bars3, quality_scores):
        if score > 0:
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{score:.2f}', ha='center', va='bottom', fontsize=8)

    ax6.set_ylabel('Quality Score (1-5)')
    ax6.set_title('Overall Quality Assessment')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    ax6.set_ylim(0, 5.5)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def print_sequential_summary(results, stations):
    """Print detailed summary of sequential registration."""
    print("\n" + "="*80)
    print("SEQUENTIAL REGISTRATION SUMMARY")
    print("="*80)

    for method, result in results.items():
        print(f"\n{'─'*80}")
        print(f"Method: {method}")
        print(f"{'─'*80}")

        print(
            f"\n{'Station':<12} {'Number':<8} {'Status':<25} {'RMSE (m)':<12} {'Time (s)':<12} {'Quality':<10}")
        print("-" * 85)

        for station in result['stations']:
            station_name = station['name']
            station_num = station.get('number', 0)

            if station.get('is_reference', False):
                print(
                    f"{station_name:<12} {station_num:<8} {'REFERENCE (E1)':<25} {'-':<12} {'-':<12} {'-':<10}")
            else:
                if station.get('coords') is not None:
                    rmse = f"{station['rmse']:.6f}" if station.get(
                        'rmse') and not np.isnan(station['rmse']) else '-'
                    time_val = f"{station['time']:.4f}" if station.get(
                        'time') and not np.isnan(station['time']) else '-'

                    # Quality assessment
                    qm = station.get('quality_metrics')
                    if qm and 'rmse' in qm:
                        if qm['rmse'] < 0.01:
                            quality = "EXCELLENT"
                        elif qm['rmse'] < 0.05:
                            quality = "GOOD"
                        elif qm['rmse'] < 0.10:
                            quality = "FAIR"
                        else:
                            quality = "POOR"
                    else:
                        quality = "N/A"

                    print(
                        f"{station_name:<12} {station_num:<8} {'SUCCESS':<25} {rmse:<12} {time_val:<12} {quality:<10}")
                else:
                    print(
                        f"{station_name:<12} {station_num:<8} {'FAILED':<25} {'-':<12} {'-':<12} {'-':<10}")

        # Statistics
        valid_stations = [s for s in result['stations']
                          [1:] if s.get('coords') is not None]
        valid_errors = [s['rmse'] for s in valid_stations if s.get(
            'rmse') and not np.isnan(s['rmse'])]
        valid_times = [s['time'] for s in valid_stations if s.get(
            'time') and not np.isnan(s['time'])]

        if valid_errors:
            print(f"\n📊 Statistics for {method}:")
            print(
                f"  ✅ Successful registrations: {len(valid_errors)}/{len(result['stations'])-1}")
            print(f"  📈 Average RMSE: {np.mean(valid_errors):.6f} m")
            print(f"  📉 Std RMSE:     {np.std(valid_errors):.6f} m")
            print(f"  🔼 Max RMSE:     {np.max(valid_errors):.6f} m")
            print(f"  🔽 Min RMSE:     {np.min(valid_errors):.6f} m")
            print(f"  ⏱️  Total time:   {np.sum(valid_times):.4f} s")
            print(f"  ⏱️  Avg time:     {np.mean(valid_times):.4f} s")

    # Best method
    print(f"\n{'='*80}")
    print("🏆 BEST METHOD COMPARISON")
    print(f"{'='*80}")

    best_method = None
    best_avg_error = float('inf')

    for method, result in results.items():
        valid_errors = [s['rmse'] for s in result['stations'][1:]
                        if s.get('rmse') and not np.isnan(s['rmse'])]
        if valid_errors:
            avg_error = np.mean(valid_errors)
            if avg_error < best_avg_error:
                best_avg_error = avg_error
                best_method = method

    if best_method:
        print(f"\n✨ Best method: {best_method}")
        print(f"📊 Average registration error: {best_avg_error:.6f} m")

    # Method ranking
    print(f"\n📋 Method Ranking (by average error):")
    method_ranking = []
    for method, result in results.items():
        valid_errors = [s['rmse'] for s in result['stations'][1:]
                        if s.get('rmse') and not np.isnan(s['rmse'])]
        if valid_errors:
            avg_err = np.mean(valid_errors)
            success_rate = sum(1 for s in result['stations'][1:] if s.get(
                'coords') is not None) / (len(result['stations'])-1) * 100
            method_ranking.append((avg_err, method, success_rate))

    method_ranking.sort()
    for i, (avg_err, method, success_rate) in enumerate(method_ranking, 1):
        print(
            f"  {i}. {method}: {avg_err:.6f} m (Success rate: {success_rate:.1f}%)")


# ===========================================================================
# SECTION 4 - MAIN
# ===========================================================================

def sequential_main():
    # Load data (already sorted by station number)
    print("📂 Loading station data...")
    stations = load_station_data(DATA_FOLDER)
    print(f"✅ Loaded {len(stations)} stations in sequential order:")
    for st in stations:
        print(
            f"  📍 {st['name']} (Number: {st['number']}) - {len(st['ctrl_ids'])} control points")

    # Define methods to test
    methods_to_test = [
        Method1_SequentialSVD,
        Schut1973_SpaceTime,
        Stepien2020_QST,
        Tao2020_CMTLS_Rigorous,
        Method4_GlobalEulerLM,
    ]

    # Run sequential registration for all methods
    print("\n" + "="*80)
    print("🚀 STARTING SEQUENTIAL REGISTRATION EXPERIMENTS")
    print("📋 Processing order: E1 -> E2 -> E3 -> E4 -> ...")
    print("="*80)

    results, exported_data, reports = compare_methods_sequential(
        stations, methods_to_test)

    # Print summary
    print_sequential_summary(results, stations)

    # Plot results
    print("\n📊 Generating plots...")
    plot_registration_results(results, stations)

    # Create a combined output file with all methods
    print("\n" + "="*80)
    print("📁 CREATING COMBINED OUTPUT FILES")
    print("="*80)

    all_methods_data = []
    for method_name, df in exported_data.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['method'] = method_name
            all_methods_data.append(df_copy)

    if all_methods_data:
        combined_df = pd.concat(all_methods_data, ignore_index=True)
        combined_file = os.path.join(
            OUTPUT_FOLDER, 'all_methods_registered_points.csv')
        combined_df.to_csv(combined_file, index=False)
        print(f"\n✅ Combined output file created: {combined_file}")
        print(f"   Total records across all methods: {len(combined_df)}")

    # Create a summary JSON file
    summary_data = {}
    for method_name, result in results.items():
        valid_errors = [s['rmse'] for s in result['stations'][1:]
                        if s.get('rmse') and not np.isnan(s['rmse'])]
        summary_data[method_name] = {
            'average_rmse': np.mean(valid_errors) if valid_errors else np.nan,
            'std_rmse': np.std(valid_errors) if valid_errors else np.nan,
            'success_rate': sum(1 for s in result['stations'][1:] if s.get('coords') is not None) / (len(result['stations'])-1) * 100 if len(result['stations']) > 1 else 0,
            'total_time': np.sum([s['time'] for s in result['stations'][1:] if s.get('time') and not np.isnan(s['time'])]),
            'report_file': reports[method_name]
        }

    summary_file = os.path.join(OUTPUT_FOLDER, 'registration_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Summary JSON created: {summary_file}")

    print("\n" + "="*80)
    print("✅ SEQUENTIAL REGISTRATION COMPLETE!")
    print("="*80)
    print(f"\n📁 Output files saved to: {OUTPUT_FOLDER}")
    print(f"   - CSV files with registered points for each method")
    print(f"   - TXT reports with detailed quality information")
    print(f"   - Combined output file with all methods")
    print(f"   - JSON summary file")


# ===========================================================================
# NETWORK REGISTRATION WORKFLOW
# ===========================================================================
"""
NETWORK REGISTRATION - Ajustamento em Rede
Ajusta todas as estações simultaneamente usando pontos que aparecem em 2+ estações
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import json
from datetime import datetime
from collections import defaultdict
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from freesurvey.data_manager import DataManager
from freesurvey.Registration_Module import (
    Method3_GlobalQuaternionsLM_ObsEquations,
    Method4_GlobalEulerLM,
    Tao2020_CMTLS_Rigorous,
    evaluate_rmse,
)
from freesurvey.methods import N_CUQ_LM

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def is_global_lm_method(method_class):
    return getattr(method_class, "is_global_lm", False) or "Global" in method_class.__name__


def is_cmtls_method(method_class):
    return getattr(method_class, "method_code", "") == "CMTLS" or method_class.__name__ in {
        "Tao2020_CMTLS_Rigorous",
        "CMTLS",
    }

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NETWORK_OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_network")

MIN_PAIRED_CPS = 4
MIN_STATIONS_PER_CP = 2
os.makedirs(NETWORK_OUTPUT_FOLDER, exist_ok=True)


def find_cps_in_at_least_n_stations(stations, min_stations=2):
    from collections import Counter
    all_ctrl_ids = []
    for station in stations:
        all_ctrl_ids.extend(station['ctrl_ids'])
    counts = Counter(all_ctrl_ids)
    return sorted([cp for cp, count in counts.items() if count >= min_stations])


def build_connectivity(stations, cp_ids):
    connectivity = defaultdict(list)
    for cp in cp_ids:
        for station in stations:
            if cp in station['ctrl_ids']:
                connectivity[cp].append(station['name'])
    return connectivity


# ===========================================================================
# SECTION 2 - NETWORK ADJUSTMENT
# ===========================================================================


def network_adjustment_lm(stations, network_cps, connectivity, method_class, min_paired_cps=4,
                          use_baarda_data_snooping=False,
                          baarda_threshold=3.0,
                          baarda_max_iterations=10,
                          baarda_min_observations_per_point=4,
                          baarda_remove_strategy="none",
                          baarda_use_posterior_sigma=True,
                          max_iter=50, tol=1e-8, weight_constraint=1e5, linear_solver='legacy_normal', initial_network_result=None):
    print(f"\n{'-'*60}")
    print("[FASE 1] INICIALIZAÇÃO DE DATUM GLOBAL")
    print("Criando malha base para evitar distorção de coordenadas locais...")
    print(f"{'-'*60}")
    
    if initial_network_result is None:
        # Usa o método sequencial rigoroso para costurar a rede e obter coordenadas globais reais
        seq_result = network_adjustment_sequential(
            stations,
            network_cps,
            connectivity,
            Tao2020_CMTLS_Rigorous,
            min_paired_cps,
            use_baarda_data_snooping=False,
            baarda_threshold=baarda_threshold,
            baarda_max_iterations=baarda_max_iterations,
            baarda_min_observations_per_point=baarda_min_observations_per_point,
            baarda_remove_strategy="none",
            baarda_use_posterior_sigma=baarda_use_posterior_sigma,
        )
    
        if not seq_result or not seq_result.get('success', True) and len(seq_result['control_points']['ids']) == 0:
            print("  [FAIL] Falha ao construir a malha base.")
            return None
        
    else:
        # Previous accepted solution supplies approximations only. All retained
        # observations are adjusted again; the datum parameters stay identical.
        seq_result = {'control_points': initial_network_result['control_points']}

    # Cria um dicionário com os pontos de controle nas suas verdadeiras posições no Datum
    global_target_pts = dict(zip(seq_result['control_points']['ids'], seq_result['control_points']['target_coords']))
    
    print(f"\n{'-'*60}")
    print("[FASE 2] OTIMIZAÇÃO SIMULTÂNEA (REDE)")
    print(f"Refinando a rede inteira em bloco com {method_class.__name__}")
    print(f"{'-'*60}")
    
    source_matrices = []
    target_matrices = []
    station_names = []
    station_numbers = []
    station_coords = []
    station_all_ids = []
    station_ctrl_ids = []
    station_fit_ids = []
    
# Agora construímos os alvos corretos referenciando a malha global que acabamos de criar
    for station in stations:
        common = [cp for cp in network_cps if cp in station['ctrl_ids'] and cp in global_target_pts]
        if len(common) >= min_paired_cps:
            src_pts = get_coords_for_cps(station, common)
            tgt_pts = np.array([global_target_pts[cp] for cp in common])
            
            valid = ~(np.isnan(src_pts).any(axis=1) | np.isnan(tgt_pts).any(axis=1))
            if np.sum(valid) >= min_paired_cps:
                source_matrices.append(src_pts[valid])
                target_matrices.append(tgt_pts[valid])
                station_names.append(station['name'])
                station_numbers.append(station['number'])
                station_coords.append(station['coords'])
                station_all_ids.append(station['all_ids'])
                station_ctrl_ids.append(station['ctrl_ids'])
                station_fit_ids.append([cp for cp, is_valid in zip(common, valid) if is_valid])
    
    print(f"Processando {len(source_matrices)} estações para refinamento global...")
    
    start_time = datetime.now()
    try:
        solve_kwargs = {}
        try:
            import inspect
            signature = inspect.signature(method_class.solve)
            if "iterative_outlier_removal" in signature.parameters:
                solve_kwargs["iterative_outlier_removal"] = False
            if "outlier_threshold" in signature.parameters:
                solve_kwargs["outlier_threshold"] = baarda_threshold
            if "outlier_min_points" in signature.parameters:
                solve_kwargs["outlier_min_points"] = baarda_min_observations_per_point
            if "max_iter" in signature.parameters:
                solve_kwargs["max_iter"] = max_iter
            if "tol" in signature.parameters:
                solve_kwargs["tol"] = tol
            if "weight_constraint" in signature.parameters:
                solve_kwargs["weight_constraint"] = weight_constraint
            if "linear_solver" in signature.parameters:
                solve_kwargs["linear_solver"] = linear_solver
            if "point_ids" in signature.parameters:
                solve_kwargs["point_ids"] = station_fit_ids
            if "station_names" in signature.parameters:
                solve_kwargs["station_names"] = station_names
            if "initial_point_coordinates" in signature.parameters:
                solve_kwargs["initial_point_coordinates"] = global_target_pts
            if "datum_station_index" in signature.parameters:
                solve_kwargs["datum_station_index"] = 0
        except Exception:
            solve_kwargs = {}
        if initial_network_result is not None:
            previous = {str(v['station']): v for v in initial_network_result['stations']}
            if set(station_names) != set(previous):
                raise ValueError('Warm-start trial must retain all stations')
            params = []
            for name in station_names:
                v = previous[name]
                q = method_class._matrix_to_quaternion(v['rotation'])
                params.append(np.r_[v['scale'], q, v['translation']])
            solve_kwargs['initial_station_parameters'] = np.asarray(params)
        solve_res = method_class.solve(source_matrices, target_matrices, **solve_kwargs)
        if len(solve_res) == 5:
            results, sigma0_sq, cov, residuals, elapsed = solve_res
            matrices = {}
        elif len(solve_res) == 6:
            results, sigma0_sq, cov, residuals, elapsed, matrices = solve_res
        else:
            results, sigma0_sq, cov, residuals, elapsed = solve_res[:5]
            matrices = {}
        shared_point_ids = [
            str(point_id) for point_id in matrices.get("shared_point_ids", [])
        ] if isinstance(matrices, dict) else []
        shared_point_coordinates = np.asarray(
            matrices.get("shared_point_coordinates", []),
            dtype=float,
        ) if isinstance(matrices, dict) else np.empty((0, 3))
        adjusted_shared_points = {
            point_id: shared_point_coordinates[index]
            for index, point_id in enumerate(shared_point_ids)
            if index < len(shared_point_coordinates)
        }
        
        print(f"  [OK] Sucesso! Global Sigma0={np.sqrt(sigma0_sq):.6f}m, Tempo={elapsed:.4f}s")
        
        station_results = []
        transformed_stations = []
        network_residual_rows = []
        active_masks = matrices.get("active_observation_masks", []) if isinstance(matrices, dict) else []
        
        for i, (name, num, (scale, R, trans)) in enumerate(zip(station_names, station_numbers, results)):
            active_mask = None
            if i < len(active_masks):
                try:
                    active_mask = np.asarray(active_masks[i], dtype=bool)
                except Exception:
                    active_mask = None
            if active_mask is None or active_mask.shape[0] != len(source_matrices[i]):
                active_mask = np.ones(len(source_matrices[i]), dtype=bool)
            fit_ids = [cp for cp, keep in zip(station_fit_ids[i], active_mask) if keep]
            fit_source = source_matrices[i][active_mask]
            if adjusted_shared_points:
                fit_target = np.asarray(
                    [
                        adjusted_shared_points.get(str(point_id), target)
                        for point_id, target in zip(
                            fit_ids,
                            target_matrices[i][active_mask],
                        )
                    ],
                    dtype=float,
                )
            else:
                fit_target = target_matrices[i][active_mask]
            removed_fit_ids = {str(cp) for cp, keep in zip(station_fit_ids[i], active_mask) if not keep}
            full_keep = np.asarray([
                str(point_id) not in removed_fit_ids
                for point_id in station_all_ids[i]
            ], dtype=bool)
            transformed_full = (scale * station_coords[i] @ R.T) + trans
            transformed = transformed_full[full_keep]
            coords_original_kept = station_coords[i][full_keep]
            all_ids_kept = [
                point_id for point_id, keep in zip(station_all_ids[i], full_keep) if keep
            ]
            ctrl_ids_kept = [
                point_id for point_id in station_ctrl_ids[i] if str(point_id) not in removed_fit_ids
            ]
            rmse = evaluate_rmse(fit_source, fit_target, scale, R, trans)
            transformed_fit = scale * (fit_source @ R.T) + trans
            residual_fit = transformed_fit - fit_target
            residual_norms = np.linalg.norm(residual_fit, axis=1)
            for cp, observed, adjusted, residual, norm in zip(
                fit_ids, fit_target, transformed_fit, residual_fit, residual_norms
            ):
                network_residual_rows.append({
                    'station': name,
                    'point': cp,
                    'point_id': cp,
                    'observed_x': observed[0],
                    'observed_y': observed[1],
                    'observed_z': observed[2],
                    'adjusted_x': adjusted[0],
                    'adjusted_y': adjusted[1],
                    'adjusted_z': adjusted[2],
                    'vx': residual[0],
                    'vy': residual[1],
                    'vz': residual[2],
                    'v_norm': norm,
                    'is_control_point': True,
                })
            
            pt_uncs = {}
            if cov is not None and len(results) * 8 == cov.shape[0]:
                st_cov = cov[i*8:(i+1)*8, i*8:(i+1)*8]
                from freesurvey.Registration_Module import Rotation
                r = Rotation.from_matrix(R)
                quat_xyzw = r.as_quat()
                q0, q1, q2, q3 = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
                if q0 < 0: q0, q1, q2, q3 = -q0, -q1, -q2, -q3
                p = np.array([scale, q0, q1, q2, q3, trans[0], trans[1], trans[2]])
                from freesurvey.Registration_Module import propagate_point_covariance, Method3_GlobalQuaternionsLM_ObsEquations
                for j, pt_id in enumerate(station_all_ids[i]):
                    try:
                        point_covariance_matrix = propagate_point_covariance(station_coords[i][j], p, st_cov, Method3_GlobalQuaternionsLM_ObsEquations._quat_to_matrix)
                        pt_uncs[pt_id] = point_covariance_matrix
                    except Exception:
                        pt_uncs[pt_id] = np.full((3, 3), np.nan)

            station_quality_metrics = {
                'rmse': rmse,
                'mean_residual': float(np.mean(residual_norms)) if residual_norms.size else np.nan,
                'std_residual': float(np.std(residual_norms)) if residual_norms.size else np.nan,
                'max_residual': float(np.max(residual_norms)) if residual_norms.size else np.nan,
                'n_control_points': len(fit_source),
                'scale': scale,
                'translation': trans,
                'rotation_matrix': R,
                'rotation_determinant': float(np.linalg.det(R)),
                'orthogonality_error': float(np.linalg.norm(R.T @ R - np.eye(3))),
            }

            station_results.append({
                'station': name,
                'number': num,
                'scale': scale,
                'rotation': R,
                'translation': trans,
                'rmse': rmse,
                'n_points': len(fit_source),
                'success': True,
                'rotation_determinant': station_quality_metrics['rotation_determinant'],
                'orthogonality_error': station_quality_metrics['orthogonality_error'],
                'quality_metrics': station_quality_metrics,
            })
            
            transformed_stations.append({
                'name': name,
                'number': num,
                'coords': transformed,
                'coords_original': coords_original_kept,
                'all_ids': all_ids_kept,
                'ctrl_ids': ctrl_ids_kept,
                'transform': (scale, R, trans),
                'rmse': rmse,
                'excluded_outliers': sorted(removed_fit_ids),
                'point_uncertainties': pt_uncs,
                'quality_metrics': station_quality_metrics,
            })
        
        rmses = [float(station['rmse']) for station in station_results if station.get('rmse') is not None]
        covariance_param_names = list(
            matrices.get("covariance_param_names", [])
        ) if isinstance(matrices, dict) else []
        if (
            not covariance_param_names
            and cov is not None
            and getattr(cov, "ndim", 0) == 2
            and len(station_results)
        ):
            block_size = cov.shape[0] // len(station_results) if cov.shape[0] % len(station_results) == 0 else 0
            if block_size == 8:
                base_names = ["scale", "q0", "q1", "q2", "q3", "tx", "ty", "tz"]
            elif block_size == 7:
                base_names = ["scale", "rx", "ry", "rz", "tx", "ty", "tz"]
            else:
                base_names = [f"p{i + 1}" for i in range(block_size)]
            for station in station_results:
                station_name = station.get('station', station.get('name', 'station'))
                covariance_param_names.extend([f"{station_name}.{name}" for name in base_names])

        iterative_trace = (
            list(matrices.get("iterative_outlier_trace", []))
            if isinstance(matrices, dict)
            else []
        )
        removed_points = sorted({
            str(entry.get("point_id"))
            for entry in iterative_trace
            if entry.get("point_id") is not None
        })
        baarda_diagnostics = {
            "enabled": bool(use_baarda_data_snooping),
            "threshold": baarda_threshold,
            "iterations": iterative_trace,
            "removed_observations": iterative_trace,
            "removed_points": removed_points,
            "final_max_w": np.nan,
            "final_sigma0_hat": float(np.sqrt(sigma0_sq)),
            "converged": matrices.get('converged'),
            "message": (
                "Outlier screening is handled by the analyzer-level geometry-preserving rerun loop."
                if use_baarda_data_snooping
                else "Iterative outlier screening was disabled for this run."
            ),
        }

        control_point_ids = shared_point_ids or list(seq_result['control_points']['ids'])
        control_point_coords = (
            shared_point_coordinates
            if len(shared_point_coordinates)
            else np.asarray(seq_result['control_points']['target_coords'], dtype=float)
        )
        return {
            **{key: matrices.get(key) for key in ('iterations','iteration_history','lm_trial_history','lm_initial_lambda','lm_acceptance_ratio','lm_gradient_tolerance','solution_type','converged','stop_reason','max_iterations','tolerance','final_increment_norm','final_objective','damping_coefficient','linear_solver','solver_description','statistical_design_matrix','statistical_parameter_basis','quaternion_constraint_matrix','degrees_of_freedom','dof')},
            'type': 'network_adjustment',
            'method': method_class.__name__,
            'registration_type': 'network',
            'baarda_enabled': bool(use_baarda_data_snooping),
            'rmse': float(np.mean(rmses)) if rmses else np.nan,
            'mean_rmse': float(np.mean(rmses)) if rmses else np.nan,
            'max_rmse': float(np.max(rmses)) if rmses else np.nan,
            'stations': station_results,
            'transformed_stations': transformed_stations,
            'control_points': {
                **seq_result['control_points'],
                'ids': control_point_ids,
                'target_coords': control_point_coords,
                'adjusted_in_coupled_network': bool(adjusted_shared_points),
            },
            'sigma0': np.sqrt(sigma0_sq),
            'sigma0_sq': sigma0_sq,
            'covariance': cov,
            'covariance_param_names': covariance_param_names,
            'design_matrix': matrices.get('design_matrix', np.array([])),
            'normal_matrix': matrices.get('normal_matrix', np.array([])),
            'observation_residuals': matrices.get('observation_residuals', residuals),
            'coordinate_metadata': matrices.get('coordinate_metadata', []),
            'point_metadata': matrices.get('point_metadata', []),
            'network_residual_rows': network_residual_rows,
            'observation_vector': matrices.get('observation_vector', np.array([])),
            'correlation_matrix': matrices.get('correlation_matrix', np.array([])),
            'coupled_design_matrix': matrices.get('coupled_design_matrix', np.array([])),
            'coupled_covariance': matrices.get('coupled_covariance', np.array([])),
            'coupled_correlation_matrix': matrices.get('coupled_correlation_matrix', np.array([])),
            'coupled_parameter_names': matrices.get('coupled_parameter_names', []),
            'coupled_covariance_param_names': matrices.get('coupled_parameter_names', []),
            'coupled_correlation_matrix_param_names': matrices.get('coupled_parameter_names', []),
            'network_observation_covariance_matrix': matrices.get(
                'network_observation_covariance_matrix',
                np.array([]),
            ),
            'network_observation_correlation_matrix': matrices.get(
                'network_observation_correlation_matrix',
                np.array([]),
            ),
            'network_observation_correlation_matrix_param_names': matrices.get(
                'network_observation_correlation_matrix_param_names',
                [],
            ),
            'stationwise_observation_covariance_matrix': matrices.get(
                'stationwise_observation_covariance_matrix',
                np.array([]),
            ),
            'stationwise_observation_correlation_matrix': matrices.get(
                'stationwise_observation_correlation_matrix',
                np.array([]),
            ),
            'stationwise_observation_correlation_matrix_param_names': matrices.get(
                'stationwise_observation_correlation_matrix_param_names',
                [],
            ),
            'observation_comparison_design_matrix': matrices.get(
                'observation_comparison_design_matrix',
                np.array([]),
            ),
            'shared_point_ids': shared_point_ids,
            'shared_point_coordinates': shared_point_coordinates,
            'datum_station_index': matrices.get('datum_station_index'),
            'datum_station_name': matrices.get('datum_station_name'),
            'fully_coupled_network': bool(matrices.get('fully_coupled_network', False)),
            'baarda_diagnostics': baarda_diagnostics,
            'time': elapsed,
            'n_stations': len(source_matrices),
            'n_control_points': len(global_target_pts)
        }
        
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def network_adjustment_sequential(stations, network_cps, connectivity, method_class, min_paired_cps=4,
                                  use_baarda_data_snooping=False,
                                  baarda_threshold=3.0,
                                  baarda_max_iterations=10,
                                  baarda_min_observations_per_point=4,
                                  baarda_remove_strategy="none",
                                  baarda_use_posterior_sigma=True):
    print(f"\nPerforming incremental/cumulative network adjustment with {method_class.__name__}")
    
    # Escolhe a estação de referencia inicial (aquela com mais CPs)
    ref_station = max(stations, key=lambda s: len(s['ctrl_ids']))
    print(f"Reference Center: {ref_station['name']} ({len(ref_station['ctrl_ids'])} CPs)")
    
    # Banco de coordenadas globais (Dynamic pool)
    global_target_pts = {}
    for cp in network_cps:
        if cp in ref_station['ctrl_ids']:
            try:
                idx = ref_station['all_ids'].index(cp)
                global_target_pts[cp] = ref_station['coords'][idx]
            except ValueError:
                pass
    
    print(f"Reference initial pool provides {len(global_target_pts)} points")
    
    res_dict = {}
    trans_dict = {}
    network_residual_rows = []
    baarda_report_rows = []
    baarda_final_rows = []
    
    # Salva a referencia de imediato
    reference_quality_metrics = {
        'rmse': 0.0,
        'mean_residual': 0.0,
        'std_residual': 0.0,
        'max_residual': 0.0,
        'n_control_points': len(global_target_pts),
        'scale': 1.0,
        'translation': np.zeros(3),
        'rotation_matrix': np.eye(3),
        'rotation_determinant': 1.0,
        'orthogonality_error': 0.0,
    }
    res_dict[ref_station['name']] = {
        'station': ref_station['name'],
        'number': ref_station['number'],
        'scale': 1.0,
        'rotation': np.eye(3),
        'translation': np.zeros(3),
        'rmse': 0.0,
        'n_points': len(global_target_pts),
        'is_reference': True,
        'success': True,
        'rotation_determinant': 1.0,
        'orthogonality_error': 0.0,
        'quality_metrics': reference_quality_metrics,
    }
    
    trans_dict[ref_station['name']] = {
        'name': ref_station['name'],
        'number': ref_station['number'],
        'coords': ref_station['coords'].copy(),
        'coords_original': ref_station['coords'].copy(),
        'all_ids': ref_station['all_ids'].copy(),
        'ctrl_ids': ref_station['ctrl_ids'].copy(),
        'transform': (1.0, np.eye(3), np.zeros(3)),
        'rmse': 0.0,
        'success': True,
        'quality_metrics': reference_quality_metrics,
    }
    
    total_time = 0
    pending_stations = [s for s in stations if s['name'] != ref_station['name']]
    
    # Loop de Expansao de Rede em Cadeia
    while True:
        added_any_in_this_round = False
        
        for station in list(pending_stations):
# Procura pontos comuns com QUALQUER estacao ja integrada no pool global
            common = [cp for cp in network_cps if cp in station['ctrl_ids'] and cp in global_target_pts]
            
            if len(common) < min_paired_cps:
                continue
                
            src = get_coords_for_cps(station, common)
            tgt = np.array([global_target_pts[cp] for cp in common])
            
            valid = ~(np.isnan(src).any(axis=1) | np.isnan(tgt).any(axis=1))
            src_v, tgt_v = src[valid], tgt[valid]
            common_v = [cp for cp, is_valid in zip(common, valid) if is_valid]
            
            if len(src_v) < min_paired_cps:
                continue
                
            try:
                start = datetime.now()
                if is_cmtls_method(method_class):
                    scale, R, trans, Q, stats, elapsed = method_class.solve(src_v, tgt_v)
                else:
                    scale, R, trans, elapsed = method_class.solve(src_v, tgt_v)
                
                rmse = evaluate_rmse(src_v, tgt_v, scale, R, trans)
                total_time += elapsed
                
                transformed = (scale * station['coords'] @ R.T) + trans
                transformed_ctrl = scale * (src_v @ R.T) + trans
                residuals = transformed_ctrl - tgt_v
                residual_norms = np.linalg.norm(residuals, axis=1)
                design_matrix = _numeric_design_matrix(src_v, scale, R, trans)
                baarda_diagnostics = _run_baarda_diagnostics(
                    design_matrix,
                    residuals,
                    common_v,
                    station.get("name", ""),
                    method_class.__name__,
                    "network_sequential",
                    use_baarda_data_snooping=use_baarda_data_snooping,
                    baarda_threshold=baarda_threshold,
                    baarda_max_iterations=baarda_max_iterations,
                    baarda_min_observations_per_point=baarda_min_observations_per_point,
                    baarda_remove_strategy=baarda_remove_strategy,
                    baarda_use_posterior_sigma=baarda_use_posterior_sigma,
                )
                baarda_report_rows.extend(baarda_diagnostics.get("outlier_report_rows", []))
                baarda_final_rows.extend(baarda_diagnostics.get("final_residual_rows", []))
                print(f"  [OK] {station['name']}: scale={scale:.6f}, RMSE={rmse:.6f}m (using {len(src_v)} CPs)")
                for cp, observed, adjusted, residual, norm in zip(
                    common_v, tgt_v, transformed_ctrl, residuals, residual_norms
                ):
                    network_residual_rows.append({
                        'station': station['name'],
                        'point': cp,
                        'point_id': cp,
                        'observed_x': observed[0],
                        'observed_y': observed[1],
                        'observed_z': observed[2],
                        'adjusted_x': adjusted[0],
                        'adjusted_y': adjusted[1],
                        'adjusted_z': adjusted[2],
                        'vx': residual[0],
                        'vy': residual[1],
                        'vz': residual[2],
                        'v_norm': norm,
                        'is_control_point': True,
                    })

                station_quality_metrics = {
                    'rmse': rmse,
                    'mean_residual': float(np.mean(residual_norms)) if residual_norms.size else np.nan,
                    'std_residual': float(np.std(residual_norms)) if residual_norms.size else np.nan,
                    'max_residual': float(np.max(residual_norms)) if residual_norms.size else np.nan,
                    'n_control_points': len(src_v),
                    'scale': scale,
                    'translation': trans,
                    'rotation_matrix': R,
                    'rotation_determinant': float(np.linalg.det(R)),
                    'orthogonality_error': float(np.linalg.norm(R.T @ R - np.eye(3))),
                }
                
                res_dict[station['name']] = {
                    'station': station['name'],
                    'number': station['number'],
                    'scale': scale,
                    'rotation': R,
                    'translation': trans,
                    'rmse': rmse,
                    'n_points': len(src_v),
                    'success': True,
                    'rotation_determinant': station_quality_metrics['rotation_determinant'],
                    'orthogonality_error': station_quality_metrics['orthogonality_error'],
                    'quality_metrics': station_quality_metrics,
                    'baarda_diagnostics': baarda_diagnostics,
                }
                
                trans_dict[station['name']] = {
                    'name': station['name'],
                    'number': station['number'],
                    'coords': transformed,
                    'coords_original': station['coords'].copy(),
                    'all_ids': station['all_ids'].copy(),
                    'ctrl_ids': station['ctrl_ids'].copy(),
                    'transform': (scale, R, trans),
                    'rmse': rmse,
                    'success': True,
                    'quality_metrics': station_quality_metrics,
                    'baarda_diagnostics': baarda_diagnostics,
                }
                
                # ADICIONA NOVOS PONTOS DESTA ESTACAO AO POOL GLOBAL PARA AS PROXIMAS!
                for cp in station['ctrl_ids']:
                    if cp not in global_target_pts:
                        idx = station['all_ids'].index(cp)
                        global_target_pts[cp] = transformed[idx]
                
                pending_stations.remove(station)
                added_any_in_this_round = True
                break # Reinicia o loop com o pool de coordenadas expandido
                
            except Exception as e:
                pass
                
        if not added_any_in_this_round:
            break # Nao ha mais estacoes vizinhas conectaveis
            
    # Trata as estacoes restantes falhas (sem amarra real alguma na rede)
    for station in pending_stations:
        common = [cp for cp in network_cps if cp in station['ctrl_ids'] and cp in global_target_pts]
        print(f"  [WARN] {station['name']}: only {len(common)} available CPs - skipped completely (Island).")
        res_dict[station['name']] = {
            'station': station['name'],
            'number': station['number'],
            'success': False,
            'error': f'Only {len(common)} CPs connected to active network.'
        }
        trans_dict[station['name']] = {
            'name': station['name'],
            'number': station['number'],
            'coords': None,
            'success': False
        }
        
    # Organiza na ordem original para evitar erros de plotagem
    station_results = [res_dict[s['name']] for s in stations]
    transformed_stations = [trans_dict[s['name']] for s in stations]
    
    final_cp_ids = list(global_target_pts.keys())
    final_cp_coords = np.array([global_target_pts[cp] for cp in final_cp_ids])
    rmses = [
        float(station['rmse']) for station in station_results
        if station.get('success', False) and station.get('rmse') is not None
    ]
    
    return {
        'type': 'sequential_network',
        'method': method_class.__name__,
        'registration_type': 'network',
        'rmse': float(np.mean(rmses)) if rmses else np.nan,
        'mean_rmse': float(np.mean(rmses)) if rmses else np.nan,
        'max_rmse': float(np.max(rmses)) if rmses else np.nan,
        'stations': station_results,
        'transformed_stations': transformed_stations,
        'control_points': {
            'ids': final_cp_ids,
            'target_coords': final_cp_coords,
            'connectivity': {cp: connectivity[cp] for cp in final_cp_ids if cp in connectivity}
        },
        'reference_station': ref_station['name'],
        'time': total_time,
        'n_stations': len([s for s in station_results if s.get('success', False)]),
        'n_control_points': len(final_cp_ids),
        'network_residual_rows': network_residual_rows,
        'baarda_enabled': bool(use_baarda_data_snooping),
        'baarda_diagnostics': {
            "enabled": bool(use_baarda_data_snooping),
            "threshold": baarda_threshold,
            "iterations": [],
            "removed_observations": [],
            "removed_points": [],
            "final_max_w": np.nan,
            "final_sigma0_hat": np.nan,
            "converged": True,
            "message": "Aggregated station-level Baarda diagnostics.",
            "outlier_report_rows": baarda_report_rows,
            "final_residual_rows": baarda_final_rows,
        },
    }


# ===========================================================================
# SECTION 3 - EXPORT FUNCTIONS
# ===========================================================================

def export_transformed_points(result, method_name):
    if not result or 'transformed_stations' not in result:
        return
    
    all_points = []
    for station in result['transformed_stations']:
        if station.get('coords') is not None:
            coords = station['coords']
            for i, pid in enumerate(station['all_ids']):
                if i < len(coords):
                    point = {
                        'point_id': pid,
                        'station_origin': station['name'],
                        'station_number': station.get('number', 0),
                        'east': coords[i, 0],
                        'north': coords[i, 1],
                        'height': coords[i, 2],
                        'is_control_point': pid.startswith('CP'),
                        'adjustment_type': result['type']
                    }
                    if station.get('transform'):
                        point['transform_scale'] = station['transform'][0]
                        point['transform_trans_e'] = station['transform'][2][0]
                        point['transform_trans_n'] = station['transform'][2][1]
                        point['transform_trans_h'] = station['transform'][2][2]
                    if station.get('rmse'):
                        point['registration_rmse'] = station['rmse']
                    all_points.append(point)
    
    if all_points:
        df = pd.DataFrame(all_points)
        output_file = os.path.join(NETWORK_OUTPUT_FOLDER, f'network_points_{method_name}.csv')
        df.to_csv(output_file, index=False)
        print(f"  Exported {len(df)} points to: {output_file}")
        return df
    return None


def export_parameters(result, method_name):
    if not result:
        return
    
    params_data = []
    for station in result['stations']:
        row = {
            'station': station['station'],
            'station_number': station.get('number', 0),
            'scale': station.get('scale', np.nan),
            'rmse': station.get('rmse', np.nan),
            'n_control_points': station.get('n_points', 0),
            'success': station.get('success', False)
        }
        if 'translation' in station:
            row['trans_e'] = station['translation'][0]
            row['trans_n'] = station['translation'][1]
            row['trans_h'] = station['translation'][2]
        params_data.append(row)
    
    df = pd.DataFrame(params_data)
    output_file = os.path.join(NETWORK_OUTPUT_FOLDER, f'network_params_{method_name}.csv')
    df.to_csv(output_file, index=False)
    print(f"  Parameters saved to: {output_file}")


def export_control_points(result, method_name):
    if not result:
        return
    
    cp_data = []
    for i, cp_id in enumerate(result['control_points']['ids']):
        row = {
            'control_point': cp_id,
            'target_east': result['control_points']['target_coords'][i, 0],
            'target_north': result['control_points']['target_coords'][i, 1],
            'target_height': result['control_points']['target_coords'][i, 2],
            'n_observations': len(result['control_points']['connectivity'].get(cp_id, []))
        }
        cp_data.append(row)
    
    df = pd.DataFrame(cp_data)
    output_file = os.path.join(NETWORK_OUTPUT_FOLDER, f'network_cps_{method_name}.csv')
    df.to_csv(output_file, index=False)
    print(f"  Control points saved to: {output_file}")



# ===========================================================================
# SECTION 4 - PLOTTING
# ===========================================================================

def plot_network_results(results, fig=None):
    if not results:
        return
    if plt is None:
        logger.warning("matplotlib is not available; network plots were not generated.")
        return None
    
    if fig is None:
        fig = plt.figure(figsize=(14, 10))
    else:
        fig.clear()
        
    axes = fig.subplots(2, 2)
    
    ax1 = axes[0, 0]
    for method, result in results.items():
        if result:
            stations = [s['station'] for s in result['stations'] if s.get('success')]
            rmses = [s['rmse'] for s in result['stations'] if s.get('success') and 'rmse' in s]
            if stations and rmses:
                ax1.plot(stations, rmses, 'o-', label=method, linewidth=2, markersize=8)
    ax1.set_xlabel('Station')
    ax1.set_ylabel('RMSE (m)')
    ax1.set_title('Network Adjustment - RMSE per Station')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    ax2 = axes[0, 1]
    for method, result in results.items():
        if result:
            stations = [s['station'] for s in result['stations'] if s.get('success') and not s.get('is_reference')]
            scales = [s['scale'] for s in result['stations'] if s.get('success') and not s.get('is_reference') and 'scale' in s]
            if stations and scales:
                ax2.plot(stations, scales, 'o-', label=method, linewidth=2, markersize=8)
    ax2.set_xlabel('Station')
    ax2.set_ylabel('Scale Factor')
    ax2.set_title('Network Adjustment - Scale Factors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    ax3 = axes[1, 0]
    for method, result in results.items():
        if result:
            stations = [s['station'] for s in result['stations'] if s.get('success')]
            n_cps = [s['n_points'] for s in result['stations'] if s.get('success')]
            if stations and n_cps:
                ax3.bar(stations, n_cps, alpha=0.7, label=method)
    ax3.set_xlabel('Station')
    ax3.set_ylabel('Number of Control Points')
    ax3.set_title('Control Points per Station')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    ax4 = axes[1, 1]
    for method, result in results.items():
        if result and 'control_points' in result and 'connectivity' in result['control_points']:
            conn = result['control_points']['connectivity']
            n_stations = [len(v) for v in conn.values()]
            if n_stations:
                ax4.hist(n_stations, bins=range(2, max(n_stations)+2), alpha=0.5, label=method, density=True)
    ax4.set_xlabel('Number of Stations per Control Point')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Network Connectivity Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ===========================================================================
# SECTION 5 - MAIN
# ===========================================================================

def network_main():
    print("="*80)
    print("NETWORK ADJUSTMENT - All stations simultaneously")
    print("="*80)
    
    print("\n[LOAD] Loading station data...")
    stations = load_station_data(DATA_FOLDER)
    print(f"Loaded {len(stations)} stations:")
    for st in stations:
        print(f"  {st['name']} (Num:{st['number']}) - {len(st['ctrl_ids'])} CPs")
    
    methods = [
        N_CUQ_LM,
    ]
    
    results = {}
    
    for method_class in methods:
        print(f"\n{'#'*70}")
        print(f"Running {method_class.__name__}")
        print(f"{'#'*70}")
        
        result = run_network_adjustment(stations, method_class)
        if result:
            results[method_class.__name__] = result
            export_transformed_points(result, method_class.__name__)
            export_parameters(result, method_name=method_class.__name__)
            export_control_points(result, method_class.__name__)
            export_baarda_outputs(result, NETWORK_OUTPUT_FOLDER, method_class.__name__)
    
    if results:
        print("\n[PLOTS] Generating plots...")
        plot_network_results(results)
        
        summary = {}
        for method, result in results.items():
            successful = [s for s in result['stations'] if s.get('success', False)]
            avg_rmse = np.mean([s['rmse'] for s in successful if 'rmse' in s]) if successful else np.nan
            summary[method] = {
                'type': result['type'],
                'average_rmse': avg_rmse,
                'n_stations': len(successful),
                'n_control_points': result['n_control_points'],
                'computation_time': result.get('time', 0)
            }
        
        summary_file = os.path.join(NETWORK_OUTPUT_FOLDER, 'network_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"\n[OK] Summary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("[OK] NETWORK ADJUSTMENT COMPLETE!")
    print(f"[OUT] Outputs saved to: {NETWORK_OUTPUT_FOLDER}")
    print("="*80)


def run_network_adjustment(stations, method_class, selected_control_points=None, min_paired_cps=4,
                           use_baarda_data_snooping=False,
                           baarda_threshold=3.0,
                           baarda_max_iterations=10,
                           baarda_min_observations_per_point=4,
                           baarda_remove_strategy="none",
                           baarda_use_posterior_sigma=True,
                          max_iter=50, tol=1e-8, weight_constraint=1e5, linear_solver='legacy_normal', initial_network_result=None):
    print(f"\n{'='*70}")
    print(f"NETWORK ADJUSTMENT using {method_class.__name__}")
    print(f"{'='*70}")
    
    network_cps = find_cps_in_at_least_n_stations(stations, MIN_STATIONS_PER_CP)
    if selected_control_points:
        selected = set(selected_control_points)
        network_cps = [cp for cp in network_cps if cp in selected]
    print(f"Found {len(network_cps)} CPs in >= {MIN_STATIONS_PER_CP} stations")
    
    if len(network_cps) < min_paired_cps:
        print(f"[WARN] Only {len(network_cps)} CPs found (need {min_paired_cps})")
        return None
    
    connectivity = build_connectivity(stations, network_cps)
    cp_counts = [len(connectivity[cp]) for cp in network_cps]
    print(f"  Avg stations/CP: {np.mean(cp_counts):.2f}")
    print(f"  Min/Max: {np.min(cp_counts)}/{np.max(cp_counts)}")
    
    if is_global_lm_method(method_class):
        return network_adjustment_lm(
            stations,
            network_cps,
            connectivity,
            method_class,
            min_paired_cps,
            max_iter=max_iter, tol=tol, weight_constraint=weight_constraint, linear_solver=linear_solver,
            initial_network_result=initial_network_result,
            use_baarda_data_snooping=use_baarda_data_snooping,
            baarda_threshold=baarda_threshold,
            baarda_max_iterations=baarda_max_iterations,
            baarda_min_observations_per_point=baarda_min_observations_per_point,
            baarda_remove_strategy=baarda_remove_strategy,
            baarda_use_posterior_sigma=baarda_use_posterior_sigma,
        )
    else:
        return network_adjustment_sequential(
            stations,
            network_cps,
            connectivity,
            method_class,
            min_paired_cps,
            use_baarda_data_snooping=use_baarda_data_snooping,
            baarda_threshold=baarda_threshold,
            baarda_max_iterations=baarda_max_iterations,
            baarda_min_observations_per_point=baarda_min_observations_per_point,
            baarda_remove_strategy=baarda_remove_strategy,
            baarda_use_posterior_sigma=baarda_use_posterior_sigma,
        )

# ===========================================================================
# PUBLIC ENTRY POINTS
# ===========================================================================

def run_sequential_registration(stations, method_class, selected_control_points=None, min_paired_cps=4,
                                use_baarda_data_snooping=False,
                                baarda_threshold=3.0,
                                baarda_max_iterations=10,
                                baarda_min_observations_per_point=4,
                                baarda_remove_strategy="none",
                                baarda_use_posterior_sigma=True):
    """Run the sequential station-by-station registration workflow."""
    return sequential_registration(
        stations,
        method_class,
        selected_control_points=selected_control_points,
        min_paired_cps=min_paired_cps,
        use_baarda_data_snooping=use_baarda_data_snooping,
        baarda_threshold=baarda_threshold,
        baarda_max_iterations=baarda_max_iterations,
        baarda_min_observations_per_point=baarda_min_observations_per_point,
        baarda_remove_strategy=baarda_remove_strategy,
        baarda_use_posterior_sigma=baarda_use_posterior_sigma,
    )


def run_registration(stations, method_class, registration_type="sequential", selected_control_points=None, min_paired_cps=4,
                     use_baarda_data_snooping=False,
                     baarda_threshold=3.0,
                     baarda_max_iterations=10,
                     baarda_min_observations_per_point=4,
                     baarda_remove_strategy="none",
                     baarda_use_posterior_sigma=True):
    """Dispatch to the requested registration workflow."""
    registration_type = (registration_type or "sequential").lower()
    if registration_type == "network":
        return run_network_adjustment(
            stations,
            method_class,
            selected_control_points=selected_control_points,
            min_paired_cps=min_paired_cps,
            use_baarda_data_snooping=use_baarda_data_snooping,
            baarda_threshold=baarda_threshold,
            baarda_max_iterations=baarda_max_iterations,
            baarda_min_observations_per_point=baarda_min_observations_per_point,
            baarda_remove_strategy=baarda_remove_strategy,
            baarda_use_posterior_sigma=baarda_use_posterior_sigma,
        )
    if registration_type == "sequential":
        return run_sequential_registration(
            stations,
            method_class,
            selected_control_points=selected_control_points,
            min_paired_cps=min_paired_cps,
            use_baarda_data_snooping=use_baarda_data_snooping,
            baarda_threshold=baarda_threshold,
            baarda_max_iterations=baarda_max_iterations,
            baarda_min_observations_per_point=baarda_min_observations_per_point,
            baarda_remove_strategy=baarda_remove_strategy,
            baarda_use_posterior_sigma=baarda_use_posterior_sigma,
        )
    raise ValueError(f"Unknown registration_type: {registration_type!r}")


def main(registration_type="sequential"):
    """Command-line friendly entry point for either workflow."""
    if (registration_type or "sequential").lower() == "network":
        return network_main()
    return sequential_main()


__all__ = [
    "run_registration",
    "run_sequential_registration",
    "run_network_adjustment",
    "sequential_registration",
    "network_adjustment_lm",
    "network_adjustment_sequential",
    "register_station_to_reference",
    "build_accumulated_reference",
    "find_cps_in_at_least_n_stations",
    "build_connectivity",
    "load_station_data",
    "get_coords_for_cps",
    "export_detailed_report",
    "export_registered_points",
    "export_transformed_points",
    "export_parameters",
    "export_control_points",
    "plot_registration_results",
    "plot_network_results",
    "print_sequential_summary",
    "compare_methods_sequential",
    "sequential_main",
    "network_main",
    "main",
    "Method1_SequentialSVD",
    "Schut1973_SpaceTime",
    "Stepien2020_QST",
    "Method4_GlobalEulerLM",
    "Tao2020_CMTLS_Rigorous",
    "N_CUQ_LM",
]


if __name__ == "__main__":
    main()
