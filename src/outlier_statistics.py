"""Residual outlier detection for geometry-preserving observation exclusion.

The adjustment code remains responsible for solving transformations. This
module only screens residuals, recommends removals, records traceability and
keeps the legacy analyzer entry points stable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


OUTLIER_METHOD_CHOICES = ("baarda", "mad")
DEFAULT_OUTLIER_METHODS = ("baarda", "mad")
DISABLED_OUTLIER_METHODS = ("sigma", "raw_3sigma", "percentile", "rms_factor", "old_voting", "component")

OUTLIER_CONFIG = {
    "enabled": True,
    "mode": "automatic",
    "active_tests": {"baarda": True, "mad": True},
    "disabled_tests": list(DISABLED_OUTLIER_METHODS),
    "baarda": {
        "enabled": True,
        "mode": "iterative_adjustment",
        "threshold": 3.0,
        "max_iterations": 10,
        "remove_one_at_a_time": True,
        "use_posterior_sigma": True,
    },
    "mad": {
        "enabled": True,
        "mode": "iterative_adjustment",
        "threshold": 3.5,
        "scale_factor": 1.4826,
        "minimum_observations": 8,
        "numerical_epsilon": 1e-12,
        "apply_to": ["residual_norm_3d", "components"],
        "max_iterations": 10,
        "remove_one_at_a_time": True,
    },
    "removal_limits": {
        "min_observations_per_point": 4,
        "min_observations_per_station": 4,
        "maximum_condition_number": 1e12,
        "max_removed_observations_total_percent": 10.0,
        "max_removed_observations_per_iteration": 1,
        "max_removed_points_total_percent": 10.0,
        "do_not_remove_control_points": False,
        "rollback_if_solution_worsens": True,
        "max_allowed_rmse_increase_percent": 5.0,
    },
}

METHOD_OUTLIER_STRATEGY = {
    "SVD-P": {"iterative_tests": ["mad"], "post_adjustment_tests": []},
    "SCH-Q": {"iterative_tests": ["mad"], "post_adjustment_tests": []},
    "Q-ST": {"iterative_tests": ["baarda", "mad"], "post_adjustment_tests": []},
    "CMTLS": {"iterative_tests": ["baarda", "mad"], "post_adjustment_tests": []},
    "TFS": {"iterative_tests": ["mad"], "post_adjustment_tests": []},
    "N-CUQ-LM": {"iterative_tests": ["baarda", "mad"], "post_adjustment_tests": []},
}

OUTLIER_TRACE_COLUMNS = [
    "iteration",
    "run_label",
    "stop_condition",
    "solution_status",
    "station",
    "point",
    "exclusion_scope",
    "methods_triggered",
    "modes_triggered",
    "source_run_labels",
    "detection_class",
    "n_outlier_flags",
    "outlier_score",
    "baarda_score",
    "mad_score",
    "w_x",
    "w_y",
    "w_z",
    "critical_component",
    "v_norm",
    "v_x_before",
    "v_y_before",
    "v_z_before",
    "residual_norm_before",
    "exclusion_reason",
    "controls_remaining_at_station",
    "rank_before",
    "rank_after",
    "condition_number_before",
    "condition_number_after",
    "baarda_available",
    "baarda_unavailable_reason",
    "mad_reliable",
    "mad_warning",
    "mad_degenerate",
    "available_points_before",
    "minimum_points_required",
    "available_points_after",
    "new_exclusions_this_iteration",
    "cumulative_removed_after_selection",
    "residual_rows_before",
    "residual_rows_after",
    "global_rmse_before",
    "global_rmse_after",
    "global_rmse_change",
    "global_rmse_improvement_pct",
]

TRACEABILITY_COLUMNS = [
    "scenario",
    "method",
    "outlier_test",
    "mode",
    "iteration_or_cycle",
    "observation_id",
    "original_index",
    "point_id",
    "station_id",
    "component",
    "residual_x",
    "residual_y",
    "residual_z",
    "residual_norm_3d",
    "test_statistic",
    "threshold",
    "qvv",
    "redundancy_number",
    "mad",
    "robust_sigma_mad",
    "detected",
    "removed",
    "removal_reason",
    "not_removed_reason",
    "is_control_point",
    "num_observations_point_before",
    "num_observations_point_after",
    "rmse_before",
    "rmse_after",
    "sigma0_before",
    "sigma0_after",
]


def finite_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def safe_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", ""}:
            return False
    try:
        return bool(value)
    except Exception:
        return default


def robust_mad(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad


def active_outlier_methods(args):
    requested = set(getattr(args, "outlier_methods", []) or [])
    if "none" in requested:
        return set()
    if "all" in requested or not requested:
        return set(DEFAULT_OUTLIER_METHODS)
    return {method for method in requested if method in OUTLIER_METHOD_CHOICES}


def residual_global_rmse(residual_df):
    if residual_df is None or residual_df.empty or "v_norm" not in residual_df.columns:
        return np.nan
    norms = pd.to_numeric(residual_df["v_norm"], errors="coerce").dropna().to_numpy(dtype=float)
    if norms.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(norms ** 2)))


def _metadata_from_row(row, idx=0):
    point = row.get("point", row.get("point_id", ""))
    station = row.get("station", row.get("station_id", ""))
    method = row.get("method", "")
    mode = row.get("mode", "")
    component = row.get("component", "residual_norm_3d")
    return {
        "observation_id": f"{method}|{mode}|{station}|{point}|{component}|{idx}",
        "original_index": idx,
        "point_id": str(point),
        "station_id": str(station),
        "component": component,
        "method": method,
        "scenario": row.get("run_label", mode),
        "is_control_point": bool(row.get("is_control_point", True)),
    }


def _residual_values(residuals, apply_to="residual_norm_3d"):
    arr = np.asarray(residuals, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 3 and apply_to == "residual_norm_3d":
        return np.linalg.norm(arr[:, :3], axis=1), "residual_norm_3d"
    if arr.ndim == 2 and arr.shape[1] >= 3 and apply_to in {"X", "Y", "Z", "vx", "vy", "vz"}:
        idx = {"X": 0, "Y": 1, "Z": 2, "vx": 0, "vy": 1, "vz": 2}[apply_to]
        return arr[:, idx], apply_to[-1].upper()
    return arr.reshape(-1), apply_to


def _pinv_or_inv(matrix, allow_pseudoinverse=True):
    matrix = np.asarray(matrix, dtype=float)
    try:
        if np.linalg.cond(matrix) > 1e12:
            return np.linalg.pinv(matrix) if allow_pseudoinverse else np.linalg.inv(matrix)
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        if allow_pseudoinverse:
            return np.linalg.pinv(matrix)
        raise


def baarda_data_snooping(
    A,
    L,
    P,
    residuals,
    sigma0_hat=None,
    observation_metadata=None,
    threshold=3.0,
    use_posterior_sigma=True,
    sigma_apriori=1.0,
    allow_pseudoinverse=True,
):
    warnings = []

    def unavailable(reason, message, rank_a=np.nan, dof=np.nan, condition_number=np.nan):
        return {
            "test": "baarda",
            "threshold": threshold,
            "max_abs_statistic": np.nan,
            "recommended_removal": None,
            "observation_diagnostics": [],
            "warnings": [message],
            "can_be_applied": False,
            "baarda_available": False,
            "baarda_unavailable_reason": reason,
            "rank_a": finite_float(rank_a),
            "degrees_of_freedom": finite_float(dof),
            "condition_number": finite_float(condition_number),
        }

    if A is None or L is None or residuals is None:
        msg = "Baarda Data Snooping could not be applied because adjustment matrices are not available."
        return unavailable("MISSING_STOCHASTIC_MODEL", msg)
    A = np.asarray(A, dtype=float)
    v = np.asarray(residuals, dtype=float).reshape(-1)
    if A.ndim != 2:
        msg = "Baarda Data Snooping could not be applied because A is not a two-dimensional design matrix."
        return unavailable("RANK_DEFICIENT_DESIGN_MATRIX", msg)
    n_obs = A.shape[0]
    if n_obs != v.size:
        msg = "Baarda Data Snooping could not be applied because A and residuals have incompatible shapes."
        return unavailable("NUMERICAL_FAILURE", msg)
    if n_obs == 0 or A.shape[1] == 0 or not np.all(np.isfinite(A)) or not np.all(np.isfinite(v)):
        msg = "Baarda Data Snooping could not be applied because A or residuals contain no finite physical equations."
        return unavailable("NUMERICAL_FAILURE", msg)
    P = np.eye(n_obs) if P is None else np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = np.diag(P)
    if P.shape != (n_obs, n_obs) or not np.all(np.isfinite(P)):
        msg = "Baarda Data Snooping could not be applied because the weight matrix is invalid."
        return unavailable("INVALID_WEIGHT_MATRIX", msg)
    if np.any(np.diag(P) <= 0):
        msg = "Baarda Data Snooping could not be applied because the weight matrix has non-positive diagonal entries."
        return unavailable("INVALID_WEIGHT_MATRIX", msg)

    rank_a = int(np.linalg.matrix_rank(A))
    condition_number = float(np.linalg.cond(A)) if A.size else np.nan
    dof = int(n_obs - rank_a)
    if rank_a <= 0:
        msg = "Baarda Data Snooping could not be applied because the physical design matrix has insufficient rank."
        return unavailable("RANK_DEFICIENT_DESIGN_MATRIX", msg, rank_a, dof, condition_number)
    if dof <= 0:
        msg = "Baarda Data Snooping could not be applied because degrees of freedom <= 0."
        return unavailable("NON_POSITIVE_DEGREES_OF_FREEDOM", msg, rank_a, dof, condition_number)

    try:
        N = A.T @ P @ A
        Qxx = _pinv_or_inv(N, allow_pseudoinverse)
        Qll = _pinv_or_inv(P, allow_pseudoinverse)
        Qvv = Qll - A @ Qxx @ A.T
    except Exception as exc:
        msg = f"Baarda Data Snooping could not be applied because matrix inversion failed: {exc!r}"
        return unavailable("NUMERICAL_FAILURE", msg, rank_a, dof, condition_number)

    if sigma0_hat is None or use_posterior_sigma:
        try:
            sigma0_squared = float(v.T @ P @ v) / dof
        except Exception as exc:
            msg = f"Baarda Data Snooping could not be applied because posterior variance failed: {exc!r}"
            return unavailable("INVALID_POSTERIOR_VARIANCE", msg, rank_a, dof, condition_number)
        if not np.isfinite(sigma0_squared) or sigma0_squared <= 0:
            msg = "Baarda Data Snooping could not be applied because posterior variance is not positive."
            return unavailable("INVALID_POSTERIOR_VARIANCE", msg, rank_a, dof, condition_number)
        sigma0_hat = float(np.sqrt(sigma0_squared))
    else:
        sigma0_hat = float(sigma_apriori)
        sigma0_squared = sigma0_hat ** 2
        if not np.isfinite(sigma0_hat) or sigma0_hat <= 0:
            msg = "Baarda Data Snooping could not be applied because the a priori sigma is not positive."
            return unavailable("INVALID_POSTERIOR_VARIANCE", msg, rank_a, dof, condition_number)

    qvv_diag = np.diag(Qvv)
    if qvv_diag.size != n_obs or np.any(~np.isfinite(qvv_diag)) or np.any(qvv_diag <= 0):
        msg = "Baarda Data Snooping could not be applied because Qvv has a non-positive or invalid diagonal."
        return unavailable("NON_POSITIVE_QVV_DIAGONAL", msg, rank_a, dof, condition_number)
    w = v / (sigma0_hat * np.sqrt(qvv_diag))
    redundancy = np.diag(P) * qvv_diag if np.allclose(P, np.diag(np.diag(P))) else np.full(n_obs, np.nan)
    metadata = observation_metadata or [{} for _ in range(n_obs)]

    diagnostics = []
    for idx in range(n_obs):
        meta = metadata[idx] if idx < len(metadata) else {}
        diagnostics.append({
            **meta,
            "observation_index": idx,
            "residual": float(v[idx]) if np.isfinite(v[idx]) else np.nan,
            "qvv": float(qvv_diag[idx]) if np.isfinite(qvv_diag[idx]) else np.nan,
            "redundancy_number": float(redundancy[idx]) if np.isfinite(redundancy[idx]) else np.nan,
            "standardized_residual_w": float(w[idx]) if np.isfinite(w[idx]) else np.nan,
            "detected": bool(np.isfinite(w[idx]) and abs(w[idx]) > threshold),
            "baarda_available": True,
            "baarda_unavailable_reason": "",
            "degrees_of_freedom": dof,
            "rank_a": rank_a,
            "condition_number": condition_number,
        })

    if np.all(~np.isfinite(w)):
        recommended = None
        max_abs = np.nan
    else:
        max_idx = int(np.nanargmax(np.abs(w)))
        max_abs = float(abs(w[max_idx]))
        recommended = diagnostics[max_idx] if max_abs > threshold else None
    return {
        "test": "baarda",
        "threshold": threshold,
        "max_abs_statistic": max_abs,
        "recommended_removal": recommended,
        "observation_diagnostics": diagnostics,
        "warnings": warnings,
        "can_be_applied": True,
        "baarda_available": True,
        "baarda_unavailable_reason": "",
        "rank_a": rank_a,
        "degrees_of_freedom": dof,
        "condition_number": finite_float(condition_number),
        "sigma0_hat": sigma0_hat,
        "sigma0_squared": sigma0_squared,
    }


def mad_residual_screening(
    residuals,
    observation_metadata=None,
    threshold=3.5,
    scale_factor=1.4826,
    apply_to="residual_norm_3d",
    iteration=0,
    minimum_observations=8,
    numerical_epsilon=1e-12,
):
    values, component = _residual_values(residuals, apply_to=apply_to)
    values = np.asarray(values, dtype=float)
    warnings = []
    if values.size == 0 or np.all(~np.isfinite(values)):
        warnings.append("MAD could not be applied because residual values are missing.")
        return {
            "test": "mad",
            "threshold": threshold,
            "median": np.nan,
            "mad": np.nan,
            "robust_sigma": np.nan,
            "max_abs_statistic": np.nan,
            "recommended_removal": None,
            "observation_diagnostics": [],
            "warnings": warnings,
            "can_be_applied": False,
            "mad_reliable": False,
            "mad_warning": "MISSING_RESIDUALS",
            "mad_degenerate": False,
        }
    median, mad = robust_mad(values)
    robust_sigma = float(scale_factor * mad) if np.isfinite(mad) else np.nan
    finite_count = int(np.count_nonzero(np.isfinite(values)))
    mad_reliable = finite_count >= int(minimum_observations or 8)
    mad_warning = "" if mad_reliable else "SMALL_SAMPLE"
    if not mad_reliable:
        warnings.append("MAD sample is smaller than the configured minimum; diagnostic only for MAD-only candidates.")
    mad_degenerate = bool(not np.isfinite(robust_sigma) or robust_sigma <= numerical_epsilon)
    if mad_degenerate:
        warnings.append("MAD distribution is degenerate; scores are diagnostic and require safeguards before removal.")
        z = np.zeros(values.size, dtype=float)
        for idx, value in enumerate(values):
            if not np.isfinite(value):
                z[idx] = np.nan
            elif np.isclose(value, median, atol=numerical_epsilon, rtol=0.0):
                z[idx] = 0.0
            else:
                z[idx] = np.inf
    else:
        z = np.abs(values - median) / robust_sigma
    metadata = observation_metadata or [{} for _ in range(values.size)]
    diagnostics = []
    for idx, value in enumerate(values):
        meta = metadata[idx] if idx < len(metadata) else {}
        detected = bool(
            np.isfinite(value)
            and (np.isinf(z[idx]) or np.isfinite(z[idx]))
            and z[idx] > threshold
            and value > median
        )
        diagnostics.append({
            **meta,
            "observation_index": idx,
            "component": meta.get("component", component),
            "residual_value": float(value) if np.isfinite(value) else np.nan,
            "median": median,
            "mad": mad,
            "robust_sigma_mad": robust_sigma,
            "z_mad": float(z[idx]) if np.isfinite(z[idx]) else np.nan,
            "mad_score": float(z[idx]) if np.isfinite(z[idx]) else (np.inf if np.isinf(z[idx]) else np.nan),
            "detected": detected,
            "mad_reliable": mad_reliable,
            "mad_warning": mad_warning,
            "mad_degenerate": mad_degenerate,
            "iteration": iteration,
        })
    detected_rows = [row for row in diagnostics if row.get("detected")]
    if detected_rows:
        recommended = max(
            detected_rows,
            key=lambda row: (
                row.get("mad_score", -np.inf)
                if np.isfinite(row.get("mad_score", np.nan)) or np.isinf(row.get("mad_score", np.nan))
                else -np.inf,
                row.get("residual_value", -np.inf),
            ),
        )
        max_z = recommended.get("mad_score", np.nan)
    else:
        recommended = None
        finite_z = z[np.isfinite(z)]
        max_z = float(np.max(finite_z)) if finite_z.size else (np.inf if np.any(np.isinf(z)) else np.nan)
    return {
        "test": "mad",
        "threshold": threshold,
        "median": median,
        "mad": mad,
        "robust_sigma": robust_sigma,
        "max_abs_statistic": max_z,
        "recommended_removal": recommended,
        "observation_diagnostics": diagnostics,
        "warnings": warnings,
        "can_be_applied": True,
        "mad_reliable": mad_reliable,
        "mad_warning": mad_warning,
        "mad_degenerate": mad_degenerate,
    }


def validate_outlier_removal(observation, active_observations, removal_history, removal_limits):
    limits = {**OUTLIER_CONFIG["removal_limits"], **(removal_limits or {})}
    if not observation:
        return {"can_remove": False, "reason": "no_candidate"}
    point_id = str(observation.get("point_id", observation.get("point", "")))
    station_id = str(observation.get("station_id", observation.get("station", "")))
    if not point_id or not station_id:
        return {"can_remove": False, "reason": "missing_metadata"}
    if limits.get("do_not_remove_control_points", False) and bool(observation.get("is_control_point", True)):
        return {"can_remove": False, "reason": "control_point_protected"}
    active = pd.DataFrame(active_observations)
    if active.empty:
        return {"can_remove": False, "reason": "no_active_observations"}
    point_col = "point" if "point" in active.columns else "point_id"
    station_col = "station" if "station" in active.columns else "station_id"
    point_count = int(active[active[point_col].astype(str).eq(point_id)].shape[0])
    station_count = int(active[active[station_col].astype(str).eq(station_id)].shape[0])
    if point_count - 1 < int(limits.get("min_observations_per_point", 4)):
        return {"can_remove": False, "reason": "minimum_observations_per_point"}
    if station_count - 1 < int(limits.get("min_observations_per_station", 4)):
        return {"can_remove": False, "reason": "minimum_observations_per_station"}
    max_removed = int(math.floor(len(active) * float(limits.get("max_removed_observations_total_percent", 10.0)) / 100.0))
    if len(removal_history or []) >= max(1, max_removed):
        return {"can_remove": False, "reason": "removal_limit_reached"}
    return {"can_remove": True, "reason": "accepted"}


def select_outlier_to_remove(baarda_result=None, mad_result=None, removal_limits=None, active_observations=None):
    candidates = []
    for result in (baarda_result, mad_result):
        if not result or not result.get("can_be_applied"):
            continue
        candidate = result.get("recommended_removal")
        statistic = finite_float(result.get("max_abs_statistic"))
        threshold = finite_float(result.get("threshold"))
        if candidate and statistic is not None and threshold is not None and statistic > threshold:
            candidates.append((result["test"], statistic / threshold if threshold else statistic, candidate))
    if not candidates:
        return {"remove": False, "reason": "no_outlier_above_threshold"}
    candidates.sort(key=lambda item: (item[0] != "baarda", -item[1]))
    test_name, _, candidate = candidates[0]
    validation = validate_outlier_removal(candidate, active_observations or [], [], removal_limits)
    return {
        "remove": bool(validation["can_remove"]),
        "observation_id": candidate.get("observation_id"),
        "point": candidate.get("point_id", candidate.get("point")),
        "station": candidate.get("station_id", candidate.get("station")),
        "outlier_test": test_name,
        "reason": f"Removed by {test_name.upper()}" if validation["can_remove"] else validation["reason"],
        "candidate": candidate,
    }


def evaluate_outliers(residual_df, args):
    active_methods = active_outlier_methods(args)
    if not active_methods or residual_df is None or residual_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    working = residual_df.copy()
    for col in ("vx", "vy", "vz", "v_norm"):
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")
    if "is_control_point" in working.columns:
        control_mask = working["is_control_point"].map(lambda value: safe_bool(value, True))
        working = working[control_mask].copy()
    if "role" in working.columns:
        working = working[~working["role"].astype(str).str.upper().eq("ICP")].copy()
    if working.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    has_run_label = "run_label" in working.columns
    group_cols = ["method", "mode", "run_label", "station"] if has_run_label else ["method", "mode", "station"]

    for keys, group in working.groupby(group_cols, dropna=False):
        if has_run_label:
            method, mode, run_label, station = keys
        else:
            method, mode, station = keys
            run_label = ""
        residual_vectors = group[["vx", "vy", "vz"]].to_numpy(dtype=float) if {"vx", "vy", "vz"}.issubset(group.columns) else group["v_norm"].to_numpy(dtype=float)
        metadata = [_metadata_from_row(row, idx) for idx, (_, row) in enumerate(group.iterrows())]
        mad_result = mad_residual_screening(
            residual_vectors,
            metadata,
            threshold=float(getattr(args, "outlier_mad", OUTLIER_CONFIG["mad"]["threshold"])),
            apply_to="residual_norm_3d",
            minimum_observations=int(getattr(args, "outlier_mad_min_observations", OUTLIER_CONFIG["mad"]["minimum_observations"])),
            numerical_epsilon=float(getattr(args, "outlier_numerical_epsilon", OUTLIER_CONFIG["mad"]["numerical_epsilon"])),
        ) if "mad" in active_methods else None
        mad_by_idx = {row["observation_index"]: row for row in (mad_result or {}).get("observation_diagnostics", [])}
        for local_idx, (_, source) in enumerate(group.iterrows()):
            mad_row = mad_by_idx.get(local_idx, {})
            flag_mad = safe_bool(mad_row.get("detected", False)) if "mad" in active_methods else False
            baarda_score_raw = source.get("baarda_score", source.get("max_abs_w", np.nan))
            try:
                baarda_score = float(baarda_score_raw)
            except (TypeError, ValueError):
                baarda_score = np.nan
            flag_baarda = safe_bool(source.get("flag_baarda", False)) if "baarda" in active_methods else False
            if "baarda" in active_methods and np.isfinite(baarda_score):
                flag_baarda = flag_baarda or baarda_score > float(getattr(args, "outlier_sigma", OUTLIER_CONFIG["baarda"]["threshold"]))
            n_flags = int(flag_baarda) + int(flag_mad)
            mad_score_raw = mad_row.get("mad_score", mad_row.get("z_mad", np.nan))
            try:
                mad_score = float(mad_score_raw)
            except (TypeError, ValueError):
                mad_score = np.nan
            if flag_baarda and np.isfinite(baarda_score):
                outlier_score = baarda_score
            elif flag_mad and (np.isfinite(mad_score) or np.isinf(mad_score)):
                outlier_score = mad_score
            else:
                outlier_score = np.nan
            if flag_baarda and flag_mad:
                detection_class = "BAARDA_AND_MAD"
            elif flag_baarda:
                detection_class = "BAARDA_ONLY"
            elif flag_mad:
                detection_class = "MAD_ONLY"
            else:
                detection_class = "NOT_FLAGGED"
            w_values = {
                "X": source.get("w_x", np.nan),
                "Y": source.get("w_y", np.nan),
                "Z": source.get("w_z", np.nan),
            }
            finite_w = {
                comp: abs(float(value))
                for comp, value in w_values.items()
                if finite_float(value) is not None
            }
            critical_component = source.get("critical_component", "")
            if not critical_component and finite_w:
                critical_component = max(finite_w, key=finite_w.get)
            rows.append({
                "method": method,
                "mode": mode,
                "run_label": run_label,
                "station": station,
                "point": source.get("point", source.get("point_id")),
                "v_norm": finite_float(source.get("v_norm")),
                "vx": source.get("vx", np.nan),
                "vy": source.get("vy", np.nan),
                "vz": source.get("vz", np.nan),
                "is_control_point": safe_bool(source.get("is_control_point", True), True),
                "is_icp": False,
                "group_size": int(len(group)),
                "median_norm": mad_row.get("median", np.nan),
                "mad_norm": mad_row.get("mad", np.nan),
                "robust_sigma_norm": mad_row.get("robust_sigma_mad", np.nan),
                "mad_score": mad_score,
                "baarda_score": baarda_score,
                "w_x": source.get("w_x", np.nan),
                "w_y": source.get("w_y", np.nan),
                "w_z": source.get("w_z", np.nan),
                "critical_component": critical_component,
                "baarda_available": safe_bool(source.get("baarda_available", False)),
                "baarda_unavailable_reason": source.get("baarda_unavailable_reason", ""),
                "rank_before": finite_float(source.get("rank_a", source.get("rank_before"))),
                "condition_number_before": finite_float(source.get("condition_number", source.get("condition_number_before"))),
                "mad_reliable": safe_bool(mad_row.get("mad_reliable", True), True),
                "mad_warning": mad_row.get("mad_warning", ""),
                "mad_degenerate": safe_bool(mad_row.get("mad_degenerate", False)),
                "flag_baarda": flag_baarda,
                "flag_mad": flag_mad,
                "detection_class": detection_class,
                "active_outlier_methods": ",".join(sorted(active_methods)),
                "n_outlier_flags": n_flags,
                "outlier_score": outlier_score,
                "is_outlier_candidate": detection_class != "NOT_FLAGGED",
            })

    outlier_df = pd.DataFrame(rows)
    if outlier_df.empty:
        return outlier_df, pd.DataFrame()
    summary_rows = []
    for keys, group in outlier_df.groupby(group_cols, dropna=False):
        if has_run_label:
            method, mode, run_label, station = keys
        else:
            method, mode, station = keys
            run_label = ""
        candidates = group[group["is_outlier_candidate"] == True]  # noqa: E712
        summary_rows.append({
            "method": method,
            "mode": mode,
            "run_label": run_label,
            "station": station,
            "points_tested": len(group),
            "outlier_candidates": len(candidates),
            "outlier_rate": len(candidates) / len(group) if len(group) else 0.0,
            "max_v_norm": group["v_norm"].max(),
            "max_mad_score": group["mad_score"].max(),
            "max_baarda_score": group["baarda_score"].max(),
            "max_flags": group["n_outlier_flags"].max(),
            "candidate_points": ", ".join(candidates["point"].astype(str).tolist()),
        })
    return outlier_df, pd.DataFrame(summary_rows)


def _available_points_for_candidate(outlier_df, station, point):
    station_rows = outlier_df[outlier_df["station"].astype(str).eq(str(station))].copy()
    if station_rows.empty:
        return 0
    if "point" not in station_rows.columns:
        return 0
    point_mask = station_rows["point"].astype(str).eq(str(point))
    group_cols = [col for col in ("method", "mode", "run_label", "station") if col in station_rows.columns]
    if not group_cols:
        return int(station_rows["point"].astype(str).nunique())
    counts = []
    for _, group in station_rows.groupby(group_cols, dropna=False):
        if point_mask.loc[group.index].any():
            counts.append(int(group["point"].astype(str).nunique()))
    if not counts:
        return 0
    return min(counts)


def network_exclusion_preserves_connectivity(
    stations,
    cumulative_exclusions,
    candidate_station,
    candidate_point,
    min_common_points=4,
):
    """Check the minimum-count reachability required by chained initialization.

    This combinatorial check is not a proof of full network rank.
    """
    min_common = max(1, int(min_common_points or 4))
    candidate_station = str(candidate_station)
    candidate_point = str(candidate_point)
    retained_by_station = {}

    for station in stations or []:
        station_name = str(station.get("name", ""))
        excluded = {
            str(point_id)
            for point_id in (cumulative_exclusions or {}).get(("*", station_name), [])
        }
        if station_name == candidate_station:
            excluded.add(candidate_point)
        retained_by_station[station_name] = {
            str(point_id)
            for point_id in station.get("ctrl_ids", [])
            if str(point_id) not in excluded
        }

    if not retained_by_station:
        return False

    point_counts = defaultdict(int)
    for point_ids in retained_by_station.values():
        for point_id in point_ids:
            point_counts[point_id] += 1
    network_points = {
        point_id for point_id, count in point_counts.items() if count >= 2
    }
    station_network_points = {
        station_name: point_ids.intersection(network_points)
        for station_name, point_ids in retained_by_station.items()
    }

    if any(
        len(point_ids) < min_common
        for point_ids in station_network_points.values()
    ):
        return False

    reference_name = max(
        station_network_points,
        key=lambda name: len(station_network_points[name]),
    )
    connected = {reference_name}
    known_points = set(station_network_points[reference_name])
    pending = set(station_network_points).difference(connected)

    while pending:
        newly_connected = {
            station_name
            for station_name in pending
            if len(station_network_points[station_name].intersection(known_points)) >= min_common
        }
        if not newly_connected:
            return False
        for station_name in newly_connected:
            known_points.update(station_network_points[station_name])
        connected.update(newly_connected)
        pending.difference_update(newly_connected)

    return len(connected) == len(station_network_points)


def coupled_network_exclusion_precheck(stations, cumulative_exclusions,
                                      candidate_station, candidate_point,
                                      min_common_points=4):
    """Count shared controls over the whole network, not per station pair.

    Connectivity and counts are necessary checks only. A transactional trial
    adjustment must additionally confirm rank, convergence and valid tests.
    """
    retained = {}
    for station in stations:
        name = str(station['name'])
        excluded = set(map(str, (cumulative_exclusions or {}).get(('*', name), [])))
        if name == str(candidate_station):
            excluded.add(str(candidate_point))
        retained[name] = set(map(str, station['ctrl_ids'])) - excluded
    owners = defaultdict(set)
    for name, points in retained.items():
        for point in points:
            owners[point].add(name)
    shared = {p for p, names in owners.items() if len(names) >= 2}
    retained = {name: points & shared for name, points in retained.items()}
    counts = {name: len(points) for name, points in retained.items()}
    if not counts or min(counts.values()) < int(min_common_points):
        return {'accepted': False, 'reason': 'MINIMUM_TOTAL_SHARED_CONTROLS', 'counts': counts}
    reached = {next(iter(retained))}
    while True:
        linked = set().union(*(owners[p] for name in reached for p in retained[name]))
        new = reached | linked
        if new == reached:
            break
        reached = new
    return {'accepted': len(reached) == len(retained),
            'reason': 'CONNECTED_COUNTS_OK' if len(reached) == len(retained) else 'DISCONNECTED_NETWORK',
            'counts': counts, 'n_shared_points': len(shared)}


def coupled_solution_validation(result, expected_stations):
    """Validate an uncommitted coupled-network trial in physical parameter space."""
    if not result or not result.get('residual_rows') or not result.get('outputs'):
        return {'accepted': False, 'reason': 'NO_SOLUTION'}
    obj = next(iter(result['outputs'].values()))
    names = {str(s['station']) for s in obj.get('stations', [])}
    if names != set(map(str, expected_stations)):
        return {'accepted': False, 'reason': 'MISSING_STATIONS'}
    if not obj.get('converged', False):
        return {'accepted': False, 'reason': 'NOT_CONVERGED', 'stop_reason': obj.get('stop_reason')}
    A = np.asarray(obj.get('statistical_design_matrix', []), dtype=float)
    if A.ndim != 2 or not A.size or not np.isfinite(A).all():
        return {'accepted': False, 'reason': 'INVALID_PHYSICAL_DESIGN'}
    scale = np.linalg.norm(A, axis=0)
    singular = np.linalg.svd(A / np.where(scale > 0, scale, 1), compute_uv=False)
    rank = int(np.sum(singular > singular[0] * max(A.shape) * np.finfo(float).eps))
    full_rank = rank == A.shape[1]
    return {'accepted': full_rank and A.shape[0] > rank,
            'reason': 'OK' if full_rank and A.shape[0] > rank else 'RANK_OR_REDUNDANCY_FAILURE',
            'rank_after': rank, 'unknowns': A.shape[1], 'dof': A.shape[0] - rank,
            'condition_number_after': float(singular[0] / singular[-1]) if singular[-1] > 0 else float('inf')}


def select_outlier_exclusions(
    outlier_df,
    min_points_per_station=4,
    max_remove_per_iteration=1,
    candidate_validator=None,
):
    exclusions = defaultdict(set)
    selected_rows = []
    if outlier_df is None or outlier_df.empty:
        return exclusions, selected_rows
    candidates = outlier_df[outlier_df["is_outlier_candidate"] == True].copy()  # noqa: E712
    if "mode" in candidates.columns:
        candidates = candidates[candidates["mode"].isin(["sequential", "network"])]
    if "is_control_point" in candidates.columns:
        candidates = candidates[candidates["is_control_point"].map(lambda value: safe_bool(value, True))]
    if "is_icp" in candidates.columns:
        candidates = candidates[~candidates["is_icp"].map(lambda value: safe_bool(value, False))]
    priority = {"BAARDA_AND_MAD": 0, "BAARDA_ONLY": 1, "MAD_ONLY": 2}
    if "detection_class" not in candidates.columns:
        candidates["detection_class"] = np.where(
            candidates.get("flag_mad", False),
            "MAD_ONLY",
            "NOT_FLAGGED",
        )
    candidates = candidates[candidates["detection_class"].isin(priority)]
    if "mad_reliable" in candidates.columns:
        mad_only = candidates["detection_class"].astype(str).eq("MAD_ONLY")
        unreliable = ~candidates["mad_reliable"].map(lambda value: safe_bool(value, False))
        candidates = candidates[~(mad_only & unreliable)]
    if "mad_degenerate" in candidates.columns:
        mad_only = candidates["detection_class"].astype(str).eq("MAD_ONLY")
        degenerate = candidates["mad_degenerate"].map(lambda value: safe_bool(value, False))
        candidates = candidates[~(mad_only & degenerate)]
    candidates = candidates.dropna(subset=["station", "point"])
    if candidates.empty:
        return exclusions, selected_rows
    max_remove = max(1, int(max_remove_per_iteration or 1))
    min_points = max(1, int(min_points_per_station or 4))
    candidates["_priority"] = candidates["detection_class"].map(priority).fillna(99).astype(int)
    candidates["_baarda_sort"] = (
        pd.to_numeric(candidates["baarda_score"], errors="coerce").fillna(-np.inf)
        if "baarda_score" in candidates.columns
        else -np.inf
    )
    candidates["_mad_sort"] = (
        pd.to_numeric(candidates["mad_score"], errors="coerce").fillna(-np.inf)
        if "mad_score" in candidates.columns
        else -np.inf
    )
    candidates["_v_sort"] = (
        pd.to_numeric(candidates["v_norm"], errors="coerce").fillna(-np.inf)
        if "v_norm" in candidates.columns
        else -np.inf
    )
    selected = candidates.sort_values(
        ["_priority", "_baarda_sort", "_mad_sort", "_v_sort", "station", "point"],
        ascending=[True, False, False, False, True, True],
    )
    for _, row in selected.iterrows():
        if len(selected_rows) >= max_remove:
            break
        station = str(row["station"])
        point = str(row["point"])
        available_before = _available_points_for_candidate(outlier_df, station, point)
        if available_before - 1 < min_points:
            continue
        validator_result = candidate_validator(station, point) if candidate_validator is not None else True
        validator_info = {}
        if isinstance(validator_result, dict):
            validator_info = validator_result
            if not bool(validator_result.get("accepted", validator_result.get("ok", False))):
                continue
        elif not validator_result:
            continue
        exclusions[("*", station)].add(point)
        triggered = []
        if safe_bool(row.get("flag_baarda", False)):
            triggered.append("baarda")
        if safe_bool(row.get("flag_mad", False)):
            triggered.append("mad")
        selected_rows.append({
            "station": station,
            "point": point,
            "exclusion_scope": "*",
            "methods_triggered": "/".join(triggered),
            "modes_triggered": row.get("mode", ""),
            "source_run_labels": row.get("run_label", ""),
            "n_outlier_flags": finite_float(row.get("n_outlier_flags")),
            "outlier_score": finite_float(row.get("outlier_score")),
            "baarda_score": finite_float(row.get("baarda_score")),
            "mad_score": finite_float(row.get("mad_score")),
            "w_x": finite_float(row.get("w_x")),
            "w_y": finite_float(row.get("w_y")),
            "w_z": finite_float(row.get("w_z")),
            "critical_component": row.get("critical_component", ""),
            "detection_class": row.get("detection_class", ""),
            "v_norm": finite_float(row.get("v_norm")),
            "v_x_before": finite_float(row.get("vx")),
            "v_y_before": finite_float(row.get("vy")),
            "v_z_before": finite_float(row.get("vz")),
            "residual_norm_before": finite_float(row.get("v_norm")),
            "exclusion_reason": "GEOMETRY_PRESERVING_CANDIDATE_ACCEPTED",
            "controls_remaining_at_station": available_before - 1,
            "rank_before": finite_float(row.get("rank_before")),
            "rank_after": finite_float(validator_info.get("rank_after")),
            "condition_number_before": finite_float(row.get("condition_number_before")),
            "condition_number_after": finite_float(validator_info.get("condition_number_after")),
            "baarda_available": safe_bool(row.get("baarda_available", False)),
            "baarda_unavailable_reason": row.get("baarda_unavailable_reason", ""),
            "mad_reliable": safe_bool(row.get("mad_reliable", True), True),
            "mad_warning": row.get("mad_warning", ""),
            "mad_degenerate": safe_bool(row.get("mad_degenerate", False)),
            "available_points_before": available_before,
            "minimum_points_required": min_points,
            "available_points_after": available_before - 1,
        })
    return exclusions, selected_rows


def outlier_exclusion_map(outlier_df, min_points_per_station=4, max_remove_per_iteration=1):
    exclusions, _ = select_outlier_exclusions(outlier_df, min_points_per_station, max_remove_per_iteration)
    return exclusions


def merge_outlier_exclusions(target, source):
    additions = 0
    for key, point_ids in (source or {}).items():
        before = len(target[key])
        target[key].update(str(point_id) for point_id in point_ids)
        additions += len(target[key]) - before
    return additions


def outlier_trace_dataframe(rows):
    return pd.DataFrame(rows, columns=OUTLIER_TRACE_COLUMNS)


def write_outlier_catalog(output_dir, args):
    output_dir = Path(output_dir)
    active_methods = active_outlier_methods(args)
    rows = [
        {
            "key": "baarda",
            "detector": "Baarda Data Snooping",
            "mode": "iterative_adjustment",
            "rule": "standardized residual |w_i| > threshold when A, P, Qvv and residuals are available",
            "active_this_run": "baarda" in active_methods,
        },
        {
            "key": "mad",
            "detector": "Median Absolute Deviation",
            "mode": "iterative_adjustment",
            "rule": "z_MAD > threshold on retained control residual norm and residual norm above median",
            "active_this_run": "mad" in active_methods,
        },
    ]
    pd.DataFrame(rows).to_csv(output_dir / "outlier_methods_catalog.csv", index=False)
    lines = [
        "# Outlier detection methods",
        "",
        "Automatic outlier exclusion is restricted to Baarda Data Snooping and MAD.",
        "Disabled legacy detectors: " + ", ".join(DISABLED_OUTLIER_METHODS) + ".",
        "",
        "Active in this run: " + (", ".join(sorted(active_methods)) if active_methods else "none"),
        "",
    ]
    for row in rows:
        lines.extend([
            f"## {row['detector']}",
            f"- Mode: {row['mode']}",
            f"- Active: {'yes' if row['active_this_run'] else 'no'}",
            f"- Rule: {row['rule']}",
            "",
        ])
    (output_dir / "outlier_methods_catalog.md").write_text("\n".join(lines), encoding="utf-8")


def _ensure_outlier_dirs(output_dir):
    base = Path(output_dir) / "outlier_diagnostics"
    dirs = {
        "base": base,
        "png": base / "figures_png",
        "pdf": base / "figures_pdf",
        "svg": base / "figures_svg",
        "csv": base / "tables_csv",
        "logs": base / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _save_figure(fig, dirs, name):
    fig.savefig(dirs["png"] / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(dirs["pdf"] / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(dirs["svg"] / f"{name}.svg", dpi=300, bbox_inches="tight")


def _traceability_from_flags(outlier_df, outlier_trace_df=None):
    rows = []
    if outlier_df is None or outlier_df.empty:
        return pd.DataFrame(columns=TRACEABILITY_COLUMNS)
    removed_points = set()
    if outlier_trace_df is not None and not outlier_trace_df.empty:
        for _, row in outlier_trace_df.iterrows():
            removed_points.add((str(row.get("station", "")), str(row.get("point", ""))))
    for idx, row in outlier_df.iterrows():
        station = str(row.get("station", ""))
        point = str(row.get("point", ""))
        detected_tests = [
            name for name in ("baarda", "mad")
            if bool(row.get(f"flag_{name}", False))
        ]
        if not detected_tests:
            detected_tests = ["none"]
        for test_name in detected_tests:
            removed = (station, point) in removed_points
            rows.append({
                "scenario": row.get("run_label", row.get("mode", "")),
                "method": row.get("method", ""),
                "outlier_test": test_name,
                "mode": "iterative_adjustment",
                "iteration_or_cycle": np.nan,
                "observation_id": f"{row.get('method','')}|{row.get('mode','')}|{station}|{point}|residual_norm_3d|{idx}",
                "original_index": idx,
                "point_id": point,
                "station_id": station,
                "component": "residual_norm_3d",
                "residual_x": row.get("vx", np.nan),
                "residual_y": row.get("vy", np.nan),
                "residual_z": row.get("vz", np.nan),
                "residual_norm_3d": row.get("v_norm", np.nan),
                "test_statistic": (
                    row.get("baarda_score", np.nan)
                    if test_name == "baarda"
                    else row.get("mad_score", np.nan)
                ),
                "threshold": np.nan,
                "qvv": row.get("qvv", np.nan),
                "redundancy_number": np.nan,
                "mad": row.get("mad_norm", np.nan),
                "robust_sigma_mad": row.get("robust_sigma_norm", np.nan),
                "lower_limit": np.nan,
                "upper_limit": np.nan,
                "extreme_lower_limit": np.nan,
                "extreme_upper_limit": np.nan,
                "detected": test_name != "none",
                "removed": removed,
                "removal_reason": f"Removed by {test_name.upper()}" if removed else "",
                "not_removed_reason": "" if removed or test_name == "none" else "reported_only_or_safety_limit",
                "is_control_point": True,
                "num_observations_point_before": np.nan,
                "num_observations_point_after": np.nan,
                "rmse_before": np.nan,
                "rmse_after": np.nan,
                "sigma0_before": np.nan,
                "sigma0_after": np.nan,
            })
    return pd.DataFrame(rows, columns=TRACEABILITY_COLUMNS)


def _before_after_summary(residual_df, outlier_trace_df):
    before = residual_global_rmse(residual_df)
    removed = 0 if outlier_trace_df is None or outlier_trace_df.empty else len(outlier_trace_df)
    rows = []
    if residual_df is None or residual_df.empty:
        return pd.DataFrame()
    group_cols = ["method", "run_label"] if "run_label" in residual_df.columns else ["method"]
    for keys, group in residual_df.groupby(group_cols, dropna=False):
        if len(group_cols) == 2:
            method, scenario = keys
        else:
            method = keys
            scenario = ""
        rmse = residual_global_rmse(group)
        rows.append({
            "scenario": scenario,
            "method": method,
            "outlier_strategy": "baarda_mad",
            "rmse_before": rmse,
            "rmse_after": rmse,
            "rmse_improvement": 0.0,
            "rmse_improvement_percent": 0.0,
            "sigma0_before": np.nan,
            "sigma0_after": np.nan,
            "max_residual_before": pd.to_numeric(group.get("v_norm"), errors="coerce").max(),
            "max_residual_after": pd.to_numeric(group.get("v_norm"), errors="coerce").max(),
            "num_observations_before": len(group),
            "num_observations_after": len(group),
            "num_removed_observations": removed,
            "num_removed_points": removed,
            "convergence_status": "reported" if removed == 0 else "rerun_completed",
                "notes": "Outlier detection restricted to Baarda and MAD.",
        })
    return pd.DataFrame(rows)


def write_outlier_diagnostics(output_dir, residual_df, outlier_df, outlier_summary_df, outlier_trace_df, args):
    dirs = _ensure_outlier_dirs(output_dir)
    traceability = _traceability_from_flags(outlier_df, outlier_trace_df)
    before_after = _before_after_summary(residual_df, outlier_trace_df)
    traceability.to_csv(dirs["csv"] / "outlier_traceability_table.csv", index=False)
    before_after.to_csv(dirs["csv"] / "outlier_before_after_summary.csv", index=False)
    if outlier_df is not None:
        outlier_df.to_csv(dirs["csv"] / "outlier_flags_baarda_mad.csv", index=False)
    if outlier_summary_df is not None:
        outlier_summary_df.to_csv(dirs["csv"] / "outlier_summary_baarda_mad.csv", index=False)

    log_lines = [
        "Outlier detection log",
        "=====================",
        "Active tests: " + ", ".join(sorted(active_outlier_methods(args))),
        "Disabled tests: " + ", ".join(DISABLED_OUTLIER_METHODS),
        "Baarda: applied only when adjustment matrices and metadata are available.",
        "MAD: residual norm/component screening.",
        f"Detected observations: {int(outlier_df['is_outlier_candidate'].sum()) if outlier_df is not None and not outlier_df.empty else 0}",
        f"Removed observations: {len(outlier_trace_df) if outlier_trace_df is not None else 0}",
    ]
    (dirs["logs"] / "outlier_detection_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not hasattr(plt, "subplots"):
        return

    if outlier_df is None or outlier_df.empty:
        return
    x = np.arange(len(outlier_df))
    if "mad_score" in outlier_df:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x, pd.to_numeric(outlier_df["mad_score"], errors="coerce").fillna(0))
        ax.axhline(float(getattr(args, "outlier_mad", 3.5)), color="red", linestyle="--")
        ax.set_title("MAD robust z values")
        ax.set_xlabel("Observation")
        ax.set_ylabel("z_MAD")
        _save_figure(fig, dirs, "mad_robust_z_values")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pd.to_numeric(outlier_df.get("v_norm"), errors="coerce").fillna(0).to_numpy(), label="residual norm")
    ax.plot(pd.to_numeric(outlier_df.get("mad_score"), errors="coerce").fillna(0).to_numpy(), label="z_MAD")
    ax.legend()
    ax.set_title("Outlier iteration history")
    _save_figure(fig, dirs, "outlier_iteration_history")
    plt.close(fig)
    if before_after is not None and not before_after.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(before_after["method"].astype(str), before_after["rmse_before"], label="before")
        ax.bar(before_after["method"].astype(str), before_after["rmse_after"], alpha=0.6, label="after")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        ax.set_title("RMSE before/after outlier removal")
        _save_figure(fig, dirs, "outlier_before_after_rmse")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("Outlier removed map XY")
    ax.text(0.5, 0.5, "XY removal map requires point coordinates", ha="center", va="center", transform=ax.transAxes)
    _save_figure(fig, dirs, "outlier_removed_map_xy")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    baarda_values = (
        pd.to_numeric(outlier_df.get("baarda_score"), errors="coerce").fillna(0)
        if "baarda_score" in outlier_df.columns
        else pd.Series(np.zeros(len(outlier_df)))
    )
    ax.bar(x, baarda_values)
    ax.axhline(float(OUTLIER_CONFIG["baarda"]["threshold"]), color="red", linestyle="--")
    ax.set_title("Baarda |w| values")
    ax.set_xlabel("Observation")
    ax.set_ylabel("|w|")
    _save_figure(fig, dirs, "baarda_w_values")
    plt.close(fig)


def run_adjustment_with_iterative_outlier_detection(*args, **kwargs):
    raise NotImplementedError("Use the analyzer-level rerun loop; this controller is reserved for observation-object workflows.")


def run_post_adjustment_outlier_screening_and_rerun(*args, **kwargs):
    raise NotImplementedError("Use evaluate_outliers/select_outlier_exclusions for the current residual-table workflow.")
