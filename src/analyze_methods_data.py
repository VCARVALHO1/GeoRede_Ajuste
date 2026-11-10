"""
Standalone audit of processing methods using the project DATA folder.

This script does not use the GUI. It loads all station files, runs each
available transformation method, and exports method inputs, outputs,
residuals, matrix inventories, summaries, and diagnostic plots.
"""

from __future__ import annotations

import argparse
import html
import inspect
import json
import math
import os
import sys
import types
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from outlier_statistics import (  # noqa: E402
    OUTLIER_METHOD_CHOICES,
    active_outlier_methods,
    baarda_data_snooping,
    evaluate_outliers,
    merge_outlier_exclusions,
    network_exclusion_preserves_connectivity,
    outlier_trace_dataframe,
    residual_global_rmse,
    select_outlier_exclusions,
    write_outlier_catalog,
    write_outlier_diagnostics,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.use = lambda *args, **kwargs: None
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    seaborn_stub = types.ModuleType("seaborn")
    sys.modules.setdefault("matplotlib", matplotlib_stub)
    sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)
    sys.modules.setdefault("seaborn", seaborn_stub)

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    PLOTLY_AVAILABLE = False
    go = None


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from freesurvey.Registration_Module import (  # noqa: E402
        Method1_SequentialSVD,
        Schut1973_SpaceTime,
        Stepien2020_QST,
        Tao2020_CMTLS_Rigorous,
        TotalFreeStation_TFS,
        create_correlation_matrix,
        evaluate_rmse,
    )
    from freesurvey.methods import (  # noqa: E402
        CMTLS,
        N_CUQ_LM,
        Q_ST,
        SCH_Q,
        SVD_P,
        TFS,
    )
    from freesurvey.results_analyzer import method_code, standardize_adjustment_outputs  # noqa: E402
    from freesurvey.data_manager import (  # noqa: E402
        DataManager,
        apply_independent_check_points as normalize_independent_check_points,
        extract_station_number,
    )
except ModuleNotFoundError as exc:
    raise ImportError(
        f"Cannot load the complete numerical methods: {exc}. "
        "Check the src/freesurvey package and install requirements.txt."
    ) from exc
    def extract_station_number(station_name):
        digits = "".join(ch for ch in str(station_name) if ch.isdigit())
        return int(digits) if digits else 999999

    def evaluate_rmse(source, target, scale, rotation, translation):
        transformed = scale * (source @ rotation.T) + translation
        diff = transformed - target
        return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    def create_correlation_matrix(cov_matrix):
        std = np.sqrt(np.diag(cov_matrix))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov_matrix / np.outer(std, std)
        corr[~np.isfinite(corr)] = 0.0
        return corr

    class Method1_SequentialSVD:
        @staticmethod
        def solve(source, target):
            started = datetime.now()
            mean_s = np.mean(source, axis=0)
            mean_t = np.mean(target, axis=0)
            source_c = source - mean_s
            target_c = target - mean_t
            h = source_c.T @ target_c
            u, _, vt = np.linalg.svd(h)
            rotation = vt.T @ u.T
            if np.linalg.det(rotation) < 0:
                vt[2, :] *= -1
                rotation = vt.T @ u.T
            scale = np.sum(target_c * (source_c @ rotation.T)) / np.sum(source_c ** 2)
            translation = mean_t - scale * (mean_s @ rotation.T)
            return scale, rotation, translation, (datetime.now() - started).total_seconds()

    class Schut1973_SpaceTime:
        @staticmethod
        def solve(source, target):
            started = datetime.now()
            mean_s = np.mean(source, axis=0)
            mean_t = np.mean(target, axis=0)
            source_c = source - mean_s
            target_c = target - mean_t
            scale = np.sqrt(np.sum(target_c ** 2) / np.sum(source_c ** 2))
            m = (source_c * scale).T @ target_c
            n = np.zeros((4, 4))
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
            eigvals, eigvecs = np.linalg.eigh(n)
            q = eigvecs[:, np.argmax(eigvals)]
            q = q / np.linalg.norm(q)
            q0, q1, q2, q3 = q
            rotation = np.array([
                [q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                [2 * (q1 * q2 + q0 * q3), q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2, 2 * (q2 * q3 - q0 * q1)],
                [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2],
            ])
            translation = mean_t - scale * (mean_s @ rotation.T)
            return scale, rotation, translation, (datetime.now() - started).total_seconds()

    class Stepien2020_QST:
        @staticmethod
        def solve(source, target):
            started = datetime.now()
            d_s = []
            d_t = []
            for i in range(len(source)):
                for j in range(i + 1, len(source)):
                    d_s.append(np.linalg.norm(source[j] - source[i]))
                    d_t.append(np.linalg.norm(target[j] - target[i]))
            d_s = np.asarray(d_s).reshape(-1, 1)
            d_t = np.asarray(d_t).reshape(-1, 1)
            scale = (np.linalg.pinv(d_s.T @ d_s) @ d_s.T @ d_t)[0, 0]
            distance_residuals_all = d_s * scale - d_t
            sigma_d = float(np.sqrt((distance_residuals_all.T @ distance_residuals_all)[0, 0] / max(1, len(distance_residuals_all) - 1)))
            mask = np.ones(len(d_s), dtype=bool)
            d_s_used = d_s
            d_t_used = d_t
            if sigma_d > 0:
                mask = np.abs(distance_residuals_all.ravel()) < 2.0 * sigma_d
                if np.count_nonzero(mask) >= 1 and np.count_nonzero(~mask) > 0:
                    d_s_used = d_s[mask]
                    d_t_used = d_t[mask]
                    scale = (np.linalg.pinv(d_s_used.T @ d_s_used) @ d_s_used.T @ d_t_used)[0, 0]
            distance_residuals = d_s_used * scale - d_t_used
            source_scaled = scale * source
            mean_s = np.mean(source_scaled, axis=0)
            mean_t = np.mean(target, axis=0)
            source_c = source_scaled - mean_s
            target_c = target - mean_t
            b, _, _, _ = np.linalg.lstsq(source_c, target_c, rcond=None)
            rotation = b.T
            initial_centroid_residuals = target_c - source_c @ rotation.T
            a_corr = np.zeros((3 * len(source), 12), dtype=float)
            l_corr = initial_centroid_residuals.ravel()
            for row, point in enumerate(source_c):
                base = 3 * row
                x, y, z = point
                a_corr[base, 0] = 1.0
                a_corr[base, 3:6] = [x, y, z]
                a_corr[base + 1, 1] = 1.0
                a_corr[base + 1, 6:9] = [x, y, z]
                a_corr[base + 2, 2] = 1.0
                a_corr[base + 2, 9:12] = [x, y, z]
            correction, _, _, _ = np.linalg.lstsq(a_corr, l_corr, rcond=None)
            centroid_correction = correction[:3]
            rotation = rotation + correction[3:].reshape(3, 3)
            quasi_rotation = rotation
            translation = mean_t - mean_s @ rotation.T + centroid_correction
            transformed = scale * (source @ rotation.T) + translation
            residuals = transformed - target
            dof = max(1, 3 * len(source) - 9)
            stats = {
                "sigma0_sq": float(np.trace(residuals.T @ residuals) / dof),
                "rmse": evaluate_rmse(source, target, scale, rotation, translation),
                "max_residual": float(np.max(np.abs(residuals))),
                "distance_design_matrix": d_s_used,
                "distance_observation_vector": d_t_used,
                "distance_residuals": distance_residuals,
                "distance_residuals_all": distance_residuals_all,
                "distance_sigma_D": sigma_d,
                "distance_pair_mask": mask,
                "scale_pairs_removed": int(np.count_nonzero(~mask)),
                "source_scaled": source_scaled,
                "source_centroid_scaled": mean_s,
                "target_centroid": mean_t,
                "source_centered_scaled": source_c,
                "target_centered": target_c,
                "initial_centroid_residuals": initial_centroid_residuals,
                "qst_correction_design_matrix": a_corr,
                "qst_correction_vector": correction,
                "qst_centroid_correction": centroid_correction,
                "quasi_rotation_matrix": quasi_rotation,
                "orthogonality_matrix": np.eye(3) - quasi_rotation.T @ quasi_rotation,
                "orthogonality_error": float(np.linalg.norm(np.eye(3) - quasi_rotation.T @ quasi_rotation)),
                "zero_deviation_max_abs": float(np.max(np.abs(residuals))),
            }
            return scale, rotation, translation, stats, (datetime.now() - started).total_seconds()

    class TotalFreeStation_TFS:
        method_code = "TFS"
        article_name = "Total Free Station fixed-scale free-station transformation"

        @staticmethod
        def solve(source, target):
            started = datetime.now()
            source = np.asarray(source, dtype=float)
            target = np.asarray(target, dtype=float)
            if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
                raise ValueError("TFS requires source and target arrays with shape (n, 3).")
            if len(source) < 3:
                raise ValueError("TFS requires at least three homologous control points.")
            mean_s = np.mean(source, axis=0)
            mean_t = np.mean(target, axis=0)
            source_c = source - mean_s
            target_c = target - mean_t
            h = source_c.T @ target_c
            u, singular_values, vt = np.linalg.svd(h)
            rotation = vt.T @ u.T
            reflection_corrected = False
            if np.linalg.det(rotation) < 0:
                vt[-1, :] *= -1
                rotation = vt.T @ u.T
                reflection_corrected = True
            scale = 1.0
            translation = mean_t - mean_s @ rotation.T
            transformed = source @ rotation.T + translation
            residuals = transformed - target
            residual_norms = np.linalg.norm(residuals, axis=1)
            dof = max(1, 3 * len(source) - 6)
            sigma0_sq = float(np.sum(residual_norms ** 2) / dof)
            orthogonality_matrix = rotation.T @ rotation - np.eye(3)
            stats = {
                "method_code": "TFS",
                "article_name": TotalFreeStation_TFS.article_name,
                "tfs_strategy": "center_of_gravity_fixed_scale_rigid_ls",
                "tfs_fixed_scale": True,
                "tfs_reference_scale": 1.0,
                "fixed_scale_applied": True,
                "fixed_scale_value": 1.0,
                "source_centroid": mean_s,
                "target_centroid": mean_t,
                "source_centered": source_c,
                "target_centered": target_c,
                "cross_covariance_h": h,
                "svd_u": u,
                "svd_singular_values": singular_values,
                "svd_vt": vt,
                "reflection_corrected": reflection_corrected,
                "rotation_determinant": float(np.linalg.det(rotation)),
                "orthogonality_matrix": orthogonality_matrix,
                "orthogonality_error": float(np.linalg.norm(orthogonality_matrix)),
                "scale_deviation": 0.0,
                "transformed_control_points": transformed,
                "residuals_xyz": residuals,
                "residual_norms": residual_norms,
                "observation_residuals": residuals.ravel(),
                "residual_sum_squares": float(np.sum(residual_norms ** 2)),
                "degrees_of_freedom": dof,
                "dof": dof,
                "sigma0_sq": sigma0_sq,
                "sigma0": math.sqrt(sigma0_sq),
                "rmse": float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))),
                "mean_residual": float(np.mean(residual_norms)),
                "std_residual": float(np.std(residual_norms)),
                "max_residual": float(np.max(residual_norms)),
            }
            return scale, rotation, translation, stats, (datetime.now() - started).total_seconds()

    class Method3_GlobalQuaternionsLM:
        is_global_lm = True
        parameterization = "quaternion"

        @classmethod
        def solve(cls, source_list, target_input, max_iter=50, tol=1e-8, verbose=False,
                  weight_constraint=1e5, use_quaternion_constraint=True):
            started = datetime.now()
            transforms = []
            residuals = []
            target_is_list = isinstance(target_input, list)
            for i, source in enumerate(source_list):
                target = target_input[i] if target_is_list else target_input
                scale, rotation, translation, _ = Schut1973_SpaceTime.solve(source, target)
                transforms.append((scale, rotation, translation))
                residuals.extend((scale * (source @ rotation.T) + translation - target).ravel())
            residuals = np.asarray(residuals, dtype=float)
            dof = max(1, residuals.size - 8 * len(source_list))
            sigma0_sq = float(np.sum(residuals ** 2) / dof)
            cov = np.eye(8 * len(source_list)) * sigma0_sq
            return transforms, sigma0_sq, cov, residuals, (datetime.now() - started).total_seconds()

    class Method4_GlobalEulerLM(Method3_GlobalQuaternionsLM):
        parameterization = "euler_zyx"

    class Tao2020_CMTLS_Rigorous:
        @staticmethod
        def solve(source, target, max_iter=30, tol=1e-6):
            started = datetime.now()
            scale, rotation, translation, _ = Method1_SequentialSVD.solve(source, target)
            transformed = scale * (source @ rotation.T) + translation
            residuals = transformed - target
            rmse = evaluate_rmse(source, target, scale, rotation, translation)
            dof = max(1, 3 * len(source) - 7)
            sigma0_sq = float(np.sum(residuals ** 2) / dof)
            q_xi = np.eye(9) * sigma0_sq
            stats = {
                "sigma0": math.sqrt(sigma0_sq),
                "sigma0_sq": sigma0_sq,
                "iterations": 1,
                "rmse": rmse,
                "max_residual": float(np.max(np.abs(residuals))),
                "dof": dof,
            }
            return scale, rotation, translation, q_xi, stats, (datetime.now() - started).total_seconds()

    SVD_P = Method1_SequentialSVD
    SCH_Q = Schut1973_SpaceTime
    Q_ST = Stepien2020_QST
    CMTLS = Tao2020_CMTLS_Rigorous
    TFS = TotalFreeStation_TFS
    N_CUQ_LM = Method3_GlobalQuaternionsLM

    def method_code(method_key):
        return str(getattr(method_key, "method_code", method_key))

    def standardize_adjustment_outputs(method_or_code, outputs=None):
        return dict(outputs or {})

try:
    from freesurvey.Survey_Registration import run_network_adjustment  # noqa: E402
    NETWORK_IMPORT_ERROR = None
except Exception as exc:
    run_network_adjustment = None
    NETWORK_IMPORT_ERROR = repr(exc)


SEQUENTIAL_METHODS = [
    ("SVD-P", SVD_P),
    ("SCH-Q", SCH_Q),
    ("Q-ST", Q_ST),
    ("CMTLS", CMTLS),
    ("TFS", TFS),
]

NETWORK_METHODS = [
    ("N-CUQ-LM", N_CUQ_LM),
]

LEGACY_METHOD_ALIASES = {
    "svd_procrustes": "SVD-P",
    "schut_1973_quaternion": "SCH-Q",
    "stepien_2020_qst": "Q-ST",
    "tao_2020_cmtls": "CMTLS",
    "stepien_2019_tfs": "TFS",
    "network_global_quaternions_lm": "N-CUQ-LM",
}

METHOD_NOTES = {
    "svd_procrustes": {
        "implemented_inputs": "S(n,3), T(n,3), optional CP selection, optional fixed scale=1, optional outlier report/remove",
        "implemented_outputs": "scale, rotation, translation, elapsed_time, centroids, centered coordinates, H, SVD U/S/Vt, residual vectors, RMSE, numeric design/normal/covariance approximation",
        "possible_inputs": "point weights and robust weights can still be added later",
        "possible_outputs": "all major audit outputs are now exported; future addition would be formal stochastic covariance from weighted observations",
    },
    "schut_1973_quaternion": {
        "implemented_inputs": "S(n,3), T(n,3), optional CP selection, optional fixed scale=1, optional outlier report/remove",
        "implemented_outputs": "scale, rotation, translation, elapsed_time, centroids, scaled source diagnostics, 4x4 quaternion matrix N, eigenvalues, eigenvectors, quaternion, residual vectors, RMSE",
        "possible_inputs": "point weights and quaternion sign convention can still be exposed later",
        "possible_outputs": "all major audit outputs are now exported; future addition would be formal weighted covariance",
    },
    "stepien_2020_qst": {
        "implemented_inputs": "S(n,3), T(n,3), optional CP selection, optional fixed scale=1, optional outlier report/remove",
        "implemented_outputs": "scale from homologous distances, robust 2-sigma distance screening, centroid-shifted coordinates, unorthogonalized QR quasi-rotation matrix, translation, residual vectors, covariance approximation",
        "possible_inputs": "pair weights and explicit distance-pair subset can still be added later",
        "possible_outputs": "all major audit outputs are now exported; future addition would be iterative centroid/QR correction history from Equation (41)",
    },
    "tao_2020_cmtls": {
        "implemented_inputs": "S(n,3), T(n,3), max_iter, tol, optional CP selection, optional fixed scale=1, optional outlier report/remove",
        "implemented_outputs": "scale, rotation, translation, Q_xi, stats, elapsed_time, numeric design/normal/covariance approximation, residual diagnostics",
        "possible_inputs": "Q_S, Q_T, initial transform, constraint options and convergence norm can still be added inside the original Tao solver",
        "possible_outputs": "G, A, Y, Q_A, Q_Y, Q_y, C, W0, W, Lagrange multipliers and E_A require modifying the original solver internals",
    },
    "stepien_2019_tfs": {
        "implemented_inputs": "S(n,3), T(n,3), optional CP selection, fixed scale lambda=1 by definition, optional outlier report/remove",
        "implemented_outputs": "control-point centres of gravity, fixed-scale rigid rotation, translation, residual vectors, RMSE, determinant and orthogonality diagnostics",
        "possible_inputs": "explicit three-point TFS subset selection could be exposed later if the experiment is designed around adjustment/check split points",
        "possible_outputs": "formal stochastic covariance would require a weighted observation-equation realization",
    },
    "network_global_quaternions_lm": {
        "implemented_inputs": "station list from DATA, common control points, constrained Method3 global quaternion solver",
        "implemented_outputs": "network stations, transformed stations, sigma0, covariance, control points, runtime",
        "possible_inputs": "datum station, minimum CP count, max_iter, tol, weight_constraint, fixed scale=1, network CP selection",
        "possible_outputs": "network design matrix, normal matrix, residual vector, parameter covariance, point covariance, connectivity graph metrics",
    },
}

METHOD_NOTES = {
    LEGACY_METHOD_ALIASES.get(method_key, method_key): notes
    for method_key, notes in METHOD_NOTES.items()
}


def load_station_data(data_folder):
    survey_data = DataManager().load_station_folder(data_folder, delimiter="auto")
    return list(getattr(survey_data, "stations", {}).values())


def apply_independent_check_points(stations, point_ids):
    return normalize_independent_check_points(stations, point_ids)


def get_coords_for_cps(station, cp_ids):
    coords = []
    for pid in cp_ids:
        try:
            coords.append(station["coords"][station["all_ids"].index(pid)])
        except ValueError:
            coords.append(np.full(3, np.nan))
    return np.asarray(coords, dtype=float)


def build_accumulated_reference(registered_dict):
    point_coords = defaultdict(list)
    all_ctrl_ids = set()
    all_icp_ids = set()
    id_order = []
    for station in registered_dict.values():
        coords = np.asarray(station.get("coords", []), dtype=float)
        ids = list(station.get("all_ids", []))
        all_ctrl_ids.update(station.get("ctrl_ids", []))
        all_icp_ids.update(station.get("icp_ids", []))
        for pid, coord in zip(ids, coords):
            if pid not in point_coords:
                id_order.append(pid)
            point_coords[pid].append(coord)
    all_ids = id_order
    return {
        "coords": np.vstack([
            np.mean(np.asarray(point_coords[pid], dtype=float), axis=0)
            for pid in all_ids
        ]),
        "all_ids": all_ids,
        "ctrl_ids": sorted(set(all_ctrl_ids), key=extract_station_number),
        "icp_ids": sorted(set(all_icp_ids), key=extract_station_number),
        "point_observation_counts": {pid: len(vals) for pid, vals in point_coords.items()},
    }


def finite_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def array_summary(value, max_values=12):
    if value is None:
        return {"type": "array", "shape": [], "values": []}
    arr = np.asarray(value)
    if arr.size == 0:
        return {"type": "array", "shape": list(arr.shape), "values": []}
    numeric = arr.astype(float, copy=False)
    flat = numeric.ravel()
    return {
        "type": "array",
        "shape": list(numeric.shape),
        "min": finite_float(np.nanmin(flat)),
        "max": finite_float(np.nanmax(flat)),
        "mean": finite_float(np.nanmean(flat)),
        "norm": finite_float(np.linalg.norm(np.nan_to_num(flat))),
        "values_preview": [finite_float(v) for v in flat[:max_values]],
    }


def jsonable(value):
    if isinstance(value, np.ndarray):
        return array_summary(value)
    if isinstance(value, (np.floating, np.integer)):
        return finite_float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def method_kwargs(method_class, args):
    signature = inspect.signature(method_class.solve)
    kwargs = {}
    if "max_iter" in signature.parameters:
        kwargs["max_iter"] = args.max_iter
    if "tol" in signature.parameters:
        kwargs["tol"] = args.tol
    if "weight_constraint" in signature.parameters:
        kwargs["weight_constraint"] = args.weight_constraint
    if "use_quaternion_constraint" in signature.parameters:
        if getattr(method_class, "force_quaternion_constraint", None) is False:
            kwargs["use_quaternion_constraint"] = False
        else:
            kwargs["use_quaternion_constraint"] = args.use_quaternion_constraint
    if "iterative_outlier_removal" in signature.parameters:
        kwargs["iterative_outlier_removal"] = False
    if "outlier_threshold" in signature.parameters:
        kwargs["outlier_threshold"] = args.outlier_sigma
    if "outlier_min_points" in signature.parameters:
        kwargs["outlier_min_points"] = args.min_common_cp
    return kwargs


def solve_fixed_scale(source, target):
    started = datetime.now()
    mean_s = np.mean(source, axis=0)
    mean_t = np.mean(target, axis=0)
    source_c = source - mean_s
    target_c = target - mean_t
    h = source_c.T @ target_c
    u, sigma, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    reflection_corrected = False
    if np.linalg.det(rotation) < 0:
        vt[2, :] *= -1
        rotation = vt.T @ u.T
        reflection_corrected = True
    scale = 1.0
    translation = mean_t - mean_s @ rotation.T
    elapsed = (datetime.now() - started).total_seconds()
    return scale, rotation, translation, elapsed, {
        "fixed_scale_applied": True,
        "fixed_scale_value": 1.0,
        "fixed_scale_svd_u": u,
        "fixed_scale_svd_singular_values": sigma,
        "fixed_scale_svd_vt": vt,
        "fixed_scale_reflection_corrected": reflection_corrected,
    }


def solve_qst_fixed_scale(source, target):
    """Run the Q-ST geometry with lambda fixed to 1.0 for the article workflow."""
    started = datetime.now()
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    n_points = len(source)

    d_source = []
    d_target = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d_source.append(np.linalg.norm(source[j] - source[i]))
            d_target.append(np.linalg.norm(target[j] - target[i]))
    d_source_all = np.asarray(d_source, dtype=float).reshape(-1, 1)
    d_target_all = np.asarray(d_target, dtype=float).reshape(-1, 1)
    distance_residuals_all = d_source_all - d_target_all
    if len(distance_residuals_all) > 1:
        sigma_d = float(
            np.sqrt(
                (distance_residuals_all.T @ distance_residuals_all)[0, 0]
                / (len(distance_residuals_all) - 1)
            )
        )
    else:
        sigma_d = 0.0
    pair_mask = np.ones(len(d_source_all), dtype=bool)
    d_source_used = d_source_all
    d_target_used = d_target_all
    if sigma_d > 0:
        pair_mask = np.abs(distance_residuals_all.ravel()) < 2.0 * sigma_d
        if np.count_nonzero(pair_mask) >= 1 and np.count_nonzero(~pair_mask) > 0:
            d_source_used = d_source_all[pair_mask]
            d_target_used = d_target_all[pair_mask]
    distance_residuals = d_source_used - d_target_used

    scale = 1.0
    source_scaled = source
    source_centroid_scaled = np.mean(source_scaled, axis=0)
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source_scaled - source_centroid_scaled
    target_centered = target - target_centroid

    b_matrix, _, _, _ = np.linalg.lstsq(source_centered, target_centered, rcond=None)
    quasi_rotation = b_matrix.T
    initial_centroid_residuals = target_centered - source_centered @ quasi_rotation.T

    correction_design = np.zeros((3 * n_points, 12), dtype=float)
    correction_observation = initial_centroid_residuals.ravel()
    for row, point in enumerate(source_centered):
        base = 3 * row
        x, y, z = point
        correction_design[base, 0] = 1.0
        correction_design[base, 3:6] = [x, y, z]
        correction_design[base + 1, 1] = 1.0
        correction_design[base + 1, 6:9] = [x, y, z]
        correction_design[base + 2, 2] = 1.0
        correction_design[base + 2, 9:12] = [x, y, z]
    correction, _, _, _ = np.linalg.lstsq(correction_design, correction_observation, rcond=None)
    centroid_correction = correction[:3]
    quasi_rotation = quasi_rotation + correction[3:].reshape(3, 3)
    translation = target_centroid - source_centroid @ quasi_rotation.T + centroid_correction
    transformed = source @ quasi_rotation.T + translation
    residuals = transformed - target
    rank_a = int(np.linalg.matrix_rank(correction_design))
    dof = 3 * n_points - rank_a
    sigma0_sq = float(np.sum(residuals**2) / dof) if dof > 0 else np.nan
    orthogonality_matrix = np.eye(3) - quasi_rotation.T @ quasi_rotation
    elapsed = (datetime.now() - started).total_seconds()
    stats = {
        'rank_a': rank_a, 'dof': dof, 'degrees_of_freedom': dof,
        'variance_estimable': bool(dof > 0 and rank_a == 12),
        "sigma0_sq": sigma0_sq,
        "rmse": float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))),
        "max_residual": float(np.max(np.abs(residuals))),
        "distance_design_matrix": d_source_used,
        "distance_observation_vector": d_target_used,
        "distance_residuals": distance_residuals,
        "distance_residuals_all": distance_residuals_all,
        "distance_sigma_D": sigma_d,
        "distance_pair_mask": pair_mask,
        "scale_pairs_removed": int(np.count_nonzero(~pair_mask)),
        "source_scaled": source_scaled,
        "source_centroid_scaled": source_centroid_scaled,
        "target_centroid": target_centroid,
        "source_centered_scaled": source_centered,
        "target_centered": target_centered,
        "initial_centroid_residuals": initial_centroid_residuals,
        "qst_correction_design_matrix": correction_design,
        "qst_correction_vector": correction,
        "qst_centroid_correction": centroid_correction,
        "quasi_rotation_matrix": quasi_rotation,
        "orthogonality_matrix": orthogonality_matrix,
        "orthogonality_error": float(np.linalg.norm(orthogonality_matrix)),
        "zero_deviation_max_abs": float(np.max(np.abs(residuals))),
        "fixed_scale_applied": True,
        "fixed_scale_value": 1.0,
    }
    return scale, quasi_rotation, translation, stats, elapsed


def compute_distance_design(source, target):
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


def compute_schut_quaternion_matrix(source_c, target_c, scale):
    m = (source_c * scale).T @ target_c
    n = np.zeros((4, 4))
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
    quaternion = quaternion / np.linalg.norm(quaternion)
    return m, n, eigenvalues, eigenvectors, quaternion


def compute_design_matrix_numeric(source, scale, rotation, translation):
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


def compute_audit_outputs(source, target, scale, rotation, translation, method_key):
    mean_source = np.mean(source, axis=0)
    mean_target = np.mean(target, axis=0)
    source_centered = source - mean_source
    target_centered = target - mean_target
    cross_covariance_h = source_centered.T @ target_centered
    svd_u, svd_s, svd_vt = np.linalg.svd(cross_covariance_h)
    transformed = scale * (source @ rotation.T) + translation
    residuals = transformed - target
    residual_norms = np.linalg.norm(residuals, axis=1)
    dof = max(1, 3 * len(source) - 7)
    sigma0_sq = float(np.sum(residual_norms ** 2) / dof)
    observation_covariance = np.eye(3 * len(source)) * sigma0_sq
    weight_matrix = np.linalg.pinv(observation_covariance)
    residual_covariance = (
        np.cov(residuals.T) if len(source) > 1 else np.zeros((3, 3))
    )

    pair_i, pair_j, d_source, d_target = compute_distance_design(source, target)
    distance_design_matrix = d_source
    distance_observation_vector = d_target
    distance_residuals = scale * d_source - d_target
    distance_scale = float((np.linalg.pinv(d_source.T @ d_source) @ d_source.T @ d_target)[0, 0]) if d_source.size else np.nan

    quasi_rotation = target_centered.T @ source_centered @ np.linalg.pinv(source_centered.T @ source_centered)
    qr_u, qr_s, qr_vt = np.linalg.svd(quasi_rotation)
    schut_m, schut_n, schut_evals, schut_evecs, schut_q = compute_schut_quaternion_matrix(
        source_centered, target_centered, scale
    )
    design_matrix = compute_design_matrix_numeric(source, scale, rotation, translation)
    normal_matrix = design_matrix.T @ design_matrix
    covariance_approx = sigma0_sq * np.linalg.pinv(normal_matrix)

    return {
        "input_source_coordinates": source,
        "input_target_coordinates": target,
        "input_n_points": len(source),
        "input_method_key": method_key,
        "scale": scale,
        "rotation_matrix": rotation,
        "translation": translation,
        "transformed_control_points": transformed,
        "residuals_xyz": residuals,
        "residual_norms": residual_norms,
        "source_centroid": mean_source,
        "target_centroid": mean_target,
        "source_centered": source_centered,
        "target_centered": target_centered,
        "cross_covariance_h": cross_covariance_h,
        "svd_u": svd_u,
        "svd_singular_values": svd_s,
        "svd_vt": svd_vt,
        "rotation_determinant": float(np.linalg.det(rotation)),
        "orthogonality_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
        "distance_pair_i": pair_i,
        "distance_pair_j": pair_j,
        "distance_design_matrix": distance_design_matrix,
        "distance_observation_vector": distance_observation_vector,
        "distance_residuals": distance_residuals,
        "distance_scale_estimate": distance_scale,
        "quasi_rotation_matrix": quasi_rotation,
        "quasi_rotation_svd_u": qr_u,
        "quasi_rotation_svd_singular_values": qr_s,
        "quasi_rotation_svd_vt": qr_vt,
        "schut_cross_covariance_m": schut_m,
        "schut_quaternion_matrix_n": schut_n,
        "schut_eigenvalues": schut_evals,
        "schut_eigenvectors": schut_evecs,
        "schut_quaternion": schut_q,
        "numeric_design_matrix": design_matrix,
        "numeric_normal_matrix": normal_matrix,
        "numeric_covariance_approx": covariance_approx,
        "numeric_correlation_approx": create_correlation_matrix(covariance_approx),
        "observation_covariance_approx": observation_covariance,
        "weight_matrix_approx": weight_matrix,
        "residual_covariance_approx": residual_covariance,
        "residual_sum_squares": float(np.sum(residual_norms ** 2)),
        "degrees_of_freedom": dof,
        "audit_sigma0_sq": sigma0_sq,
        "audit_sigma0": float(np.sqrt(sigma0_sq)),
    }


def propagate_similarity_covariance(point_local, scale, rotation, translation, covariance):
    """Propagate 13-parameter similarity covariance to one transformed point.

    Parameter order follows compute_design_matrix_numeric:
    scale, nine row-major rotation coefficients, tx, ty, tz.
    """
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (13, 13) or not np.all(np.isfinite(cov)):
        return None
    point = np.asarray(point_local, dtype=float)
    jac = np.zeros((3, 13), dtype=float)
    jac[:, 0] = point @ rotation.T
    for row in range(3):
        for col in range(3):
            jac[row, 1 + row * 3 + col] = scale * point[col]
    jac[:, 10:13] = np.eye(3)
    point_cov = jac @ cov @ jac.T
    return 0.5 * (point_cov + point_cov.T)


def solve_pair(method_key, method_class, source, target, args):
    kwargs = method_kwargs(method_class, args)
    started = datetime.now()
    canonical_key = method_code(method_class)

    forced_outputs = {}
    if canonical_key == "TFS":
        scale, rotation, translation, stats, elapsed = method_class.solve(
            source, target, **kwargs
        )
    elif args.fixed_scale and canonical_key == "Q-ST":
        scale, rotation, translation, stats, elapsed = solve_qst_fixed_scale(source, target)
    elif args.fixed_scale:
        scale, rotation, translation, elapsed, forced_outputs = solve_fixed_scale(source, target)
        stats = {}
    elif getattr(method_class, "is_global_lm", False):
        result = method_class.solve([source], target, **kwargs)
        transforms, sigma0_sq, cov, residuals, elapsed = result[:5]
        scale, rotation, translation = transforms[0]
        extra = result[5] if len(result) > 5 and isinstance(result[5], dict) else {}
        stats = {
            "sigma0_sq": sigma0_sq,
            "sigma0": math.sqrt(sigma0_sq) if sigma0_sq >= 0 else None,
            "covariance": cov,
            "observation_residuals": residuals,
            **extra,
        }
    elif canonical_key == "CMTLS":
        scale, rotation, translation, q_xi, stats, elapsed = method_class.solve(
            source, target, **kwargs
        )
        stats = {"Q_xi": q_xi, **stats}
    elif canonical_key == "Q-ST":
        scale, rotation, translation, stats, elapsed = method_class.solve(
            source, target
        )
    else:
        scale, rotation, translation, elapsed = method_class.solve(source, target)
        stats = {}

    active_mask = None
    masks = stats.get("active_observation_masks")
    if isinstance(masks, list) and masks:
        try:
            active_mask = np.asarray(masks[0], dtype=bool)
        except Exception:
            active_mask = None
    if active_mask is None or active_mask.shape[0] != len(source):
        active_mask = np.ones(len(source), dtype=bool)
    fit_source = source[active_mask]
    fit_target = target[active_mask]

    transformed = scale * (fit_source @ rotation.T) + translation
    residuals_xyz = transformed - fit_target
    residual_norms = np.linalg.norm(residuals_xyz, axis=1)
    rmse = evaluate_rmse(fit_source, fit_target, scale, rotation, translation)

    outputs = {
        "scale": scale,
        "rotation_matrix": rotation,
        "translation": translation,
        "active_observation_mask": active_mask,
        "transformed_control_points": transformed,
        "residuals_xyz": residuals_xyz,
        "residual_norms": residual_norms,
        "rmse": rmse,
        "mean_residual": float(np.mean(residual_norms)),
        "std_residual": float(np.std(residual_norms)),
        "max_residual": float(np.max(residual_norms)),
        "elapsed_time_solver": elapsed,
        "elapsed_time_total": (datetime.now() - started).total_seconds(),
        **stats,
        **forced_outputs,
    }
    outputs = standardize_adjustment_outputs(method_class, outputs)
    outputs.update(compute_audit_outputs(fit_source, fit_target, scale, rotation, translation, method_key))
    if outputs.get("covariance") is None:
        for covariance_key in ("numeric_covariance_approx", "Q_xi"):
            if outputs.get(covariance_key) is not None:
                outputs["covariance"] = outputs[covariance_key]
                outputs["covariance_source"] = covariance_key
                break
    if outputs.get("covariance_param_names") is None and outputs.get("covariance") is not None:
        outputs["covariance_param_names"] = adjustment_parameter_names(outputs["covariance"])
    if canonical_key in {"Q-ST", "stepien_2020_qst"}:
        # Preserve the article Q-ST QR matrix and distance-screening diagnostics;
        # the generic audit helper also computes a Procrustes-style diagnostic
        # quasi matrix, which is not the transformation used by Q-ST.
        outputs.update(stats)
    if canonical_key in {'SVD-P', 'SCH-Q', 'Q-ST', 'TFS'}:
        outputs.update(iterations=0, solution_type='direct', converged=None,
                       stop_reason='DIRECT_SOLUTION', iteration_history=[])
    if "covariance" in outputs and outputs["covariance"] is not None:
        try:
            outputs["correlation_matrix"] = create_correlation_matrix(
                np.asarray(outputs["covariance"])
            )
        except Exception:
            pass
    return outputs


def common_control_points(source_station, reference_station, selected_control_points=None):
    common = [pid for pid in source_station["all_ids"] if pid in set(reference_station["all_ids"])]
    icp_ids = {str(point_id) for point_id in source_station.get("icp_ids", [])}
    icp_ids.update(str(point_id) for point_id in reference_station.get("icp_ids", []))
    common = [pid for pid in common if str(pid) not in icp_ids]
    if selected_control_points:
        selected = set(selected_control_points)
        common = [cp for cp in common if cp in selected]
    return common


def run_sequential_method(method_key, method_class, stations, args, excluded_points=None, run_label="initial"):
    excluded_points = excluded_points or {}
    registered = []
    registered.append({
        "name": stations[0]["name"],
        "number": stations[0]["number"],
        "coords": stations[0]["coords"].copy(),
        "all_ids": stations[0]["all_ids"].copy(),
        "ctrl_ids": stations[0]["ctrl_ids"].copy(),
        "icp_ids": stations[0].get("icp_ids", []).copy(),
        "point_codes": dict(stations[0].get("point_codes", {})),
        "point_comments": dict(stations[0].get("point_comments", {})),
        "is_reference": True,
    })
    registered_dict = {stations[0]["name"]: registered[0]}
    station_rows = []
    residual_rows = []
    independent_rows = []
    full_outputs = {
        stations[0]["name"]: {
            "coords": stations[0]["coords"].copy(),
            "coords_original": stations[0]["coords"].copy(),
            "all_ids": stations[0]["all_ids"].copy(),
            "ctrl_ids": stations[0]["ctrl_ids"].copy(),
            "icp_ids": stations[0].get("icp_ids", []).copy(),
            "point_codes": dict(stations[0].get("point_codes", {})),
            "point_comments": dict(stations[0].get("point_comments", {})),
            "is_reference": True,
        }
    }

    def failure_row(source_station, error, n_points=0, excluded_for_station=None, attempt_count=0):
        return {
            "method": method_key,
            "mode": "sequential",
            "run_label": run_label,
            "station": source_station["name"],
            "success": False,
            "excluded_outliers": ",".join(sorted(excluded_for_station or [])),
            "n_control_points": n_points,
            "attempt_count": attempt_count,
            "error": error,
        }

    pending = list(stations[1:])
    attempt_counts = defaultdict(int)
    last_failures = {}
    pass_number = 1

    while pending:
        progress = False
        next_pending = []

        for source_station in pending:
            attempt_counts[source_station["name"]] += 1
            reference = build_accumulated_reference(registered_dict)
            cps = common_control_points(source_station, reference, args.control_points)
            station_name = source_station["name"]
            excluded_for_station = set(excluded_points.get((method_key, station_name), []))
            excluded_for_station.update(excluded_points.get(("*", station_name), []))
            if excluded_for_station:
                cps = [cp for cp in cps if cp not in excluded_for_station]
            if len(cps) < args.min_common_cp:
                last_failures[source_station["name"]] = failure_row(
                    source_station,
                    f"Only {len(cps)} common adjusted points",
                    len(cps),
                    excluded_for_station,
                    attempt_counts[source_station["name"]],
                )
                next_pending.append(source_station)
                continue

            source = get_coords_for_cps(source_station, cps)
            target = get_coords_for_cps(reference, cps)
            valid = ~(np.isnan(source).any(axis=1) | np.isnan(target).any(axis=1))
            source_valid = source[valid]
            target_valid = target[valid]
            cp_valid = [cp for cp, ok in zip(cps, valid) if ok]
            if len(source_valid) < args.min_common_cp:
                last_failures[source_station["name"]] = failure_row(
                    source_station,
                    f"Only {len(source_valid)} valid common adjusted points",
                    len(source_valid),
                    excluded_for_station,
                    attempt_counts[source_station["name"]],
                )
                next_pending.append(source_station)
                continue

            try:
                outputs = solve_pair(method_key, method_class, source_valid, target_valid, args)
                scale = outputs["scale"]
                rotation = outputs["rotation_matrix"]
                translation = outputs["translation"]
                active_mask = np.asarray(
                    outputs.get("active_observation_mask", np.ones(len(cp_valid), dtype=bool)),
                    dtype=bool,
                )
                if active_mask.shape[0] != len(cp_valid):
                    active_mask = np.ones(len(cp_valid), dtype=bool)
                cp_fit = [cp for cp, keep in zip(cp_valid, active_mask) if keep]
                target_fit = target_valid[active_mask]
                internally_removed = {
                    str(cp) for cp, keep in zip(cp_valid, active_mask) if not keep
                }
                transformed_all = scale * (source_station["coords"] @ rotation.T) + translation
                icp_rows_for_station = []
                icp_ids = [
                    point_id for point_id in source_station.get("icp_ids", [])
                    if point_id in source_station["all_ids"] and point_id in reference["all_ids"]
                ]
                for point_id in icp_ids:
                    source_index = source_station["all_ids"].index(point_id)
                    reference_index = reference["all_ids"].index(point_id)
                    adjusted = transformed_all[source_index]
                    observed = reference["coords"][reference_index]
                    residual = adjusted - observed
                    norm = float(np.linalg.norm(residual))
                    row = {
                        "method": method_key,
                        "mode": "sequential",
                        "run_label": run_label,
                        "station": source_station["name"],
                        "point": point_id,
                        "observed_x": observed[0],
                        "observed_y": observed[1],
                        "observed_z": observed[2],
                        "adjusted_x": adjusted[0],
                        "adjusted_y": adjusted[1],
                        "adjusted_z": adjusted[2],
                        "vx": residual[0],
                        "vy": residual[1],
                        "vz": residual[2],
                        "v_norm": norm,
                        "is_control_point": False,
                        "role": "ICP",
                    }
                    icp_rows_for_station.append(row)
                    independent_rows.append(row)
                icp_norms = np.asarray([row["v_norm"] for row in icp_rows_for_station], dtype=float)
                icp_rmse = float(np.sqrt(np.mean(icp_norms ** 2))) if icp_norms.size else np.nan
                icp_max = float(np.max(icp_norms)) if icp_norms.size else np.nan
                point_uncertainties = {}
                cov_for_points = outputs.get("numeric_covariance_approx")
                if cov_for_points is not None:
                    for pt_id, point_local in zip(source_station["all_ids"], source_station["coords"]):
                        point_cov = propagate_similarity_covariance(
                            point_local, scale, rotation, translation, cov_for_points
                        )
                        if point_cov is not None:
                            point_uncertainties[pt_id] = point_cov
                rotation_angles = rotation_to_omega_phi_kappa(rotation)

                station_rows.append({
                    "method": method_key,
                    "mode": "sequential",
                    "run_label": run_label,
                    "station": source_station["name"],
                    "success": True,
                    "fixed_scale": bool(args.fixed_scale),
                    "excluded_outliers": ",".join(sorted(set(excluded_for_station) | internally_removed)),
                    "n_control_points": len(cp_fit),
                    "n_icp_points": len(icp_rows_for_station),
                    "icp_rmse": icp_rmse,
                    "icp_max_residual": icp_max,
                    "n_common_adjusted_points": len(cp_fit),
                    "attempt_count": attempt_counts[source_station["name"]],
                    "registration_pass": pass_number,
                    "reference_used": "accumulated_all_adjusted_points",
                    "rmse": outputs["rmse"],
                    "mean_residual": outputs["mean_residual"],
                    "std_residual": outputs["std_residual"],
                    "max_residual": outputs["max_residual"],
                    "scale": scale,
                    "tx": translation[0],
                    "ty": translation[1],
                    "tz": translation[2],
                    "rotation_determinant": outputs.get("rotation_determinant"),
                    "orthogonality_error": outputs.get("orthogonality_error"),
                    "omega_deg": rotation_angles[0],
                    "phi_deg": rotation_angles[1],
                    "kappa_deg": rotation_angles[2],
                    "elapsed_time_solver": outputs["elapsed_time_solver"],
                    "elapsed_time_total": outputs["elapsed_time_total"],
                    "sigma0": outputs.get("sigma0"),
                    "sigma0_sq": outputs.get("sigma0_sq"),
                    "iterations": outputs.get("iterations"),
                    "dof": outputs.get("dof"),
                })
                for cp, observed, adjusted, residual, norm in zip(
                    cp_fit,
                    target_fit,
                    outputs["transformed_control_points"],
                    outputs["residuals_xyz"],
                    outputs["residual_norms"],
                ):
                    residual_rows.append({
                        "method": method_key,
                        "mode": "sequential",
                        "run_label": run_label,
                        "station": source_station["name"],
                        "point": cp,
                        "observed_x": observed[0],
                        "observed_y": observed[1],
                        "observed_z": observed[2],
                        "adjusted_x": adjusted[0],
                        "adjusted_y": adjusted[1],
                        "adjusted_z": adjusted[2],
                        "vx": residual[0],
                        "vy": residual[1],
                        "vz": residual[2],
                        "v_norm": norm,
                        "is_control_point": True,
                        "role": "CP",
                    })

                excluded_for_station_str = {str(pid) for pid in excluded_for_station}
                excluded_for_station_str.update(internally_removed)
                reference_keep = np.asarray([
                    str(pid) not in excluded_for_station_str
                    for pid in source_station["all_ids"]
                ], dtype=bool)
                registered_ids = [
                    pid for pid, keep in zip(source_station["all_ids"], reference_keep) if keep
                ]
                registered_ctrl_ids = [
                    pid for pid in source_station["ctrl_ids"]
                    if str(pid) not in excluded_for_station_str
                ]

                registered_entry = {
                    "name": source_station["name"],
                    "number": source_station["number"],
                    "coords": transformed_all[reference_keep].copy(),
                    "all_ids": registered_ids,
                    "ctrl_ids": registered_ctrl_ids,
                    "icp_ids": [
                        pid for pid in source_station.get("icp_ids", [])
                        if str(pid) not in excluded_for_station_str
                    ],
                    "point_codes": {
                        pid: source_station.get("point_codes", {}).get(pid, "")
                        for pid in registered_ids
                    },
                    "point_comments": {
                        pid: source_station.get("point_comments", {}).get(pid, "")
                        for pid in registered_ids
                        if source_station.get("point_comments", {}).get(pid, "")
                    },
                    "is_reference": False,
                }
                registered.append(registered_entry)
                registered_dict[source_station["name"]] = registered_entry
                full_outputs[source_station["name"]] = outputs
                full_outputs[source_station["name"]]["coords"] = transformed_all[reference_keep].copy()
                full_outputs[source_station["name"]]["coords_original"] = source_station["coords"][reference_keep].copy()
                full_outputs[source_station["name"]]["all_ids"] = registered_ids.copy()
                full_outputs[source_station["name"]]["ctrl_ids"] = registered_ctrl_ids.copy()
                full_outputs[source_station["name"]]["icp_ids"] = registered_entry["icp_ids"].copy()
                full_outputs[source_station["name"]]["point_codes"] = dict(registered_entry["point_codes"])
                full_outputs[source_station["name"]]["point_comments"] = dict(registered_entry["point_comments"])
                full_outputs[source_station["name"]]["excluded_outliers"] = sorted(excluded_for_station_str)
                full_outputs[source_station["name"]]["point_uncertainties_approx"] = {
                    pid: point_uncertainties[pid]
                    for pid in registered_ids
                    if pid in point_uncertainties
                }
                full_outputs[source_station["name"]]["independent_check_rows"] = icp_rows_for_station
                full_outputs[source_station["name"]]["icp_rmse"] = icp_rmse
                full_outputs[source_station["name"]]["icp_max_residual"] = icp_max
                full_outputs[source_station["name"]]["attempt_count"] = attempt_counts[source_station["name"]]
                full_outputs[source_station["name"]]["registration_pass"] = pass_number
                last_failures.pop(source_station["name"], None)
                progress = True
            except Exception as exc:
                last_failures[source_station["name"]] = failure_row(
                    source_station,
                    repr(exc),
                    len(source_valid),
                    excluded_for_station,
                    attempt_counts[source_station["name"]],
                )
                next_pending.append(source_station)

        if not progress:
            for station in next_pending:
                station_rows.append(last_failures.get(
                    station["name"],
                    failure_row(
                        station,
                        "Registration did not converge after retry passes",
                        0,
                        set(),
                        attempt_counts[station["name"]],
                    ),
                ))
            break

        pending = next_pending
        pass_number += 1

    return {
        "method": method_key,
        "mode": "sequential",
        "station_rows": station_rows,
        "residual_rows": residual_rows,
        "independent_rows": independent_rows,
        "outputs": full_outputs,
    }


def filter_station_outlier_points(station, excluded_ids):
    excluded_ids = {str(point_id) for point_id in (excluded_ids or [])}
    if not excluded_ids:
        return station

    all_ids = list(station.get("all_ids", []))
    keep_mask = np.asarray([
        str(point_id) not in excluded_ids for point_id in all_ids
    ], dtype=bool)
    excluded_existing = {
        str(point_id) for point_id, keep in zip(all_ids, keep_mask) if not keep
    }
    if not excluded_existing:
        return station

    filtered = dict(station)
    filtered["all_ids"] = [
        point_id for point_id, keep in zip(all_ids, keep_mask) if keep
    ]
    filtered["ctrl_ids"] = [
        point_id for point_id in station.get("ctrl_ids", [])
        if str(point_id) not in excluded_existing
    ]
    filtered["icp_ids"] = [
        point_id for point_id in station.get("icp_ids", [])
        if str(point_id) not in excluded_existing
    ]
    for coord_key in ("coords", "raw_coords", "coords_original", "original_coords"):
        coords = station.get(coord_key)
        if coords is None:
            continue
        arr = np.asarray(coords)
        if arr.ndim >= 1 and len(arr) == len(all_ids):
            filtered[coord_key] = arr[keep_mask].copy()
    filtered["outlier_excluded_ids"] = sorted(excluded_existing)
    return filtered


def stations_with_method_exclusions(stations, exclusions, method_key):
    exclusions = exclusions or {}
    filtered_stations = []
    excluded_by_station = {}
    for station in stations:
        station_name = station.get("name")
        excluded = set(exclusions.get((method_key, station_name), []))
        excluded.update(exclusions.get(("*", station_name), []))
        excluded_by_station[station_name] = {str(point_id) for point_id in excluded}
        filtered_stations.append(filter_station_outlier_points(station, excluded))
    return filtered_stations, excluded_by_station


def parse_manual_exclusion_specs(entries):
    """Parse GUI/CLI exclusions as STATION:POINT or METHOD:STATION:POINT."""
    exclusions = defaultdict(set)
    invalid = []
    for raw in entries or []:
        text = str(raw).strip()
        if not text:
            continue
        for sep in ("/", ","):
            text = text.replace(sep, ":")
        parts = [part.strip() for part in text.split(":") if part.strip()]
        if len(parts) == 2:
            station, point = parts
            exclusions[("*", station)].add(point)
        elif len(parts) >= 3:
            method, station, point = parts[:3]
            exclusions[(method, station)].add(point)
        else:
            invalid.append(str(raw))
    if invalid:
        raise ValueError(
            "Invalid --manual-exclusions entries. Use STATION:POINT or "
            f"METHOD:STATION:POINT. Invalid: {', '.join(invalid)}"
        )
    return exclusions


def network_baarda_lookup(result, args):
    if "baarda" not in active_outlier_methods(args):
        return {}, {}
    A = result.get("coupled_design_matrix")
    residuals = result.get("observation_residuals")
    metadata = result.get("coordinate_metadata", [])
    try:
        A = np.asarray(A, dtype=float)
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
    except Exception:
        A = np.array([])
        residuals = np.array([])
    if A.ndim != 2 or A.size == 0 or residuals.size == 0:
        status = {
            "baarda_available": False,
            "baarda_unavailable_reason": "MISSING_STOCHASTIC_MODEL",
        }
        return {}, status

    station_names = [
        station.get("station", station.get("name", f"station_{idx + 1}"))
        for idx, station in enumerate(result.get("stations", []))
    ]
    component_names = ("X", "Y", "Z")
    observation_metadata = []
    for idx in range(A.shape[0]):
        meta = dict(metadata[idx]) if idx < len(metadata) else {}
        station_index = int(meta.get("station_index", 0))
        component_index = int(meta.get("component", idx % 3))
        station_id = station_names[station_index] if station_index < len(station_names) else str(station_index)
        component = component_names[component_index] if 0 <= component_index < 3 else str(component_index)
        observation_metadata.append({
            **meta,
            "station_id": station_id,
            "point_id": str(meta.get("point_id", "")),
            "component": component,
            "is_control_point": True,
        })

    result_baarda = baarda_data_snooping(
        A=A,
        L=result.get("observation_vector"),
        P=np.eye(A.shape[0]),
        residuals=residuals,
        observation_metadata=observation_metadata,
        threshold=float(getattr(args, "outlier_sigma", 3.0)),
        use_posterior_sigma=True,
    )
    status = {
        "baarda_available": bool(result_baarda.get("baarda_available", result_baarda.get("can_be_applied", False))),
        "baarda_unavailable_reason": result_baarda.get("baarda_unavailable_reason", ""),
        "rank_a": result_baarda.get("rank_a"),
        "degrees_of_freedom": result_baarda.get("degrees_of_freedom"),
        "condition_number": result_baarda.get("condition_number"),
    }
    lookup = {}
    if not status["baarda_available"]:
        return lookup, status

    for diag in result_baarda.get("observation_diagnostics", []):
        station_id = str(diag.get("station_id", ""))
        point_id = str(diag.get("point_id", ""))
        component = str(diag.get("component", "")).upper()
        if not station_id or not point_id:
            continue
        key = (station_id, point_id)
        entry = lookup.setdefault(key, {
            **status,
            "baarda_score": np.nan,
            "flag_baarda": False,
            "w_x": np.nan,
            "w_y": np.nan,
            "w_z": np.nan,
            "critical_component": "",
        })
        w_value = finite_float(diag.get("standardized_residual_w"))
        if component in {"X", "Y", "Z"}:
            entry[f"w_{component.lower()}"] = w_value if w_value is not None else np.nan
        abs_w = abs(w_value) if w_value is not None else np.nan
        if np.isfinite(abs_w) and (not np.isfinite(entry["baarda_score"]) or abs_w > entry["baarda_score"]):
            entry["baarda_score"] = float(abs_w)
            entry["critical_component"] = component
        entry["flag_baarda"] = bool(entry["flag_baarda"] or diag.get("detected", False))
    return lookup, status


def run_network_method(method_key, method_class, stations, args, excluded_points=None, run_label="network"):
    station_rows = []
    residual_rows = []
    independent_rows = []
    outputs = {}
    working_stations, excluded_by_station = stations_with_method_exclusions(
        stations,
        excluded_points,
        method_key,
    )
    try:
        if run_network_adjustment is None:
            result = run_network_adjustment_local(method_key, method_class, working_stations)
            result["network_import_fallback_reason"] = NETWORK_IMPORT_ERROR
        else:
            result = run_network_adjustment(
                working_stations,
                method_class,
                args.control_points,
                use_baarda_data_snooping=False,
                baarda_threshold=args.outlier_sigma,
                baarda_max_iterations=args.outlier_removal_iterations,
                baarda_min_observations_per_point=args.min_common_cp,
                baarda_remove_strategy="none",
                max_iter=args.max_iter, tol=args.tol,
                weight_constraint=args.weight_constraint,
                linear_solver=getattr(args, 'network_linear_solver', 'legacy_normal'),
                initial_network_result=getattr(args, 'initial_network_result', None),
            )
        if not result:
            raise RuntimeError("Network adjustment returned no result")
        rmses = [
            finite_float(station.get("rmse"))
            for station in result.get("stations", [])
            if finite_float(station.get("rmse")) is not None
        ]
        if rmses:
            result["rmse"] = float(np.mean(rmses))
            result["mean_rmse"] = float(np.mean(rmses))
            result["max_rmse"] = float(np.max(rmses))
        coupled_cov = result.get("coupled_covariance")
        coupled_labels = result.get("coupled_parameter_names", [])
        if coupled_cov is not None and np.asarray(coupled_cov).size:
            coupled_cov = np.asarray(coupled_cov, dtype=float)
            result["network_coupled_covariance"] = coupled_cov
            result["network_coupled_correlation_matrix"] = create_correlation_matrix(coupled_cov)
            result["network_coupled_covariance_param_names"] = list(coupled_labels)
            result["network_coupled_covariance_note"] = (
                "Covariance from the fully coupled N-CUQ-LM adjustment with "
                "shared point coordinates included as unknowns."
            )
        else:
            coupled_cov, coupled_labels, coupled_info = network_coupled_covariance(result)
            if coupled_cov is not None:
                result["network_coupled_covariance"] = coupled_cov
                result["network_coupled_correlation_matrix"] = create_correlation_matrix(coupled_cov)
                result["network_coupled_covariance_param_names"] = coupled_labels
                result["network_coupled_covariance_note"] = coupled_info
        if result.get("covariance") is not None:
            param_names = network_parameter_names(result)
            if param_names:
                result["covariance_param_names"] = param_names
        if "network_audit_outputs" not in result:
            result["network_audit_outputs"] = build_network_audit_outputs(result, method_key)
        baarda_lookup, baarda_status = network_baarda_lookup(result, args)
        output_key = "network" if run_label == "network" else f"network_{run_label}"
        outputs[output_key] = result
        for station in result.get("stations", []):
            station_name = station.get("station")
            excluded_for_station = excluded_by_station.get(station_name, set())
            internal_excluded = {str(pid) for pid in station.get("excluded_outliers", [])}
            combined_excluded = set(excluded_for_station) | internal_excluded
            trans = np.asarray(station.get("translation", [np.nan] * 3), dtype=float)
            rotation = station.get("rotation")
            if rotation is None:
                rotation = station.get("rotation_matrix")
            rotation = _matrix_or_none(rotation)
            rotation_angles = rotation_to_omega_phi_kappa(rotation) if rotation is not None else np.array([np.nan, np.nan, np.nan])
            rotation_det = float(np.linalg.det(rotation)) if rotation is not None else np.nan
            orthogonality_error = (
                float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
                if rotation is not None
                else np.nan
            )
            station_rows.append({
                "method": method_key,
                "mode": "network",
                "run_label": run_label,
                "station": station_name,
                "success": station.get("success", True),
                "excluded_outliers": ",".join(sorted(combined_excluded)),
                "n_control_points": station.get("n_points"),
                "rmse": station.get("rmse"),
                "scale": station.get("scale"),
                "tx": trans[0] if trans.size > 0 else np.nan,
                "ty": trans[1] if trans.size > 1 else np.nan,
                "tz": trans[2] if trans.size > 2 else np.nan,
                "rotation_determinant": station.get("rotation_determinant", rotation_det),
                "orthogonality_error": station.get("orthogonality_error", orthogonality_error),
                "omega_deg": rotation_angles[0],
                "phi_deg": rotation_angles[1],
                "kappa_deg": rotation_angles[2],
                "sigma0": result.get("sigma0"),
                "sigma0_sq": result.get("sigma0_sq"),
                "elapsed_time_solver": result.get("time"),
            })
        for item in result.get("network_residual_rows", []):
            item = dict(item)
            item.setdefault("method", method_key)
            item.setdefault("mode", "network")
            item.setdefault("run_label", run_label)
            item.setdefault("is_control_point", True)
            item.setdefault("role", "CP")
            if "point" not in item and "point_id" in item:
                item["point"] = item["point_id"]
            baarda_diag = {
                **baarda_status,
                **baarda_lookup.get((str(item.get("station", "")), str(item.get("point", ""))), {}),
            }
            item.update({key: value for key, value in baarda_diag.items() if key not in item or item.get(key) in ("", None)})
            excluded_for_station = excluded_by_station.get(item.get("station"), set())
            internal_excluded = set()
            for station in result.get("stations", []):
                if station.get("station") == item.get("station"):
                    internal_excluded = {str(pid) for pid in station.get("excluded_outliers", [])}
                    break
            item.setdefault("excluded_outliers", ",".join(sorted(set(excluded_for_station) | internal_excluded)))
            residual_rows.append(item)

        independent_rows.extend(
            network_independent_check_rows(
                method_key, run_label, result.get("transformed_stations", []), working_stations
            )
        )
        icp_by_station = defaultdict(list)
        for row in independent_rows:
            if (
                str(row.get("method")) != str(method_key)
                or str(row.get("run_label")) != str(run_label)
            ):
                continue
            icp_by_station[row.get("station")].append(float(row.get("v_norm", np.nan)))
        for row in station_rows:
            station_name = row.get("station")
            norms = np.asarray(icp_by_station.get(station_name, []), dtype=float)
            norms = norms[np.isfinite(norms)]
            row["n_icp_points"] = int(norms.size)
            row["icp_rmse"] = float(np.sqrt(np.mean(norms ** 2))) if norms.size else np.nan
            row["icp_max_residual"] = float(np.max(norms)) if norms.size else np.nan
    except Exception as exc:
        station_rows.append({
            "method": method_key,
            "mode": "network",
            "run_label": run_label,
            "station": "network",
            "success": False,
            "excluded_outliers": "",
            "error": repr(exc),
        })
    return {
        "method": method_key,
        "mode": "network",
        "station_rows": station_rows,
        "residual_rows": residual_rows,
        "independent_rows": independent_rows,
        "outputs": outputs,
    }


def network_independent_check_rows(method_key, run_label, transformed_stations, original_stations=None):
    """Compute ICP validation rows for network methods from transformed stations."""
    original_by_name = {
        station.get("name", station.get("station")): station
        for station in (original_stations or [])
    }
    point_observations = defaultdict(list)
    for station in transformed_stations or []:
        station_name = station.get("name", station.get("station", ""))
        ids = list(station.get("all_ids", []))
        coords = np.asarray(station.get("coords", []), dtype=float)
        for pid, coord in zip(ids, coords):
            point_observations[str(pid)].append((station_name, np.asarray(coord, dtype=float)))

    rows = []
    for station in transformed_stations or []:
        station_name = station.get("name", station.get("station", ""))
        ids = list(station.get("all_ids", []))
        coords = np.asarray(station.get("coords", []), dtype=float)
        icp_ids = {str(pid) for pid in station.get("icp_ids", [])}
        original = original_by_name.get(station_name, {})
        icp_ids.update(str(pid) for pid in original.get("icp_ids", []))
        if not icp_ids:
            point_codes = station.get("point_codes", {}) or {}
            point_codes.update(original.get("point_codes", {}) or {})
            icp_ids = {
                str(pid) for pid in ids
                if str(point_codes.get(pid, "")).strip().upper() == "ICP"
            }
        for point_id in sorted(icp_ids, key=extract_station_number):
            if point_id not in ids:
                continue
            observations = [
                coord for other_station, coord in point_observations.get(point_id, [])
                if other_station != station_name
            ]
            if not observations:
                continue
            adjusted = coords[ids.index(point_id)]
            observed = np.mean(np.vstack(observations), axis=0)
            residual = adjusted - observed
            norm = float(np.linalg.norm(residual))
            rows.append({
                "method": method_key,
                "mode": "network",
                "run_label": run_label,
                "station": station_name,
                "point": point_id,
                "observed_x": observed[0],
                "observed_y": observed[1],
                "observed_z": observed[2],
                "adjusted_x": adjusted[0],
                "adjusted_y": adjusted[1],
                "adjusted_z": adjusted[2],
                "vx": residual[0],
                "vy": residual[1],
                "vz": residual[2],
                "v_norm": norm,
                "is_control_point": False,
                "role": "ICP",
            })
    return rows


def skew_matrix(vector):
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=float)


def network_coupled_covariance(result):
    """Build an approximate coupled network covariance with shared point coordinates.

    The original network solvers estimate each station transform against a fixed
    global target. This diagnostic covariance re-linearizes the solved network
    with shared control-point coordinates as unknowns, which exposes the
    cross-station correlations expected in a connected network.
    """
    transformed = result.get("transformed_stations", []) or []
    residual_rows = result.get("network_residual_rows", []) or []
    if len(transformed) < 2 or not residual_rows:
        return None, [], "not_available: not enough network observations"

    station_by_name = {
        station.get("name", station.get("station")): station
        for station in transformed
        if station.get("coords_original") is not None and station.get("transform") is not None
    }
    point_ids = sorted({
        str(row.get("point", row.get("point_id")))
        for row in residual_rows
        if row.get("point", row.get("point_id")) is not None
    }, key=extract_station_number)
    if len(station_by_name) < 2 or not point_ids:
        return None, [], "not_available: no shared station/point structure"

    station_names = sorted(station_by_name.keys(), key=extract_station_number)
    datum_station = station_names[0]
    free_stations = [name for name in station_names if name != datum_station]
    station_offsets = {name: idx * 7 for idx, name in enumerate(free_stations)}
    point_offset_base = len(free_stations) * 7
    point_offsets = {
        point_id: point_offset_base + idx * 3
        for idx, point_id in enumerate(point_ids)
    }
    n_params = point_offset_base + len(point_ids) * 3
    if n_params == 0:
        return None, [], "not_available: empty parameter system"

    rows = []
    residual_values = []
    for row in residual_rows:
        station_name = row.get("station")
        point_id = str(row.get("point", row.get("point_id")))
        station = station_by_name.get(station_name)
        if station is None or point_id not in point_offsets:
            continue
        all_ids = list(station.get("all_ids", []))
        if point_id not in [str(pid) for pid in all_ids]:
            continue
        local_index = [str(pid) for pid in all_ids].index(point_id)
        source = np.asarray(station["coords_original"][local_index], dtype=float)
        scale, rotation, _translation = station["transform"]
        rotation = np.asarray(rotation, dtype=float)

        jac = np.zeros((3, n_params), dtype=float)
        if station_name != datum_station:
            offset = station_offsets[station_name]
            rotated = rotation @ source
            jac[:, offset] = rotated
            jac[:, offset + 1:offset + 4] = -float(scale) * rotation @ skew_matrix(source)
            jac[:, offset + 4:offset + 7] = np.eye(3)
        point_offset = point_offsets[point_id]
        jac[:, point_offset:point_offset + 3] = -np.eye(3)
        rows.append(jac)
        residual_values.extend([
            finite_float(row.get("vx")) or 0.0,
            finite_float(row.get("vy")) or 0.0,
            finite_float(row.get("vz")) or 0.0,
        ])

    if not rows:
        return None, [], "not_available: no valid coupled rows"
    design = np.vstack(rows)
    residual_vector = np.asarray(residual_values, dtype=float)
    dof = max(1, design.shape[0] - design.shape[1])
    sigma0_sq = float(np.sum(residual_vector ** 2) / dof)
    normal = design.T @ design
    covariance = sigma0_sq * np.linalg.pinv(normal)

    labels = []
    for station_name in free_stations:
        labels.extend([
            f"{station_name}.scale",
            f"{station_name}.rx",
            f"{station_name}.ry",
            f"{station_name}.rz",
            f"{station_name}.tx",
            f"{station_name}.ty",
            f"{station_name}.tz",
        ])
    for point_id in point_ids:
        labels.extend([f"point_{point_id}.x", f"point_{point_id}.y", f"point_{point_id}.z"])
    note = (
        f"coupled covariance with datum station {datum_station}; "
        f"{len(free_stations)} free stations and {len(point_ids)} shared points"
    )
    return covariance, labels, note


def build_network_audit_outputs(result, method_key):
    audit = {}
    residual_rows = result.get("network_residual_rows", [])
    rows_by_station = defaultdict(list)
    for row in residual_rows:
        rows_by_station[row.get("station", row.get("station_name"))].append(row)
    for station in result.get("transformed_stations", []):
        station_name = station.get("name", station.get("station"))
        rows = rows_by_station.get(station_name, [])
        transform = station.get("transform")
        if not rows or transform is None:
            continue
        try:
            scale, rotation, translation = transform
            all_ids = list(station.get("all_ids", []))
            original = np.asarray(station.get("coords_original", []), dtype=float)
            source = []
            target = []
            for row in rows:
                point_id = row.get("point", row.get("point_id"))
                if point_id not in all_ids:
                    continue
                source.append(original[all_ids.index(point_id)])
                target.append([row["observed_x"], row["observed_y"], row["observed_z"]])
            if len(source) >= 3:
                audit[station_name] = compute_audit_outputs(
                    np.asarray(source, dtype=float),
                    np.asarray(target, dtype=float),
                    scale,
                    np.asarray(rotation, dtype=float),
                    np.asarray(translation, dtype=float),
                    method_key,
                )
        except Exception:
            continue
    return audit


def run_network_adjustment_local(method_key, method_class, stations):
    """Network adjustment fallback that avoids importing the GUI-bound package."""
    from collections import Counter

    cp_counts = Counter()
    for station in stations:
        cp_counts.update(station.get("ctrl_ids", []))
    network_cps = sorted(
        [cp for cp, count in cp_counts.items() if count >= 2],
        key=extract_station_number,
    )
    if not network_cps:
        raise RuntimeError("No control points observed in at least two stations")

    reference = max(stations, key=lambda st: len(set(st.get("ctrl_ids", [])).intersection(network_cps)))
    global_targets = {}
    target_accumulator = defaultdict(list)
    for cp in network_cps:
        if cp in reference["all_ids"]:
            coord = reference["coords"][reference["all_ids"].index(cp)]
            global_targets[cp] = coord
            target_accumulator[cp].append(coord)

    pending = [station for station in stations if station["name"] != reference["name"]]
    transformed_seed = [{
        "name": reference["name"],
        "number": reference["number"],
        "coords": reference["coords"],
        "all_ids": reference["all_ids"],
        "ctrl_ids": reference["ctrl_ids"],
    }]

    progress = True
    while pending and progress:
        progress = False
        next_pending = []
        for station in pending:
            common = [cp for cp in station["ctrl_ids"] if cp in global_targets]
            if len(common) < 4:
                next_pending.append(station)
                continue
            source = get_coords_for_cps(station, common)
            target = np.asarray([global_targets[cp] for cp in common], dtype=float)
            valid = ~(np.isnan(source).any(axis=1) | np.isnan(target).any(axis=1))
            if int(np.sum(valid)) < 4:
                next_pending.append(station)
                continue
            scale, rotation, translation, _ = Method1_SequentialSVD.solve(source[valid], target[valid])
            transformed = scale * (station["coords"] @ rotation.T) + translation
            transformed_seed.append({
                "name": station["name"],
                "number": station["number"],
                "coords": transformed,
                "coords_original": station["coords"],
                "all_ids": station["all_ids"],
                "ctrl_ids": station["ctrl_ids"],
                "transform": (scale, rotation, translation),
            })
            for cp in station["ctrl_ids"]:
                if cp not in network_cps:
                    continue
                try:
                    coord = transformed[station["all_ids"].index(cp)]
                except ValueError:
                    continue
                target_accumulator[cp].append(coord)
                global_targets[cp] = np.mean(target_accumulator[cp], axis=0)
            progress = True
        pending = next_pending

    source_matrices = []
    target_matrices = []
    station_meta = []
    for station in stations:
        common = [cp for cp in station["ctrl_ids"] if cp in global_targets]
        if len(common) < 4:
            continue
        source = get_coords_for_cps(station, common)
        target = np.asarray([global_targets[cp] for cp in common], dtype=float)
        valid = ~(np.isnan(source).any(axis=1) | np.isnan(target).any(axis=1))
        if int(np.sum(valid)) < 4:
            continue
        valid_cps = [cp for cp, ok in zip(common, valid) if ok]
        source_matrices.append(source[valid])
        target_matrices.append(target[valid])
        station_meta.append((station, valid_cps))

    if not source_matrices:
        raise RuntimeError("No station has enough common CPs for local network adjustment")

    started = datetime.now()
    kwargs = method_kwargs(method_class, args)
    solve_res = method_class.solve(source_matrices, target_matrices, **kwargs)
    transforms, sigma0_sq, covariance, residual_vector, elapsed = solve_res[:5]
    matrices = solve_res[5] if len(solve_res) > 5 and isinstance(solve_res[5], dict) else {}
    if elapsed is None:
        elapsed = (datetime.now() - started).total_seconds()

    station_results = []
    transformed_stations = []
    network_residual_rows = []
    all_audit = {}
    active_masks = matrices.get("active_observation_masks", []) if isinstance(matrices, dict) else []

    for idx, ((station, cps), (scale, rotation, translation)) in enumerate(zip(station_meta, transforms)):
        active_mask = None
        if idx < len(active_masks):
            try:
                active_mask = np.asarray(active_masks[idx], dtype=bool)
            except Exception:
                active_mask = None
        if active_mask is None or active_mask.shape[0] != len(source_matrices[idx]) or int(np.sum(active_mask)) < 4:
            active_mask = np.ones(len(source_matrices[idx]), dtype=bool)

        fit_source = source_matrices[idx][active_mask]
        fit_target = target_matrices[idx][active_mask]
        fit_cps = [cp for cp, keep in zip(cps, active_mask) if keep]
        removed_fit_ids = {str(cp) for cp, keep in zip(cps, active_mask) if not keep}

        transformed_all_full = scale * (station["coords"] @ rotation.T) + translation
        full_keep = np.asarray([
            str(point_id) not in removed_fit_ids
            for point_id in station["all_ids"]
        ], dtype=bool)
        transformed_all = transformed_all_full[full_keep]
        coords_original_kept = station["coords"][full_keep]
        all_ids_kept = [
            point_id for point_id, keep in zip(station["all_ids"], full_keep) if keep
        ]
        ctrl_ids_kept = [
            point_id for point_id in station["ctrl_ids"] if str(point_id) not in removed_fit_ids
        ]

        transformed_cp = scale * (fit_source @ rotation.T) + translation
        residuals = transformed_cp - fit_target
        residual_norms = np.linalg.norm(residuals, axis=1)
        rmse = evaluate_rmse(fit_source, fit_target, scale, rotation, translation)

        station_results.append({
            "station": station["name"],
            "number": station["number"],
            "scale": scale,
            "rotation": rotation,
            "translation": translation,
            "rmse": rmse,
            "n_points": len(fit_cps),
            "excluded_outliers": sorted(removed_fit_ids),
            "success": True,
        })
        transformed_stations.append({
            "name": station["name"],
            "number": station["number"],
            "coords": transformed_all,
            "coords_original": coords_original_kept,
            "all_ids": all_ids_kept,
            "ctrl_ids": ctrl_ids_kept,
            "transform": (scale, rotation, translation),
            "rmse": rmse,
            "excluded_outliers": sorted(removed_fit_ids),
        })
        all_audit[station["name"]] = compute_audit_outputs(
            fit_source,
            fit_target,
            scale,
            rotation,
            translation,
            method_key,
        )
        for cp, observed, adjusted, residual, norm in zip(
            fit_cps, fit_target, transformed_cp, residuals, residual_norms
        ):
            network_residual_rows.append({
                "station": station["name"],
                "point": cp,
                "observed_x": observed[0],
                "observed_y": observed[1],
                "observed_z": observed[2],
                "adjusted_x": adjusted[0],
                "adjusted_y": adjusted[1],
                "adjusted_z": adjusted[2],
                "vx": residual[0],
                "vy": residual[1],
                "vz": residual[2],
                "v_norm": norm,
            })

    return {
        "type": "network_adjustment",
        "registration_type": "network",
        "method": method_key,
        "stations": station_results,
        "transformed_stations": transformed_stations,
        "control_points": {
            "ids": list(global_targets.keys()),
            "target_coords": np.asarray(list(global_targets.values()), dtype=float),
        },
        "sigma0": math.sqrt(sigma0_sq) if sigma0_sq >= 0 else np.nan,
        "sigma0_sq": sigma0_sq,
        "covariance": covariance,
        "design_matrix": matrices.get("design_matrix", np.array([])),
        "observation_vector": matrices.get("observation_vector", np.array([])),
        "active_observation_masks": active_masks,
        "iterative_outlier_trace": matrices.get("iterative_outlier_trace", []),
        "iterative_outlier_count": matrices.get("iterative_outlier_count", 0),
        "observation_residuals": residual_vector,
        "time": elapsed,
        "n_stations": len(station_results),
        "n_control_points": len(global_targets),
        "network_residual_rows": network_residual_rows,
        "network_audit_outputs": all_audit,
    }


def selected_methods(args):
    requested = set(args.methods or ["all"])
    sequential = SEQUENTIAL_METHODS
    network = NETWORK_METHODS
    if "all" not in requested:
        sequential = [(key, cls) for key, cls in sequential if key in requested]
        network = [(key, cls) for key, cls in network if key in requested]
    if args.skip_network:
        network = []
    return sequential, network


def iter_output_scopes(method_outputs):
    for station, outputs in method_outputs.items():
        if not isinstance(outputs, dict):
            continue
        yield station, outputs
        audit_outputs = outputs.get("network_audit_outputs")
        if isinstance(audit_outputs, dict):
            for audit_station, audit_values in audit_outputs.items():
                if isinstance(audit_values, dict):
                    yield f"{station}/{audit_station}", audit_values


def matrix_inventory(outputs_by_method):
    rows = []
    matrix_terms = (
        "matrix",
        "covariance",
        "correlation",
        "residuals",
        "vector",
        "Q_",
        "G_",
        "A_",
        "Y_",
        "E_A",
        "xi",
        "constraint",
        "normal",
        "lagrange",
    )
    for method_key, method_outputs in outputs_by_method.items():
        for station, outputs in iter_output_scopes(method_outputs):
            for key, value in outputs.items():
                if not any(term.lower() in key.lower() for term in matrix_terms):
                    continue
                if value is None:
                    continue
                try:
                    summary = array_summary(value)
                except Exception:
                    continue
                if not summary.get("shape"):
                    continue
                rows.append({
                    "method": method_key,
                    "station_or_scope": station,
                    "output": key,
                    "shape": "x".join(map(str, summary.get("shape", []))),
                    "min": summary.get("min"),
                    "max": summary.get("max"),
                    "mean": summary.get("mean"),
                    "norm": summary.get("norm"),
                })
    return rows


def variable_inventory(outputs_by_method):
    rows = []
    for method_key, method_outputs in outputs_by_method.items():
        for station, outputs in iter_output_scopes(method_outputs):
            for key, value in outputs.items():
                if isinstance(value, np.ndarray):
                    shape = "x".join(map(str, value.shape))
                    kind = "array"
                elif isinstance(value, (float, int, np.floating, np.integer)):
                    shape = "scalar"
                    kind = "number"
                else:
                    shape = type(value).__name__
                    kind = "object"
                rows.append({
                    "method": method_key,
                    "station_or_scope": station,
                    "variable": key,
                    "kind": kind,
                    "shape": shape,
                })
    return rows


def safe_key(value):
    text = str(value)
    safe = []
    for ch in text:
        safe.append(ch if ch.isalnum() or ch in ("_", "-") else "_")
    return "".join(safe).strip("_")


def save_full_numeric_outputs(output_dir, outputs_by_method):
    arrays = {}
    scalars = []
    labels = []
    combined_covariances = {}
    for method_key, method_outputs in outputs_by_method.items():
        for station, outputs in iter_output_scopes(method_outputs):
            prefix = f"{safe_key(method_key)}__{safe_key(station)}"
            for key, value in outputs.items():
                name = f"{prefix}__{safe_key(key)}"
                if isinstance(value, np.ndarray):
                    arrays[name] = value
                    param_names = outputs.get(f"{key}_param_names")
                    if param_names is None and key == "network_coupled_covariance":
                        param_names = outputs.get("network_coupled_covariance_param_names")
                    if param_names is None and key == "covariance":
                        param_names = network_parameter_names(outputs)
                    if param_names:
                        for index, label in enumerate(param_names):
                            labels.append({
                                "array_name": name,
                                "method": method_key,
                                "station_or_scope": station,
                                "matrix": key,
                                "index": index,
                                "parameter": label,
                                "description": parameter_description(label),
                            })
                    if key == "covariance":
                        arr = np.asarray(value)
                        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                            combined_covariances.setdefault(method_key, []).append(
                                (station, arr, param_names or adjustment_parameter_names(arr))
                            )
                elif isinstance(value, (float, int, np.floating, np.integer, bool)):
                    scalars.append({
                        "method": method_key,
                        "station_or_scope": station,
                        "variable": key,
                        "value": finite_float(value) if not isinstance(value, bool) else bool(value),
                    })
    for method_key, blocks in combined_covariances.items():
        if len(blocks) <= 1:
            continue
        matrix = block_diagonal_matrix([block[1] for block in blocks])
        name = f"{safe_key(method_key)}__combined__parameter_covariance"
        arrays[name] = matrix
        offset = 0
        for station, block, param_names in blocks:
            if not param_names or len(param_names) != block.shape[0]:
                param_names = adjustment_parameter_names(block)
            for index, label in enumerate(param_names):
                labels.append({
                    "array_name": name,
                    "method": method_key,
                    "station_or_scope": "combined",
                    "matrix": "parameter_covariance",
                    "index": offset + index,
                    "parameter": f"{station}.{label}",
                    "description": parameter_description(label),
                })
            offset += block.shape[0]
    if arrays:
        np.savez_compressed(output_dir / "outputs_full_arrays.npz", **arrays)
    pd.DataFrame(scalars).to_csv(output_dir / "outputs_scalar_values.csv", index=False)
    if labels:
        pd.DataFrame(labels).to_csv(output_dir / "matrix_parameter_labels.csv", index=False)


def network_parameter_names(outputs):
    cov = outputs.get("covariance")
    if cov is None:
        return []
    try:
        size = int(np.asarray(cov).shape[0])
    except Exception:
        return []
    stations = outputs.get("stations") or []
    if not stations or size % len(stations) != 0:
        return []
    block = size // len(stations)
    if block == 8:
        base = ["scale", "q0", "q1", "q2", "q3", "tx", "ty", "tz"]
    elif block == 7:
        base = ["scale", "rx", "ry", "rz", "tx", "ty", "tz"]
    else:
        base = [f"p{i + 1}" for i in range(block)]
    labels = []
    for station in stations:
        station_name = station.get("station", station.get("name", "station"))
        labels.extend([f"{station_name}.{name}" for name in base])
    return labels


def adjustment_parameter_names(covariance):
    try:
        size = int(np.asarray(covariance).shape[0])
    except Exception:
        return []
    if size == 13:
        return [
            "scale",
            "r11", "r12", "r13",
            "r21", "r22", "r23",
            "r31", "r32", "r33",
            "tx", "ty", "tz",
        ]
    if size == 9:
        return ["scale", "rx", "ry", "rz", "tx", "ty", "tz", "aux1", "aux2"]
    if size == 8:
        return ["scale", "q0", "q1", "q2", "q3", "tx", "ty", "tz"]
    if size == 7:
        return ["scale", "rx", "ry", "rz", "tx", "ty", "tz"]
    return [f"p{i + 1}" for i in range(size)]


def block_diagonal_matrix(blocks):
    total = sum(block.shape[0] for block in blocks)
    result = np.zeros((total, total), dtype=float)
    offset = 0
    for block in blocks:
        size = block.shape[0]
        result[offset:offset + size, offset:offset + size] = block
        offset += size
    return result


def parameter_description(label):
    name = str(label).split(".")[-1].lower()
    descriptions = {
        "scale": "station scale factor",
        "q0": "quaternion scalar component",
        "q1": "quaternion X component",
        "q2": "quaternion Y component",
        "q3": "quaternion Z component",
        "r11": "rotation matrix coefficient row 1 column 1",
        "r12": "rotation matrix coefficient row 1 column 2",
        "r13": "rotation matrix coefficient row 1 column 3",
        "r21": "rotation matrix coefficient row 2 column 1",
        "r22": "rotation matrix coefficient row 2 column 2",
        "r23": "rotation matrix coefficient row 2 column 3",
        "r31": "rotation matrix coefficient row 3 column 1",
        "r32": "rotation matrix coefficient row 3 column 2",
        "r33": "rotation matrix coefficient row 3 column 3",
        "rx": "rotation around X axis",
        "ry": "rotation around Y axis",
        "rz": "rotation around Z axis",
        "tx": "translation in X, metres",
        "ty": "translation in Y, metres",
        "tz": "translation in Z, metres",
    }
    return descriptions.get(name, "estimated network parameter")


def write_method_notes(output_dir):
    rows = []
    for method, notes in METHOD_NOTES.items():
        rows.append({"method": method, **notes})
    pd.DataFrame(rows).to_csv(output_dir / "method_inputs_outputs_catalog.csv", index=False)

    lines = ["# Catalogo de metodos, inputs e outputs", ""]
    for method, notes in METHOD_NOTES.items():
        lines.extend([
            f"## {method}",
            f"- Inputs ja implementados: {notes['implemented_inputs']}",
            f"- Outputs ja implementados: {notes['implemented_outputs']}",
            f"- Inputs possiveis de implementar: {notes['possible_inputs']}",
            f"- Outputs possiveis de implementar: {notes['possible_outputs']}",
            "",
        ])
    (output_dir / "method_inputs_outputs_catalog.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_run_configuration(output_dir, args):
    config = {
        "data": args.data,
        "out": args.out,
        "methods": args.methods,
        "selected_plot_method": args.selected_plot_method,
        "control_points": args.control_points,
        "independent_check_points": args.independent_check_points,
        "min_common_cp": args.min_common_cp,
        "fixed_scale": args.fixed_scale,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "weight_constraint": args.weight_constraint,
        "weight_model": "unit weights P=I in the adjustment equations",
        "apriori_sigma_mm": args.apriori_sigma_mm,
        "apriori_sigma_m": args.apriori_sigma_mm / 1000.0,
        "global_variance_test_note": (
            "apriori_sigma is used only to normalize the global chi-square "
            "variance-factor test; it does not change the unit weight matrix."
        ),
        "skip_network": args.skip_network,
        "outlier_methods": args.outlier_methods,
        "active_outlier_detectors": sorted(active_outlier_methods(args)),
        "disabled_outlier_detectors": ["sigma", "raw_3sigma", "percentile", "rms_factor", "old_voting", "component"],
        "outlier_mode": args.outlier_mode,
        "outlier_action": args.outlier_action,
        "outlier_mad": args.outlier_mad,
        "outlier_mad_min_observations": args.outlier_mad_min_observations,
        "outlier_min_group": args.outlier_min_group,
        "outlier_removal_iterations": args.outlier_removal_iterations,
        "outlier_max_remove_per_iteration": args.outlier_max_remove_per_iteration,
        "outlier_max_condition_number": args.outlier_max_condition_number,
        "outlier_numerical_epsilon": args.outlier_numerical_epsilon,
        "outlier_removal_strategy": "sequential strongest station-point candidate, one observation per iteration by default",
        "manual_exclusions": getattr(args, "manual_exclusions", []),
    }
    with (output_dir / "run_configuration.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    pd.DataFrame([
        {"input_parameter": key, "value": json.dumps(value) if isinstance(value, list) else value}
        for key, value in config.items()
    ]).to_csv(output_dir / "run_configuration.csv", index=False)


def _run_label_series(df):
    if df is None or df.empty or "run_label" not in df.columns:
        return pd.Series([""] * (0 if df is None else len(df)), index=None if df is None else df.index, dtype=str)
    return df["run_label"].fillna("").astype(str)


def _initial_solution_mask(df):
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    labels = _run_label_series(df)
    return ~labels.str.startswith("outliers_removed")


def _corrected_solution_mask(df, corrected_run_label):
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    if corrected_run_label:
        return _run_label_series(df).eq(str(corrected_run_label))
    return _initial_solution_mask(df)


def _residual_metrics(df, prefix):
    metrics = {}
    if df is None or df.empty:
        for component in ("X", "Y", "Z", "3D"):
            metrics[f"{prefix} RMSE {component}"] = np.nan
        metrics[f"Maximum {prefix.lower()} residual"] = np.nan
        return metrics
    for component, column in (("X", "vx"), ("Y", "vy"), ("Z", "vz")):
        values = pd.to_numeric(df.get(column), errors="coerce").dropna().to_numpy(dtype=float)
        metrics[f"{prefix} RMSE {component}"] = float(np.sqrt(np.mean(values ** 2))) if values.size else np.nan
    norms = pd.to_numeric(df.get("v_norm"), errors="coerce").dropna().to_numpy(dtype=float)
    metrics[f"{prefix} RMSE 3D"] = float(np.sqrt(np.mean(norms ** 2))) if norms.size else np.nan
    metrics[f"Maximum {prefix.lower()} residual"] = float(np.max(norms)) if norms.size else np.nan
    return metrics


def _solution_statistics(station_df, residual_df, independent_df, excluded_count, status, stop_condition):
    stats = {
        "solution_status": status,
        "stop_condition": stop_condition,
        "Number of control observations": int(len(residual_df)) if residual_df is not None else 0,
        "Number of excluded observations": int(excluded_count),
    }
    if residual_df is not None and not residual_df.empty:
        stats["Degrees of freedom"] = int(max(0, len(residual_df) * 3 - 1))
        if "condition_number_before" in residual_df:
            cond = pd.to_numeric(residual_df["condition_number_before"], errors="coerce").dropna()
            stats["Normal-matrix condition number"] = float(cond.max()) if not cond.empty else np.nan
        else:
            stats["Normal-matrix condition number"] = np.nan
    else:
        stats["Degrees of freedom"] = 0
        stats["Normal-matrix condition number"] = np.nan
    if station_df is not None and not station_df.empty:
        sigma = (
            pd.to_numeric(station_df["sigma0_sq"], errors="coerce").dropna()
            if "sigma0_sq" in station_df.columns
            else pd.Series(dtype=float)
        )
        stats["Posterior variance factor"] = float(sigma.mean()) if not sigma.empty else np.nan
        iterations = (
            pd.to_numeric(station_df["iterations"], errors="coerce").dropna()
            if "iterations" in station_df.columns
            else pd.Series(dtype=float)
        )
        stats["Convergence iterations"] = int(iterations.max()) if not iterations.empty else np.nan
    else:
        stats["Posterior variance factor"] = np.nan
        stats["Convergence iterations"] = np.nan
    stats.update(_residual_metrics(residual_df, "Control"))
    stats.update(_residual_metrics(independent_df, "ICP"))
    return stats


def _comparison_table(initial_stats, corrected_stats):
    rows = []
    keys = [
        "Number of control observations",
        "Number of excluded observations",
        "Degrees of freedom",
        "Posterior variance factor",
        "Control RMSE X",
        "Control RMSE Y",
        "Control RMSE Z",
        "Control RMSE 3D",
        "ICP RMSE X",
        "ICP RMSE Y",
        "ICP RMSE Z",
        "ICP RMSE 3D",
        "Maximum control residual",
        "Maximum icp residual",
        "Normal-matrix condition number",
        "Convergence iterations",
    ]
    for metric in keys:
        initial = initial_stats.get(metric, np.nan)
        corrected = corrected_stats.get(metric, np.nan)
        try:
            absolute = float(corrected) - float(initial)
            relative = (absolute / float(initial) * 100.0) if float(initial) != 0 else np.nan
        except Exception:
            absolute = np.nan
            relative = np.nan
        rows.append({
            "Metric": metric,
            "Initial": initial,
            "Corrected": corrected,
            "Absolute change": absolute,
            "Relative change (%)": relative,
        })
    return pd.DataFrame(rows)


def write_initial_corrected_outlier_package(
    output_dir,
    station_df,
    residual_df,
    independent_df,
    outlier_df,
    outlier_summary_df,
    outlier_trace_df,
    corrected_status,
    stop_condition,
):
    base = Path(output_dir)
    dirs = {
        "initial": base / "initial_solution",
        "corrected": base / "corrected_solution",
        "cycles": base / "exclusion_cycles",
        "diagnostics": base / "diagnostics",
        "comparison": base / "comparison",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    corrected_run_label = ""
    if outlier_trace_df is not None and not outlier_trace_df.empty and "run_label" in outlier_trace_df.columns:
        labels = outlier_trace_df["run_label"].dropna().astype(str)
        corrected_run_label = labels.iloc[-1] if not labels.empty else ""
    excluded_count = int(len(outlier_trace_df)) if outlier_trace_df is not None else 0

    initial_station = station_df[_initial_solution_mask(station_df)].copy() if station_df is not None and not station_df.empty else pd.DataFrame()
    initial_residual = residual_df[_initial_solution_mask(residual_df)].copy() if residual_df is not None and not residual_df.empty else pd.DataFrame()
    initial_icp = independent_df[_initial_solution_mask(independent_df)].copy() if independent_df is not None and not independent_df.empty else pd.DataFrame()

    corrected_station = station_df[_corrected_solution_mask(station_df, corrected_run_label)].copy() if station_df is not None and not station_df.empty else pd.DataFrame()
    corrected_residual = residual_df[_corrected_solution_mask(residual_df, corrected_run_label)].copy() if residual_df is not None and not residual_df.empty else pd.DataFrame()
    corrected_icp = independent_df[_corrected_solution_mask(independent_df, corrected_run_label)].copy() if independent_df is not None and not independent_df.empty else pd.DataFrame()

    initial_residual.to_csv(dirs["initial"] / "initial_control_residuals.csv", index=False)
    initial_icp.to_csv(dirs["initial"] / "initial_icp_residuals.csv", index=False)
    initial_station.to_csv(dirs["initial"] / "initial_transformation_parameters.csv", index=False)
    initial_residual.to_csv(dirs["initial"] / "initial_adjusted_coordinates.csv", index=False)

    corrected_residual.to_csv(dirs["corrected"] / "corrected_control_residuals.csv", index=False)
    corrected_icp.to_csv(dirs["corrected"] / "corrected_icp_residuals.csv", index=False)
    corrected_station.to_csv(dirs["corrected"] / "corrected_transformation_parameters.csv", index=False)
    corrected_residual.to_csv(dirs["corrected"] / "corrected_adjusted_coordinates.csv", index=False)
    corrected_residual.to_csv(dirs["corrected"] / "corrected_observation_set.csv", index=False)

    if outlier_trace_df is None:
        outlier_trace_df = pd.DataFrame()
    outlier_trace_df.to_csv(dirs["cycles"] / "excluded_observations.csv", index=False)
    outlier_trace_df.to_csv(dirs["cycles"] / "exclusion_cycles.csv", index=False)
    if outlier_df is not None:
        outlier_df.to_csv(dirs["diagnostics"] / "outlier_flags.csv", index=False)
    if outlier_summary_df is not None:
        outlier_summary_df.to_csv(dirs["diagnostics"] / "outlier_summary.csv", index=False)

    initial_stats = _solution_statistics(
        initial_station,
        initial_residual,
        initial_icp,
        0,
        "INITIAL_SOLUTION",
        "",
    )
    corrected_stats = _solution_statistics(
        corrected_station,
        corrected_residual,
        corrected_icp,
        excluded_count,
        corrected_status,
        stop_condition,
    )
    with (dirs["initial"] / "initial_statistics.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(initial_stats), f, indent=2)
    with (dirs["corrected"] / "corrected_statistics.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(corrected_stats), f, indent=2)

    comparison = _comparison_table(initial_stats, corrected_stats)
    comparison.to_csv(dirs["comparison"] / "initial_vs_corrected_comparison.csv", index=False)
    report_lines = [
        "Final solution after geometry-preserving observation exclusion",
        f"Status: {corrected_status}",
        f"Stop condition: {stop_condition or 'not_applicable'}",
        f"Excluded station-point observations: {excluded_count}",
        "",
        comparison.to_string(index=False),
    ]
    (dirs["comparison"] / "initial_vs_corrected_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )


def _rms_array(value):
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return np.nan
    if arr.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(arr.ravel() ** 2)))


def _trace_or_nan(value):
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return np.nan
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.size == 0:
        return np.nan
    return float(np.trace(arr))


def _is_number(value):
    try:
        return np.isfinite(float(value))
    except Exception:
        return False


def _value_or_nd(value):
    return float(value) if _is_number(value) else "N/D"


def _metric_status(value, criterion, lower_is_better=True):
    if not _is_number(value):
        return "not_available"
    value = abs(float(value)) if "deviation" in criterion or "orthogonality" in criterion else float(value)
    if lower_is_better:
        return "better" if value <= 1.0 else "worse"
    return "neutral"


def rotation_to_omega_phi_kappa(rotation):
    """Return omega, phi, kappa angles in degrees using Rz(kappa) Ry(phi) Rx(omega)."""
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3):
        return np.array([np.nan, np.nan, np.nan])
    phi = math.asin(float(np.clip(-r[2, 0], -1.0, 1.0)))
    cos_phi = math.cos(phi)
    if abs(cos_phi) > 1e-12:
        omega = math.atan2(r[2, 1], r[2, 2])
        kappa = math.atan2(r[1, 0], r[0, 0])
    else:
        omega = 0.0
        kappa = math.atan2(-r[0, 1], r[1, 1])
    return np.degrees([omega, phi, kappa])


def _first_matrix(outputs, keys):
    for key in keys:
        value = outputs.get(key)
        arr = None if value is None else _matrix_or_none(value)
        if arr is not None:
            return arr
    return None


def _matrix_or_none(value):
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _add_accuracy_row(rows, method, scope, variable, value, unit, group, basis, status=None, note=""):
    rows.append({
        "method": method,
        "scope": scope,
        "group": group,
        "variable": variable,
        "value": _value_or_nd(value),
        "unit": unit,
        "indicator": status or ("not_available" if not _is_number(value) else "neutral"),
        "article_basis": basis,
        "note": note,
    })


def _component_stats_mm(residuals):
    arr = _matrix_or_none(residuals)
    if arr is None or arr.ndim != 2 or arr.shape[1] < 3:
        return {}
    xyz = arr[:, :3]
    rms = np.sqrt(np.mean(xyz ** 2, axis=0)) * 1000.0
    std = np.std(xyz, axis=0, ddof=1 if len(xyz) > 1 else 0) * 1000.0
    max_3sigma = 3.0 * std
    return {
        "m_x": rms[0],
        "m_y": rms[1],
        "m_z": rms[2],
        "m_p": float(np.sqrt(np.sum(rms ** 2))),
        "sigma_X": std[0],
        "sigma_Y": std[1],
        "sigma_Z": std[2],
        "M_X": max_3sigma[0],
        "M_Y": max_3sigma[1],
        "M_Z": max_3sigma[2],
    }


def accuracy_quality_rows(outputs_by_method):
    """Generate geodetic accuracy/quality variables requested for each processed method."""
    rows = []
    missing_instrument = "Instrument uncertainty was not provided in the input configuration."
    no_icp = "Independent check points are not explicitly separated from GCPs in the current dataset."
    no_angle_cov = "Angular covariance was not available; rotation angles are reported without formal angular error."
    for method, method_outputs in outputs_by_method.items():
        rows.extend(aggregate_method_accuracy_rows(method, method_outputs))
        for scope, outputs in iter_output_scopes(method_outputs):
            residuals = outputs.get("residuals_xyz")
            stats = _component_stats_mm(residuals)
            for variable in ("m_x", "m_y", "m_z", "m_p", "sigma_X", "sigma_Y", "sigma_Z", "M_X", "M_Y", "M_Z"):
                _add_accuracy_row(
                    rows, method, scope, f"GCP_{variable}", stats.get(variable), "mm",
                    "accuracy_statistics",
                    "Residual RMS and standard-deviation accuracy used for GCP validation.",
                    _metric_status(stats.get(variable), variable),
                )
            for variable in ("m_x", "m_y", "m_z", "m_p"):
                _add_accuracy_row(
                    rows, method, scope, f"ICP_{variable}", np.nan, "mm",
                    "external_validation", "Independent check-point validation.", "not_available", no_icp,
                )

            sigma0_m = outputs.get("sigma0", outputs.get("audit_sigma0"))
            _add_accuracy_row(rows, method, scope, "sigma_0", sigma0_m, "m", "accuracy_statistics",
                              "A posteriori/a priori variance factor used in geodetic weighting.",
                              _metric_status((float(sigma0_m) * 1000.0) if _is_number(sigma0_m) else np.nan, "sigma_0"))
            _add_accuracy_row(rows, method, scope, "sigma_0_mm", (float(sigma0_m) * 1000.0) if _is_number(sigma0_m) else np.nan,
                              "mm", "accuracy_statistics", "Variance factor expressed in millimetres.",
                              _metric_status((float(sigma0_m) * 1000.0) if _is_number(sigma0_m) else np.nan, "sigma_0_mm"))

            qx = _first_matrix(outputs, ("covariance", "numeric_covariance_approx", "Q_xi"))
            tr_qx = float(np.trace(qx)) if qx is not None and qx.ndim == 2 and qx.shape[0] == qx.shape[1] else np.nan
            _add_accuracy_row(rows, method, scope, "tr_Q_x", tr_qx, "parameter_variance",
                              "covariance_trace", "Trace of Q_x=(A^T P A)^-1 as global precision metric.",
                              _metric_status(tr_qx, "tr_Q_x"))
            residual_cov = _matrix_or_none(outputs.get("residual_covariance_approx"))
            enh = float(np.mean(np.diag(residual_cov))) if residual_cov is not None and residual_cov.shape == (3, 3) else np.nan
            _add_accuracy_row(rows, method, scope, "sigma_ENH2_I_trace", 3.0 * enh if _is_number(enh) else np.nan,
                              "m2", "covariance_trace", "Simplified isotropic coordinate covariance sigma_ENH^2 I.",
                              _metric_status((3.0 * enh) if _is_number(enh) else np.nan, "sigma_ENH2_I_trace"))

            if residual_cov is not None and residual_cov.shape == (3, 3):
                sigma_hd2 = float(residual_cov[0, 0] + residual_cov[1, 1])
                sigma_ah2 = float(residual_cov[2, 2])
            else:
                sigma_hd2 = np.nan
                sigma_ah2 = np.nan
            _add_accuracy_row(rows, method, scope, "sigma_AH2", sigma_ah2 * 1e6 if _is_number(sigma_ah2) else np.nan,
                              "mm2", "geometric_component_uncertainty", "Height-difference uncertainty from vertical residual covariance.",
                              _metric_status(sigma_ah2 * 1e6 if _is_number(sigma_ah2) else np.nan, "sigma_AH2"))
            _add_accuracy_row(rows, method, scope, "sigma_HD2", sigma_hd2 * 1e6 if _is_number(sigma_hd2) else np.nan,
                              "mm2", "geometric_component_uncertainty", "Horizontal-distance uncertainty from XY residual covariance.",
                              _metric_status(sigma_hd2 * 1e6 if _is_number(sigma_hd2) else np.nan, "sigma_HD2"))
            _add_accuracy_row(rows, method, scope, "HD2_sigma_psi2", np.nan, "mm2",
                              "geometric_component_uncertainty", "Horizontal angular contribution HD^2*sigma_psi^2.",
                              "not_available", missing_instrument)

            centroid = outputs.get("target_centroid", outputs.get("source_centroid"))
            centroid_arr = _matrix_or_none(centroid)
            if centroid_arr is not None:
                centroid_arr = centroid_arr.ravel()
            for idx, var in enumerate(("E_TS", "N_TS", "H_TS")):
                _add_accuracy_row(rows, method, scope, var, centroid_arr[idx] if centroid_arr is not None and centroid_arr.size > idx else np.nan,
                                  "m", "free_station_optimization",
                                  "Free-station optimum represented by control-point centre of gravity.",
                                  "neutral" if centroid_arr is not None else "not_available",
                                  "" if var != "H_TS" else "Height c does not influence the planar optimum in the referenced free-station criterion.")

            scale = outputs.get("scale", outputs.get("distance_scale_estimate"))
            distance_scale = outputs.get("distance_scale_estimate", scale)
            ortho = outputs.get("orthogonality_error")
            _add_accuracy_row(rows, method, scope, "lambda", scale, "dimensionless",
                              "similarity_isometry", "Similarity/isometry scale lambda=||p'||/||p||.",
                              _metric_status((abs(float(scale) - 1.0) * 1e6) if _is_number(scale) else np.nan, "scale_deviation"))
            _add_accuracy_row(rows, method, scope, "lambda_av", distance_scale, "dimensionless",
                              "similarity_isometry", "Average scale from homologous vector/distance combinations.",
                              _metric_status((abs(float(distance_scale) - 1.0) * 1e6) if _is_number(distance_scale) else np.nan, "scale_deviation"))
            _add_accuracy_row(rows, method, scope, "orthogonality_deviation_norm", ortho, "frobenius_norm",
                              "similarity_isometry", "Norm of I - Q_R^T Q_R for quasi-similarity/isometry validation.",
                              _metric_status(ortho, "orthogonality"))

            rotation = _matrix_or_none(outputs.get("rotation_matrix"))
            angles = rotation_to_omega_phi_kappa(rotation) if rotation is not None else np.array([np.nan, np.nan, np.nan])
            for var, val in zip(("omega", "phi", "kappa"), angles):
                _add_accuracy_row(rows, method, scope, var, val, "deg", "external_orientation",
                                  "Exterior orientation angle derived from the rotation matrix.", "neutral" if _is_number(val) else "not_available")
            for var in ("m_omega", "m_phi", "m_kappa", "M_omega", "M_phi", "M_kappa", "d_omega", "d_phi", "d_kappa"):
                _add_accuracy_row(rows, method, scope, var, np.nan, "deg", "external_orientation",
                                  "Angular error/differential for orientation uncertainty propagation.", "not_available", no_angle_cov)

            for var in ("sigma_S", "sigma_psi", "sigma_nu", "sigma_ENH", "sigma_xi", "sigma_eta"):
                _add_accuracy_row(rows, method, scope, var, np.nan, "input_unit",
                                  "instrument_uncertainty", "Instrument/sensor uncertainty input.", "not_available", missing_instrument)

            for variable, basis in (
                ("error_ellipse_2d", "2D XY error ellipse from covariance eigenvalues."),
                ("vertical_error_surface_3d", "3D vertical error surface/vector description from Z residuals."),
            ):
                description = describe_error_geometry(outputs)
                rows.append({
                    "method": method,
                    "scope": scope,
                    "group": "visual_interpretation",
                    "variable": variable,
                    "value": description,
                    "unit": "text",
                    "indicator": "neutral",
                    "article_basis": basis,
                    "note": description,
                })
    return rows


def aggregate_method_accuracy_rows(method, method_outputs):
    """Build one comparable method-level row set from all valid station/network scopes."""
    rows = []
    residual_chunks = []
    scales = []
    distance_scales = []
    orthogonality = []
    sigma0_values = []
    tr_qx_values = []
    angles = []
    centroids = []
    ellipse_notes = []
    for scope, outputs in iter_output_scopes(method_outputs):
        if scope == "network":
            qx = _first_matrix(outputs, ("covariance", "numeric_covariance_approx", "Q_xi"))
            if qx is not None and qx.ndim == 2 and qx.shape[0] == qx.shape[1]:
                tr_qx_values.append(float(np.trace(qx)))
            sigma0 = outputs.get("sigma0", outputs.get("audit_sigma0"))
            if _is_number(sigma0):
                sigma0_values.append(float(sigma0))
            continue
        residuals = _matrix_or_none(outputs.get("residuals_xyz"))
        if residuals is not None and residuals.ndim == 2 and residuals.shape[1] >= 3:
            residual_chunks.append(residuals[:, :3])
        for key, target in (("scale", scales), ("distance_scale_estimate", distance_scales), ("orthogonality_error", orthogonality)):
            value = outputs.get(key)
            if _is_number(value):
                target.append(float(value))
        sigma0 = outputs.get("sigma0", outputs.get("audit_sigma0"))
        if _is_number(sigma0):
            sigma0_values.append(float(sigma0))
        qx = _first_matrix(outputs, ("covariance", "numeric_covariance_approx", "Q_xi"))
        if qx is not None and qx.ndim == 2 and qx.shape[0] == qx.shape[1]:
            tr_qx_values.append(float(np.trace(qx)))
        rotation = _matrix_or_none(outputs.get("rotation_matrix"))
        angle_triplet = rotation_to_omega_phi_kappa(rotation) if rotation is not None else None
        if angle_triplet is not None and np.all(np.isfinite(angle_triplet)):
            angles.append(angle_triplet)
        centroid = _matrix_or_none(outputs.get("target_centroid", outputs.get("source_centroid")))
        if centroid is not None:
            flat = centroid.ravel()
            if flat.size >= 3:
                centroids.append(flat[:3])
        note = describe_error_geometry(outputs)
        if "N/D because" not in note:
            ellipse_notes.append(f"{scope}: {note}")

    combined_residuals = np.vstack(residual_chunks) if residual_chunks else None
    stats = _component_stats_mm(combined_residuals)
    scope = "method_summary"
    for variable in ("m_x", "m_y", "m_z", "m_p", "sigma_X", "sigma_Y", "sigma_Z", "M_X", "M_Y", "M_Z"):
        _add_accuracy_row(
            rows, method, scope, f"GCP_{variable}", stats.get(variable), "mm",
            "accuracy_statistics",
            "Aggregated GCP residual statistics across all successful scopes for this method.",
            _metric_status(stats.get(variable), variable),
        )
    for variable in ("m_x", "m_y", "m_z", "m_p"):
        _add_accuracy_row(
            rows, method, scope, f"ICP_{variable}", np.nan, "mm",
            "external_validation", "Independent check-point validation.", "not_available",
            "Independent check points are not explicitly separated from GCPs in the current dataset.",
        )
    sigma0_mean = float(np.mean(sigma0_values)) if sigma0_values else np.nan
    _add_accuracy_row(rows, method, scope, "sigma_0", sigma0_mean, "m",
                      "accuracy_statistics", "Mean sigma0 across available scopes.",
                      _metric_status(sigma0_mean * 1000.0 if _is_number(sigma0_mean) else np.nan, "sigma_0"))
    _add_accuracy_row(rows, method, scope, "sigma_0_mm", sigma0_mean * 1000.0 if _is_number(sigma0_mean) else np.nan, "mm",
                      "accuracy_statistics", "Mean sigma0 expressed in millimetres.",
                      _metric_status(sigma0_mean * 1000.0 if _is_number(sigma0_mean) else np.nan, "sigma_0_mm"))
    tr_qx_mean = float(np.mean(tr_qx_values)) if tr_qx_values else np.nan
    _add_accuracy_row(rows, method, scope, "tr_Q_x", tr_qx_mean, "parameter_variance",
                      "covariance_trace", "Mean trace of available Q_x matrices.",
                      _metric_status(tr_qx_mean, "tr_Q_x"))
    scale_mean = float(np.mean(scales)) if scales else np.nan
    dist_scale_mean = float(np.mean(distance_scales)) if distance_scales else scale_mean
    ortho_mean = float(np.mean(orthogonality)) if orthogonality else np.nan
    _add_accuracy_row(rows, method, scope, "lambda", scale_mean, "dimensionless",
                      "similarity_isometry", "Mean similarity/isometry scale across successful scopes.",
                      _metric_status(abs(scale_mean - 1.0) * 1e6 if _is_number(scale_mean) else np.nan, "scale_deviation"))
    _add_accuracy_row(rows, method, scope, "lambda_av", dist_scale_mean, "dimensionless",
                      "similarity_isometry", "Mean distance-derived scale across successful scopes.",
                      _metric_status(abs(dist_scale_mean - 1.0) * 1e6 if _is_number(dist_scale_mean) else np.nan, "scale_deviation"))
    _add_accuracy_row(rows, method, scope, "orthogonality_deviation_norm", ortho_mean, "frobenius_norm",
                      "similarity_isometry", "Mean norm of I - Q_R^T Q_R.",
                      _metric_status(ortho_mean, "orthogonality"))

    angle_arr = np.asarray(angles, dtype=float) if angles else np.empty((0, 3))
    angle_mean = np.mean(angle_arr, axis=0) if angle_arr.size else np.array([np.nan, np.nan, np.nan])
    angle_rms = np.sqrt(np.mean((angle_arr - angle_mean) ** 2, axis=0)) if len(angle_arr) > 1 else np.array([np.nan, np.nan, np.nan])
    for var, value in zip(("omega", "phi", "kappa"), angle_mean):
        _add_accuracy_row(rows, method, scope, var, value, "deg", "external_orientation",
                          "Mean exterior-orientation angle across successful scopes.",
                          "neutral" if _is_number(value) else "not_available")
    for var, value in zip(("m_omega", "m_phi", "m_kappa"), angle_rms):
        _add_accuracy_row(rows, method, scope, var, value, "deg", "external_orientation",
                          "RMS scatter of orientation angles across successful scopes.",
                          _metric_status(value, var))
    for var, value in zip(("M_omega", "M_phi", "M_kappa"), angle_rms * 3.0):
        _add_accuracy_row(rows, method, scope, var, value, "deg", "external_orientation",
                          "Three-sigma angular scatter across successful scopes.",
                          _metric_status(value, var))
    for var in ("d_omega", "d_phi", "d_kappa"):
        _add_accuracy_row(rows, method, scope, var, np.nan, "deg", "external_orientation",
                          "Differential angular propagation requires observation-level angular covariance.",
                          "not_available", "Angular observation covariance is not available in the current input.")

    centroid_mean = np.mean(np.asarray(centroids), axis=0) if centroids else np.array([np.nan, np.nan, np.nan])
    for idx, var in enumerate(("E_TS", "N_TS", "H_TS")):
        _add_accuracy_row(rows, method, scope, var, centroid_mean[idx], "m", "free_station_optimization",
                          "Mean control-point centre of gravity across successful scopes.",
                          "neutral" if _is_number(centroid_mean[idx]) else "not_available")
    for var in ("sigma_AH2", "sigma_HD2", "HD2_sigma_psi2", "sigma_S", "sigma_psi", "sigma_nu", "sigma_ENH", "sigma_xi", "sigma_eta", "sigma_ENH2_I_trace"):
        _add_accuracy_row(rows, method, scope, var, np.nan, "input_unit",
                          "instrument_uncertainty", "Instrument/sensor covariance input.", "not_available",
                          "Instrument uncertainty was not provided in the input configuration.")
    note = " ".join(ellipse_notes[:6]) if ellipse_notes else "N/D because no covariance/residual geometry was available."
    for variable in ("error_ellipse_2d", "vertical_error_surface_3d"):
        rows.append({
            "method": method,
            "scope": scope,
            "group": "visual_interpretation",
            "variable": variable,
            "value": note,
            "unit": "text",
            "indicator": "neutral",
            "article_basis": "Aggregated textual description of available error geometry.",
            "note": note,
        })
    return rows


def describe_error_geometry(outputs):
    residuals = _matrix_or_none(outputs.get("residuals_xyz"))
    residual_cov = _matrix_or_none(outputs.get("residual_covariance_approx"))
    parts = []
    if residual_cov is not None and residual_cov.shape == (3, 3):
        cov_xy = residual_cov[:2, :2]
        eigvals, eigvecs = np.linalg.eigh(cov_xy)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        major = 2.0 * math.sqrt(max(float(eigvals[0]), 0.0)) * 1000.0
        minor = 2.0 * math.sqrt(max(float(eigvals[1]), 0.0)) * 1000.0
        angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
        parts.append(f"2D ellipse: major={major:.3f} mm, minor={minor:.3f} mm, azimuth={angle:.3f} deg.")
    else:
        parts.append("2D ellipse: N/D because covariance is not available.")
    if residuals is not None and residuals.ndim == 2 and residuals.shape[1] >= 3:
        z = residuals[:, 2] * 1000.0
        parts.append(f"3D vertical error: min={np.min(z):.3f} mm, max={np.max(z):.3f} mm, RMS={math.sqrt(np.mean(z ** 2)):.3f} mm.")
    else:
        parts.append("3D vertical error: N/D because point residuals are not available.")
    return " ".join(parts)


def write_accuracy_quality_outputs(output_dir, outputs_by_method):
    rows = accuracy_quality_rows(outputs_by_method)
    long_df = pd.DataFrame(rows)
    long_df.to_csv(output_dir / "accuracy_quality_full.csv", index=False)
    if long_df.empty:
        return
    wide_source = long_df.copy()
    wide_source["wide_variable"] = wide_source["variable"].map(_wide_variable_name)
    wide_df = wide_source.pivot_table(
        index=["method", "scope"],
        columns="wide_variable",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide_df.to_csv(output_dir / "accuracy_quality_summary.csv", index=False)
    write_accuracy_quality_html(output_dir / "accuracy_quality_summary.html", long_df)
    write_accuracy_quality_notes(output_dir / "accuracy_quality_notes.md")


def _wide_variable_name(variable):
    name = str(variable)
    if name.startswith("GCP_M_"):
        return name.replace("GCP_M_", "GCP_max_")
    if name.startswith("ICP_M_"):
        return name.replace("ICP_M_", "ICP_max_")
    if name.startswith("M_"):
        return name.replace("M_", "max_")
    return name


def write_accuracy_quality_notes(path):
    lines = [
        "# Accuracy and quality variables",
        "",
        "Blue indicators mark comparatively better/lower-error values; red indicators mark worse/higher-error values.",
        "`N/D` means the current data or method output does not expose the observation needed for that variable.",
        "",
        "Requested groups covered: coordinate accuracy, Q_x covariance trace, simplified sigma_ENH^2 I, geometric component uncertainty, free-station centre of gravity, similarity/isometry metrics, GCP/ICP validation, instrument uncertainty placeholders and exterior-orientation angles.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_accuracy_quality_html(path, long_df):
    df = long_df.copy()
    numeric = pd.to_numeric(df["value"], errors="coerce")
    df["numeric_value"] = numeric
    ranked = {}
    for variable, group in df.groupby("variable"):
        values = group["numeric_value"].dropna()
        if values.empty:
            continue
        lo = values.quantile(0.25)
        hi = values.quantile(0.75)
        ranked[variable] = (lo, hi)
    rows = []
    for _, row in df.iterrows():
        value = row["value"]
        cls = "nd"
        if _is_number(value) and row["variable"] in ranked:
            lo, hi = ranked[row["variable"]]
            val = float(value)
            if val <= lo:
                cls = "good"
            elif val >= hi:
                cls = "bad"
            else:
                cls = "mid"
        elif row["indicator"] == "not_available":
            cls = "nd"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['method']))}</td>"
            f"<td>{html.escape(str(row['scope']))}</td>"
            f"<td>{html.escape(str(row['group']))}</td>"
            f"<td>{html.escape(str(row['variable']))}</td>"
            f"<td class='{cls}'>{html.escape(str(value))}</td>"
            f"<td>{html.escape(str(row['unit']))}</td>"
            f"<td>{html.escape(str(row['note']))}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Accuracy quality summary</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #f5f5f5; }}
.good {{ background: #d8ecff; color: #064f8f; font-weight: 700; }}
.bad {{ background: #ffd9d9; color: #9b1010; font-weight: 700; }}
.mid {{ background: #f7f7f7; }}
.nd {{ background: #eeeeee; color: #666; }}
</style>
</head>
<body>
<h1>Accuracy and Quality Summary</h1>
<p>Blue cells indicate better/lower values within the same variable; red cells indicate worse/higher values. N/D means the data was not available for that method/output.</p>
<table>
<thead><tr><th>Method</th><th>Scope</th><th>Group</th><th>Variable</th><th>Value</th><th>Unit</th><th>Note</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>"""
    path.write_text(document, encoding="utf-8")


def article_quality_assessment(outputs_by_method, station_df, residual_df):
    """Build method-specific quality indicators following the referenced papers."""
    rows = []
    basis = {
        "rmse": "Least-squares similarity transformation residual quality (Schut/Procrustes/CMTLS).",
        "sigma0": "A posteriori variance factor used in geodetic least-squares adjustment.",
        "orthogonality": "Rotation matrix validity required by 3-D similarity transformation.",
        "distance": "Q-ST scale evaluation using homologous inter-point distances.",
        "covariance": "Uncertainty modelling through covariance/correlation matrices.",
        "outlier": "Residual screening compatible with geodetic adjustment diagnostics.",
    }

    if station_df is not None and not station_df.empty:
        for method, group in station_df.groupby("method", dropna=False):
            ok = group[group["success"] == True]  # noqa: E712
            rows.extend([
                {"method": method, "scope": "method", "criterion": "successful_stations", "article_basis": "Network completeness and isolated-island diagnosis.", "value": int(len(ok)), "unit": "stations"},
                {"method": method, "scope": "method", "criterion": "attempted_stations", "article_basis": "Network completeness and isolated-island diagnosis.", "value": int(len(group)), "unit": "stations"},
                {"method": method, "scope": "method", "criterion": "mean_rmse", "article_basis": basis["rmse"], "value": float(ok["rmse"].mean()) if "rmse" in ok and not ok.empty else np.nan, "unit": "m"},
                {"method": method, "scope": "method", "criterion": "max_rmse", "article_basis": basis["rmse"], "value": float(ok["rmse"].max()) if "rmse" in ok and not ok.empty else np.nan, "unit": "m"},
                {"method": method, "scope": "method", "criterion": "mean_sigma0", "article_basis": basis["sigma0"], "value": float(ok["sigma0"].mean()) if "sigma0" in ok and not ok.empty else np.nan, "unit": "m"},
                {"method": method, "scope": "method", "criterion": "mean_scale_deviation_ppm", "article_basis": "Scale consistency in Helmert/TFS transformations.", "value": float(np.mean(np.abs(ok["scale"] - 1.0)) * 1e6) if "scale" in ok and not ok.empty else np.nan, "unit": "ppm"},
            ])

    for method, method_outputs in outputs_by_method.items():
        for scope, outputs in iter_output_scopes(method_outputs):
            rows.extend([
                {"method": method, "scope": scope, "criterion": "audit_sigma0", "article_basis": basis["sigma0"], "value": outputs.get("audit_sigma0", np.nan), "unit": "m"},
                {"method": method, "scope": scope, "criterion": "residual_sum_squares", "article_basis": basis["rmse"], "value": outputs.get("residual_sum_squares", np.nan), "unit": "m2"},
                {"method": method, "scope": scope, "criterion": "degrees_of_freedom", "article_basis": "Redundancy used for variance-factor estimation.", "value": outputs.get("degrees_of_freedom", np.nan), "unit": "count"},
                {"method": method, "scope": scope, "criterion": "rotation_determinant", "article_basis": basis["orthogonality"], "value": outputs.get("rotation_determinant", np.nan), "unit": "dimensionless"},
                {"method": method, "scope": scope, "criterion": "orthogonality_error", "article_basis": basis["orthogonality"], "value": outputs.get("orthogonality_error", np.nan), "unit": "frobenius_norm"},
                {"method": method, "scope": scope, "criterion": "distance_residual_rms", "article_basis": basis["distance"], "value": _rms_array(outputs.get("distance_residuals", np.array([]))), "unit": "m"},
                {"method": method, "scope": scope, "criterion": "distance_scale_estimate", "article_basis": basis["distance"], "value": outputs.get("distance_scale_estimate", np.nan), "unit": "dimensionless"},
                {"method": method, "scope": scope, "criterion": "covariance_trace", "article_basis": basis["covariance"], "value": _trace_or_nan(outputs.get("covariance", outputs.get("numeric_covariance_approx", np.array([])))), "unit": "parameter_variance"},
                {"method": method, "scope": scope, "criterion": "normal_matrix_trace", "article_basis": "Normal-equation strength in least-squares adjustment.", "value": _trace_or_nan(outputs.get("normal_matrix", outputs.get("numeric_normal_matrix", np.array([])))), "unit": "normal_equation_units"},
            ])

    if residual_df is not None and not residual_df.empty and {"method", "station", "v_norm"}.issubset(residual_df.columns):
        for (method, station), group in residual_df.groupby(["method", "station"], dropna=False):
            rows.append({
                "method": method,
                "scope": station,
                "criterion": "residual_norm_rms",
                "article_basis": basis["outlier"],
                "value": _rms_array(group["v_norm"]),
                "unit": "m",
            })

    return pd.DataFrame(rows)


def write_article_quality_outputs(output_dir, outputs_by_method, station_df, residual_df):
    assessment = article_quality_assessment(outputs_by_method, station_df, residual_df)
    assessment.to_csv(output_dir / "article_quality_assessment.csv", index=False)
    lines = [
        "# Article quality assessment outputs",
        "",
        "This file maps each exported quality indicator to the evaluation logic used in the reference literature.",
        "",
        "- Similarity transformation quality: RMSE, residual sum of squares, rotation determinant and orthogonality error.",
        "- Q-ST quality: scale from homologous distances and RMS of distance residuals.",
        "- Least-squares/network quality: sigma0, degrees of freedom, normal matrix and covariance trace.",
        "- Dimensional-control diagnostics: station completeness, scale deviation and residual/outlier indicators.",
        "",
        "The detailed numeric table is `article_quality_assessment.csv`.",
        "",
    ]
    (output_dir / "article_quality_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots(output_dir, station_df, residual_df, outlier_df=None):
    if plt is None:
        (output_dir / "plots_not_generated.txt").write_text(
            "matplotlib is not installed in this Python environment.\n",
            encoding="utf-8",
        )
        return

    if station_df.empty:
        return

    ok = station_df[station_df["success"] == True].copy()  # noqa: E712
    if not ok.empty and "rmse" in ok:
        fig, ax = plt.subplots(figsize=(12, 5))
        label_col = "method"
        if "run_label" in ok.columns:
            ok["method_run_label"] = ok["method"].astype(str) + " [" + ok["run_label"].astype(str) + "]"
            label_col = "method_run_label"
        by_method = ok.groupby(label_col)["rmse"].mean().sort_values()
        by_method.plot(kind="bar", ax=ax, color="#9ad0c2", edgecolor="#4b5563")
        ax.set_ylabel("Average RMSE")
        ax.set_title("Average RMSE by method and processing cycle")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "plot_average_rmse_by_method.png", dpi=160)
        plt.close(fig)

    if not residual_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        residual_df["v_norm"].dropna().plot(kind="hist", bins=40, ax=ax, color="#f2b5d4", edgecolor="#4b5563")
        ax.set_xlabel("Residual norm")
        ax.set_title("Residual norm histogram")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "plot_residual_histogram.png", dpi=160)
        plt.close(fig)

    if outlier_df is not None and not outlier_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        label_col = "method"
        plotting = outlier_df.copy()
        if "run_label" in plotting.columns:
            plotting["method_run_label"] = (
                plotting["method"].astype(str) + " [" + plotting["run_label"].astype(str) + "]"
            )
            label_col = "method_run_label"
        counts = plotting.groupby(label_col)["is_outlier_candidate"].sum().sort_values()
        counts.plot(kind="bar", ax=ax, color="#f7c59f", edgecolor="#4b5563")
        ax.set_ylabel("Outlier candidates")
        ax.set_title("Outlier candidates by method and processing cycle")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "plot_outlier_candidates_by_method.png", dpi=160)
        plt.close(fig)

    if not ok.empty and "elapsed_time_total" in ok:
        fig, ax = plt.subplots(figsize=(12, 5))
        time_by_method = ok.groupby("method")["elapsed_time_total"].sum().sort_values()
        time_by_method.plot(kind="bar", ax=ax, color="#b8c0ff", edgecolor="#4b5563")
        ax.set_ylabel("Total time (s)")
        ax.set_title("Processing time by method")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "plot_time_by_method.png", dpi=160)
        plt.close(fig)


ERROR_ELLIPSE_95_2D = math.sqrt(5.991464547)
ERROR_ELLIPSOID_95_3D = math.sqrt(7.814727903)


def _covariance_axes(covariance, dimensions):
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] < dimensions or cov.shape[1] < dimensions:
        return None
    cov = 0.5 * (cov[:dimensions, :dimensions] + cov[:dimensions, :dimensions].T)
    if not np.all(np.isfinite(cov)) or np.allclose(cov, 0):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    eigenvectors = eigenvectors[:, order]
    sigmas = np.sqrt(eigenvalues)
    if not np.any(sigmas > 0):
        return None
    return sigmas, eigenvectors


def _ellipse_xy(center, covariance, scale_factor=1.0, confidence_sigma=ERROR_ELLIPSE_95_2D, points=120):
    axes_info = _covariance_axes(covariance, 2)
    if axes_info is None:
        return None
    sigmas, eigenvectors = axes_info
    theta = np.linspace(0, 2 * np.pi, points)
    circle = np.vstack((np.cos(theta), np.sin(theta)))
    axes = np.diag(sigmas * confidence_sigma * scale_factor)
    ellipse = eigenvectors @ axes @ circle
    return center[0] + ellipse[0], center[1] + ellipse[1]


def _iter_uncertainty_covariances(method_outputs):
    for _, station in _iter_visual_stations(method_outputs):
        uncertainties = station.get("point_uncertainties", {}) or station.get("point_uncertainties_approx", {})
        if not isinstance(uncertainties, dict):
            continue
        for cov in uncertainties.values():
            if _covariance_axes(cov, 3) is not None:
                yield cov


def _method_coordinate_span(method_outputs):
    coords_all = []
    for _, station in _iter_visual_stations(method_outputs):
        coords = np.asarray(station.get("coords", []), dtype=float)
        if coords.ndim == 2 and coords.shape[1] >= 3 and coords.size:
            coords_all.append(coords[:, :3])
    if not coords_all:
        return 1.0
    coords = np.vstack(coords_all)
    ranges = np.nanmax(coords, axis=0) - np.nanmin(coords, axis=0)
    span = float(np.linalg.norm(ranges))
    return span if math.isfinite(span) and span > 0 else 1.0


def _error_geometry_visual_scale(method_outputs, dimensions=3, confidence_sigma=ERROR_ELLIPSOID_95_3D):
    major_axes = []
    for cov in _iter_uncertainty_covariances(method_outputs):
        axes_info = _covariance_axes(cov, dimensions)
        if axes_info is None:
            continue
        sigmas, _ = axes_info
        major_axes.append(float(sigmas[0] * confidence_sigma))
    if not major_axes:
        return 1.0
    representative_axis = float(np.nanpercentile(major_axes, 75))
    if not math.isfinite(representative_axis) or representative_axis <= 0:
        return 1.0
    span = _method_coordinate_span(method_outputs)
    target_radius = max(span * 0.012, 0.035)
    scale = target_radius / representative_axis
    return float(np.clip(scale, 1.0, 2500.0))


def _station_role_masks(all_ids, station):
    roles = [_point_role_for_plot(pid, station) for pid in all_ids]
    return {
        "IRRADIATED": np.asarray([role == "IRRADIATED" for role in roles], dtype=bool),
        "CP": np.asarray([role == "CP" for role in roles], dtype=bool),
        "ICP": np.asarray([role == "ICP" for role in roles], dtype=bool),
    }


def _ellipsoid_wireframe(center, covariance, scale_factor=1.0, confidence_sigma=ERROR_ELLIPSOID_95_3D, points=64):
    axes_info = _covariance_axes(covariance, 3)
    if axes_info is None:
        return None
    sigmas, eigenvectors = axes_info
    axes = sigmas * confidence_sigma * scale_factor
    theta = np.linspace(0, 2 * np.pi, points)
    circles = [
        np.vstack((np.cos(theta) * axes[0], np.sin(theta) * axes[1], np.zeros_like(theta))),
        np.vstack((np.cos(theta) * axes[0], np.zeros_like(theta), np.sin(theta) * axes[2])),
        np.vstack((np.zeros_like(theta), np.cos(theta) * axes[1], np.sin(theta) * axes[2])),
    ]
    center = np.asarray(center, dtype=float).reshape(3, 1)
    ring_lines = [(center + eigenvectors @ circle) for circle in circles]
    axis_lines = []
    for axis_index in range(3):
        direction = eigenvectors[:, axis_index].reshape(3, 1)
        endpoint = direction * axes[axis_index]
        axis_lines.append(np.hstack((center - endpoint, center + endpoint)))
    return {
        "rings": ring_lines,
        "axes": axis_lines,
        "sigmas": sigmas,
        "confidence_axes": sigmas * confidence_sigma,
        "visual_axes": axes,
        "visual_scale": scale_factor,
    }


def _iter_visual_stations(method_outputs):
    if "network" in method_outputs:
        network_result = method_outputs["network"]
        for station in network_result.get("transformed_stations", []):
            yield station.get("name", station.get("station", "Unknown")), station
    for station_name, station_outputs in method_outputs.items():
        if station_name == "network":
            continue
        if isinstance(station_outputs, dict) and "coords" in station_outputs:
            yield station_name, station_outputs


def _split_method_run_key(method_name):
    text = str(method_name)
    marker = "_outliers_removed"
    if marker not in text:
        return text, None
    base, suffix = text.split(marker, 1)
    run_label = "outliers_removed" + suffix
    return base, run_label


def _residual_rows_for_method(residual_df, method_name):
    if residual_df is None or residual_df.empty:
        return pd.DataFrame()
    base_method, run_label = _split_method_run_key(method_name)
    df = residual_df[residual_df["method"].astype(str) == str(base_method)].copy()
    if run_label and "run_label" in df.columns:
        df = df[df["run_label"].astype(str) == str(run_label)].copy()
    elif "run_label" in df.columns:
        df = df[df["run_label"].astype(str).isin({"", "initial", "network"})].copy()
    return df


def _add_residual_vectors_3d(fig, method_name, residual_df, max_vectors=800):
    if residual_df is None or residual_df.empty:
        return
    method_res = _residual_rows_for_method(residual_df, method_name)
    required = {"observed_x", "observed_y", "observed_z", "adjusted_x", "adjusted_y", "adjusted_z"}
    if method_res.empty or not required.issubset(method_res.columns):
        return
    method_res = method_res.dropna(subset=list(required)).head(max_vectors)
    for _, row in method_res.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row["observed_x"], row["adjusted_x"], None],
            y=[row["observed_y"], row["adjusted_y"], None],
            z=[row["observed_z"], row["adjusted_z"], None],
            mode="lines",
            name=f"{method_name} residual vectors",
            legendgroup=f"{method_name}_residuals",
            showlegend=False,
            line=dict(color="#d62728", width=3),
            opacity=0.55,
            hoverinfo="text",
            text=[
                f"{method_name}<br>{row.get('station', '')} {row.get('point', row.get('point_id', ''))}<br>observed",
                f"adjusted<br>|v|={row.get('v_norm', np.nan):.6f} m",
                "",
            ],
        ))


def _rmse_residual_rows(residual_df, method_name=None, max_points=5000):
    if residual_df is None or residual_df.empty:
        return pd.DataFrame()
    required = {"adjusted_x", "adjusted_y", "adjusted_z", "v_norm", "method"}
    if not required.issubset(residual_df.columns):
        return pd.DataFrame()
    df = residual_df.copy()
    if method_name:
        df = _residual_rows_for_method(df, method_name)
    df = df.dropna(subset=["adjusted_x", "adjusted_y", "adjusted_z", "v_norm"])
    if df.empty:
        return df
    for col in ("adjusted_x", "adjusted_y", "adjusted_z", "v_norm"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["adjusted_x", "adjusted_y", "adjusted_z", "v_norm"])
    df["point_rmse_mm"] = df["v_norm"].astype(float) * 1000.0
    return df.head(max_points)


def _point_role_for_plot(point_id, station):
    point_id = str(point_id)
    if point_id in {str(pid) for pid in station.get("icp_ids", [])}:
        return "ICP"
    if point_id in {str(pid) for pid in station.get("ctrl_ids", [])}:
        return "CP"
    code = str(station.get("point_codes", {}).get(point_id, "")).strip().upper()
    if code == "ICP":
        return "ICP"
    if code in {"CONTROL", "CTRL", "CP", "GCP"}:
        return "CP"
    if point_id.upper().startswith("ICP"):
        return "ICP"
    if point_id.upper().startswith("CP"):
        return "CP"
    return "IRRADIATED"


def _point_rmse_lookup(residual_df, independent_df=None, method_name=None):
    frames = []
    for df in (residual_df, independent_df):
        if df is None or df.empty:
            continue
        rows = _residual_rows_for_method(df, method_name) if method_name else df.copy()
        required = {"method", "station", "point", "v_norm"}
        if required.issubset(rows.columns):
            rows = rows.dropna(subset=["method", "station", "point", "v_norm"]).copy()
            if not rows.empty:
                rows["point_rmse_mm"] = pd.to_numeric(rows["v_norm"], errors="coerce") * 1000.0
                frames.append(rows)
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True)
    lookup = {}
    for _, row in combined.iterrows():
        key = (str(row.get("method")), str(row.get("station")), str(row.get("point")))
        value = row.get("point_rmse_mm")
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            lookup[key] = value
    return lookup


def _rmse_point_cloud_rows(outputs_by_method, residual_df=None, independent_df=None, method_name=None, max_points=15000):
    if not outputs_by_method:
        return pd.DataFrame()
    methods = [method_name] if method_name and method_name in outputs_by_method else list(outputs_by_method.keys())
    rows = []
    lookup = _point_rmse_lookup(residual_df, independent_df, method_name=method_name)
    for method in methods:
        method_outputs = outputs_by_method.get(method, {})
        for station_name, station in _iter_visual_stations(method_outputs):
            coords = np.asarray(station.get("coords", []), dtype=float)
            if coords.ndim != 2 or coords.shape[1] < 3 or coords.size == 0:
                continue
            all_ids = list(station.get("all_ids", [f"P{i + 1}" for i in range(len(coords))]))
            for point_id, coord in zip(all_ids, coords[:, :3]):
                role = _point_role_for_plot(point_id, station)
                rmse_mm = lookup.get((str(method), str(station_name), str(point_id)), np.nan)
                rows.append({
                    "method": method,
                    "station": station_name,
                    "point": point_id,
                    "role": role,
                    "adjusted_x": coord[0],
                    "adjusted_y": coord[1],
                    "adjusted_z": coord[2],
                    "point_rmse_mm": rmse_mm,
                })
                if len(rows) >= max_points:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _plotly_scene_layout():
    return dict(
        xaxis=dict(
            title="Easting / X coordinate (m)",
            showbackground=True,
            backgroundcolor="rgb(245,247,250)",
            gridcolor="rgb(210,218,228)",
            zerolinecolor="rgb(180,190,200)",
        ),
        yaxis=dict(
            title="Northing / Y coordinate (m)",
            showbackground=True,
            backgroundcolor="rgb(245,247,250)",
            gridcolor="rgb(210,218,228)",
            zerolinecolor="rgb(180,190,200)",
        ),
        zaxis=dict(
            title="Elevation / Z coordinate (m)",
            showbackground=True,
            backgroundcolor="rgb(245,247,250)",
            gridcolor="rgb(210,218,228)",
            zerolinecolor="rgb(180,190,200)",
        ),
        aspectmode="data",
    )


def _plotly_layout(title, legend_title="Method | station"):
    return dict(
        title=dict(
            text=title,
            x=0.01,
            xanchor="left",
            font=dict(size=18),
        ),
        scene=_plotly_scene_layout(),
        legend=dict(
            title=dict(text=legend_title),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=145, r=230, b=10, t=70),
        template="plotly_white",
    )


def _plotly_left_colorbar(title):
    return dict(
        title=dict(text=title, side="right"),
        x=-0.12,
        xanchor="right",
        y=0.5,
        yanchor="middle",
        len=0.72,
        thickness=18,
        outlinewidth=0.5,
        tickfont=dict(size=10),
    )


def _write_rmse_3d_figure(html_path, title, residual_df, method_name=None, outputs_by_method=None, independent_df=None):
    if not PLOTLY_AVAILABLE:
        return False
    df = _rmse_point_cloud_rows(
        outputs_by_method,
        residual_df=residual_df,
        independent_df=independent_df,
        method_name=method_name,
    ) if outputs_by_method else _rmse_residual_rows(residual_df, method_name=method_name)
    if df.empty:
        return False
    fig = go.Figure()
    colorbar_added = False
    for method, group in df.groupby("method", dropna=False):
        irradiated = group[group["role"].astype(str) == "IRRADIATED"].copy() if "role" in group.columns else pd.DataFrame()
        if not irradiated.empty:
            fig.add_trace(go.Scatter3d(
                x=irradiated["adjusted_x"],
                y=irradiated["adjusted_y"],
                z=irradiated["adjusted_z"],
                mode="markers",
                name=f"{method} irradiated points",
                legendgroup=f"{method}_irradiated",
                marker=dict(size=1.6, color="#050505", opacity=0.62),
                text=[
                    f"Method: {row.get('method', '')}<br>Station: {row.get('station', '')}"
                    f"<br>Point: {row.get('point', row.get('point_id', ''))}"
                    f"<br>Role: Irradiated"
                    f"<br>Easting/X: {float(row.get('adjusted_x', np.nan)):.4f} m"
                    f"<br>Northing/Y: {float(row.get('adjusted_y', np.nan)):.4f} m"
                    f"<br>Elevation/Z: {float(row.get('adjusted_z', np.nan)):.4f} m"
                    for _, row in irradiated.iterrows()
                ],
                hoverinfo="text",
            ))
        colored = group[group.get("role", pd.Series("", index=group.index)).astype(str).isin({"CP", "ICP"})].copy()
        if colored.empty:
            colored = group.copy()
        colored = colored[np.isfinite(pd.to_numeric(colored["point_rmse_mm"], errors="coerce"))].copy()
        if colored.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=colored["adjusted_x"],
            y=colored["adjusted_y"],
            z=colored["adjusted_z"],
            mode="markers",
            name=f"{method} CP/ICP RMSE",
            legendgroup=f"{method}_control_check",
            marker=dict(
                size=5.8,
                color=colored["point_rmse_mm"],
                colorscale="Turbo",
                colorbar=_plotly_left_colorbar("CP/ICP RMSE<br>(mm)") if not colorbar_added else None,
                showscale=not colorbar_added,
                opacity=0.94,
                line=dict(color="#111827", width=0.7),
            ),
            text=[
                f"Method: {row.get('method', '')}<br>Station: {row.get('station', '')}"
                f"<br>Point: {row.get('point', row.get('point_id', ''))}"
                f"<br>Role: {row.get('role', 'CP/ICP')}"
                f"<br>Easting/X: {float(row.get('adjusted_x', np.nan)):.4f} m"
                f"<br>Northing/Y: {float(row.get('adjusted_y', np.nan)):.4f} m"
                f"<br>Elevation/Z: {float(row.get('adjusted_z', np.nan)):.4f} m"
                f"<br>RMSE: {row['point_rmse_mm']:.3f} mm"
                f"<br>Vx: {float(row.get('vx', np.nan)) * 1000:.3f} mm"
                f"<br>Vy: {float(row.get('vy', np.nan)) * 1000:.3f} mm"
                f"<br>Vz: {float(row.get('vz', np.nan)) * 1000:.3f} mm"
                for _, row in colored.iterrows()
            ],
            hoverinfo="text",
        ))
        colorbar_added = True
    fig.update_layout(**_plotly_layout(title, legend_title="Method"))
    fig.write_html(str(html_path), include_plotlyjs="embed")
    return True


def _add_method_to_3d_figure(fig, method_name, method_outputs, include_ellipsoids=True, max_ellipsoids=250):
    ellipsoid_count = 0
    ellipsoid_legend_shown = False
    visual_scale = _error_geometry_visual_scale(method_outputs)
    for station_name, station in _iter_visual_stations(method_outputs):
        coords = np.asarray(station.get("coords", []), dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 3 or coords.size == 0:
            continue
        all_ids = station.get("all_ids", [f"P{i + 1}" for i in range(len(coords))])
        role_masks = _station_role_masks(all_ids, station)
        style_by_role = {
            "IRRADIATED": ("irradiated", dict(size=1.7, color="#050505", opacity=0.58, symbol="circle")),
            "CP": ("control points", dict(size=4.7, color="#2563eb", opacity=0.9, symbol="diamond", line=dict(color="#0f172a", width=0.7))),
            "ICP": ("ICP check points", dict(size=5.4, color="#f97316", opacity=0.94, symbol="circle", line=dict(color="#111827", width=0.8))),
        }
        for role, mask in role_masks.items():
            if not np.any(mask):
                continue
            label, marker = style_by_role[role]
            role_coords = coords[mask]
            role_ids = [pid for pid, keep in zip(all_ids, mask) if keep]
            fig.add_trace(go.Scatter3d(
                x=role_coords[:, 0],
                y=role_coords[:, 1],
                z=role_coords[:, 2],
                mode="markers",
                name=f"{method_name} | {station_name} | {label}",
                legendgroup=f"{method_name}_{station_name}_{role}",
                marker=marker,
                text=[
                    f"Method: {method_name}<br>Station: {station_name}<br>Point: {pid}"
                    f"<br>Role: {role}"
                    f"<br>Easting/X: {x:.4f} m"
                    f"<br>Northing/Y: {y:.4f} m"
                    f"<br>Elevation/Z: {z:.4f} m"
                    for pid, x, y, z in zip(role_ids, role_coords[:, 0], role_coords[:, 1], role_coords[:, 2])
                ],
                hoverinfo="text",
            ))

        if not include_ellipsoids or ellipsoid_count >= max_ellipsoids:
            continue
        uncertainties = station.get("point_uncertainties", {}) or station.get("point_uncertainties_approx", {})
        if not isinstance(uncertainties, dict):
            continue
        ring_x, ring_y, ring_z, ring_text = [], [], [], []
        axis_x, axis_y, axis_z, axis_text = [], [], [], []
        for pid, coord in zip(all_ids, coords):
            if ellipsoid_count >= max_ellipsoids:
                break
            cov = uncertainties.get(pid)
            if cov is None:
                continue
            geometry = _ellipsoid_wireframe(coord, cov, scale_factor=visual_scale)
            if geometry is None:
                continue
            axes_mm = geometry["confidence_axes"] * 1000.0
            hover = (
                f"Method: {method_name}<br>Station: {station_name}<br>Point: {pid}"
                f"<br>95% semi-axes: {axes_mm[0]:.2f}, {axes_mm[1]:.2f}, {axes_mm[2]:.2f} mm"
                f"<br>Visual scale factor: {visual_scale:.1f}x"
                f"<br>Displayed only for points with covariance"
            )
            for circle in geometry["rings"]:
                ring_x.extend(circle[0].tolist() + [None])
                ring_y.extend(circle[1].tolist() + [None])
                ring_z.extend(circle[2].tolist() + [None])
                ring_text.extend([hover] * circle.shape[1] + [""])
            for axis in geometry["axes"]:
                axis_x.extend(axis[0].tolist() + [None])
                axis_y.extend(axis[1].tolist() + [None])
                axis_z.extend(axis[2].tolist() + [None])
                axis_text.extend([hover] * axis.shape[1] + [""])
            ellipsoid_count += 1
        if ring_x:
            fig.add_trace(go.Scatter3d(
                x=ring_x,
                y=ring_y,
                z=ring_z,
                mode="lines",
                name=f"{method_name} 95% error ellipsoids | visual scale {visual_scale:.0f}x",
                legendgroup=f"{method_name}_ellipsoids",
                showlegend=not ellipsoid_legend_shown,
                line=dict(color="#dc2626", width=2.4),
                opacity=0.6,
                hoverinfo="text",
                text=ring_text,
            ))
            fig.add_trace(go.Scatter3d(
                x=axis_x,
                y=axis_y,
                z=axis_z,
                mode="lines",
                name=f"{method_name} ellipsoid principal axes",
                legendgroup=f"{method_name}_ellipsoids",
                showlegend=False,
                line=dict(color="#111827", width=3.2),
                opacity=0.78,
                hoverinfo="text",
                text=axis_text,
            ))
            ellipsoid_legend_shown = True
    return ellipsoid_count


def _write_3d_figure(fig, html_path, title):
    if len(fig.data) == 0:
        return False
    fig.update_layout(**_plotly_layout(title, legend_title="Method | station"))
    fig.write_html(str(html_path), include_plotlyjs="embed")
    return True


def _manual_plotly_trace(name, x, y, z, mode="markers", text=None, line=None, marker=None, showlegend=True, opacity=None):
    def coord_values(values):
        converted = []
        for value in values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                converted.append(None)
                continue
            converted.append(numeric if math.isfinite(numeric) else None)
        return converted

    trace = {
        "type": "scatter3d",
        "mode": mode,
        "name": name,
        "x": coord_values(x),
        "y": coord_values(y),
        "z": coord_values(z),
        "showlegend": showlegend,
    }
    if text is not None:
        trace["text"] = list(map(str, text))
        trace["hoverinfo"] = "text"
    if line is not None:
        trace["line"] = line
    if marker is not None:
        trace["marker"] = marker
    if opacity is not None:
        trace["opacity"] = opacity
    return trace


def _manual_method_traces(method_name, method_outputs, residual_df=None, include_ellipsoids=True, max_ellipsoids=120):
    traces = []
    ellipsoid_count = 0
    ellipsoid_legend_shown = False
    visual_scale = _error_geometry_visual_scale(method_outputs)
    for station_name, station in _iter_visual_stations(method_outputs):
        coords = np.asarray(station.get("coords", []), dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 3 or coords.size == 0:
            continue
        all_ids = station.get("all_ids", [f"P{i + 1}" for i in range(len(coords))])
        role_masks = _station_role_masks(all_ids, station)
        style_by_role = {
            "IRRADIATED": ("irradiated", {"size": 1.7, "color": "#050505", "opacity": 0.58, "symbol": "circle"}),
            "CP": ("control points", {"size": 4.7, "color": "#2563eb", "opacity": 0.9, "symbol": "diamond", "line": {"color": "#0f172a", "width": 0.7}}),
            "ICP": ("ICP check points", {"size": 5.4, "color": "#f97316", "opacity": 0.94, "symbol": "circle", "line": {"color": "#111827", "width": 0.8}}),
        }
        for role, mask in role_masks.items():
            if not np.any(mask):
                continue
            label, marker = style_by_role[role]
            role_coords = coords[mask]
            role_ids = [pid for pid, keep in zip(all_ids, mask) if keep]
            traces.append(_manual_plotly_trace(
                f"{method_name} | {station_name} | {label}",
                role_coords[:, 0],
                role_coords[:, 1],
                role_coords[:, 2],
                marker=marker,
                text=[
                    f"Method: {method_name}<br>Station: {station_name}<br>Point: {pid}"
                    f"<br>Role: {role}"
                    f"<br>Easting/X: {x:.4f} m"
                    f"<br>Northing/Y: {y:.4f} m"
                    f"<br>Elevation/Z: {z:.4f} m"
                    for pid, x, y, z in zip(role_ids, role_coords[:, 0], role_coords[:, 1], role_coords[:, 2])
                ],
            ))
        if include_ellipsoids and ellipsoid_count < max_ellipsoids:
            uncertainties = station.get("point_uncertainties", {}) or station.get("point_uncertainties_approx", {})
            if isinstance(uncertainties, dict):
                ring_x, ring_y, ring_z, ring_text = [], [], [], []
                axis_x, axis_y, axis_z, axis_text = [], [], [], []
                for pid, coord in zip(all_ids, coords):
                    if ellipsoid_count >= max_ellipsoids:
                        break
                    cov = uncertainties.get(pid)
                    if cov is None:
                        continue
                    geometry = _ellipsoid_wireframe(coord, cov, scale_factor=visual_scale)
                    if geometry is None:
                        continue
                    axes_mm = geometry["confidence_axes"] * 1000.0
                    hover = (
                        f"Method: {method_name}<br>Station: {station_name}<br>Point: {pid}"
                        f"<br>95% semi-axes: {axes_mm[0]:.2f}, {axes_mm[1]:.2f}, {axes_mm[2]:.2f} mm"
                        f"<br>Visual scale factor: {visual_scale:.1f}x"
                        f"<br>Displayed only for points with covariance"
                    )
                    for circle in geometry["rings"]:
                        ring_x.extend(circle[0].tolist() + [None])
                        ring_y.extend(circle[1].tolist() + [None])
                        ring_z.extend(circle[2].tolist() + [None])
                        ring_text.extend([hover] * circle.shape[1] + [""])
                    for axis in geometry["axes"]:
                        axis_x.extend(axis[0].tolist() + [None])
                        axis_y.extend(axis[1].tolist() + [None])
                        axis_z.extend(axis[2].tolist() + [None])
                        axis_text.extend([hover] * axis.shape[1] + [""])
                    ellipsoid_count += 1
                if ring_x:
                    traces.append(_manual_plotly_trace(
                        f"{method_name} 95% error ellipsoids | visual scale {visual_scale:.0f}x",
                        ring_x,
                        ring_y,
                        ring_z,
                        mode="lines",
                        line={"color": "#dc2626", "width": 2.4},
                        showlegend=not ellipsoid_legend_shown,
                        text=ring_text,
                        opacity=0.68,
                    ))
                    traces.append(_manual_plotly_trace(
                        f"{method_name} ellipsoid principal axes",
                        axis_x,
                        axis_y,
                        axis_z,
                        mode="lines",
                        line={"color": "#111827", "width": 3.2},
                        showlegend=False,
                        text=axis_text,
                        opacity=0.82,
                    ))
                    ellipsoid_legend_shown = True

    if residual_df is not None and not residual_df.empty:
        method_res = _residual_rows_for_method(residual_df, method_name)
        required = {"observed_x", "observed_y", "observed_z", "adjusted_x", "adjusted_y", "adjusted_z"}
        if not method_res.empty and required.issubset(method_res.columns):
            for _, row in method_res.dropna(subset=list(required)).head(800).iterrows():
                traces.append(_manual_plotly_trace(
                    f"{method_name} residual vectors",
                    [row["observed_x"], row["adjusted_x"], None],
                    [row["observed_y"], row["adjusted_y"], None],
                    [row["observed_z"], row["adjusted_z"], None],
                    mode="lines",
                    line={"color": "#d62728", "width": 3},
                    showlegend=False,
                    text=[
                        f"{method_name}<br>{row.get('station', '')} {row.get('point', row.get('point_id', ''))}<br>observed",
                        f"adjusted<br>|v|={row.get('v_norm', np.nan)} m",
                        "",
                    ],
                ))
    return traces


def _manual_rmse_traces(residual_df, method_name=None, outputs_by_method=None, independent_df=None):
    df = _rmse_point_cloud_rows(
        outputs_by_method,
        residual_df=residual_df,
        independent_df=independent_df,
        method_name=method_name,
    ) if outputs_by_method else _rmse_residual_rows(residual_df, method_name=method_name)
    if df.empty:
        return []
    traces = []
    colorbar_added = False
    for method, group in df.groupby("method", dropna=False):
        irradiated = group[group["role"].astype(str) == "IRRADIATED"].copy() if "role" in group.columns else pd.DataFrame()
        if not irradiated.empty:
            traces.append(_manual_plotly_trace(
                f"{method} irradiated points",
                irradiated["adjusted_x"],
                irradiated["adjusted_y"],
                irradiated["adjusted_z"],
                marker={
                    "size": 1.6,
                    "color": "#050505",
                    "opacity": 0.62,
                },
                text=[
                    f"Method: {row.get('method', '')}<br>Station: {row.get('station', '')}"
                    f"<br>Point: {row.get('point', row.get('point_id', ''))}"
                    f"<br>Role: Irradiated"
                    f"<br>Easting/X: {float(row.get('adjusted_x', np.nan)):.4f} m"
                    f"<br>Northing/Y: {float(row.get('adjusted_y', np.nan)):.4f} m"
                    f"<br>Elevation/Z: {float(row.get('adjusted_z', np.nan)):.4f} m"
                    for _, row in irradiated.iterrows()
                ],
            ))
        colored = group[group.get("role", pd.Series("", index=group.index)).astype(str).isin({"CP", "ICP"})].copy()
        if colored.empty:
            colored = group.copy()
        colored = colored[np.isfinite(pd.to_numeric(colored["point_rmse_mm"], errors="coerce"))].copy()
        if colored.empty:
            continue
        traces.append(_manual_plotly_trace(
            f"{method} CP/ICP RMSE",
            colored["adjusted_x"],
            colored["adjusted_y"],
            colored["adjusted_z"],
            marker={
                "size": 5.8,
                "color": [float(v) for v in colored["point_rmse_mm"]],
                "colorscale": "Turbo",
                "colorbar": _plotly_left_colorbar("CP/ICP RMSE (mm)") if not colorbar_added else None,
                "showscale": not colorbar_added,
                "opacity": 0.9,
                "line": {"color": "#111827", "width": 0.7},
            },
            text=[
                f"Method: {row.get('method', '')}<br>Station: {row.get('station', '')}"
                f"<br>Point: {row.get('point', row.get('point_id', ''))}"
                f"<br>Role: {row.get('role', 'CP/ICP')}"
                f"<br>Easting/X: {float(row.get('adjusted_x', np.nan)):.4f} m"
                f"<br>Northing/Y: {float(row.get('adjusted_y', np.nan)):.4f} m"
                f"<br>Elevation/Z: {float(row.get('adjusted_z', np.nan)):.4f} m"
                f"<br>RMSE: {row['point_rmse_mm']:.3f} mm"
                for _, row in colored.iterrows()
            ],
        ))
        colorbar_added = True
    return traces


def _write_manual_plotly_html(html_path, title, traces):
    if not traces:
        return False
    layout = _plotly_layout(title, legend_title="Method | station")
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>html,body,#plot{{width:100%;height:100%;margin:0;}}</style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const data = {json.dumps(traces, allow_nan=False)};
    const layout = {json.dumps(layout, allow_nan=False)};
    Plotly.newPlot('plot', data, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")
    return True


def make_plotly_3d_outputs_without_python_plotly(output_dir, outputs_by_method, residual_df=None, selected_method=None, independent_df=None):
    """Generate Plotly HTML directly with plotly.js when the Python package is unavailable."""
    plot_dir = output_dir / "plotly_3d_by_method"
    plot_dir.mkdir(parents=True, exist_ok=True)
    index_lines = ["# Plotly 3D outputs", "", "Generated with plotly.js fallback."]
    combined_traces = []
    write_combined = len(outputs_by_method) > 1
    for method_name, method_outputs in outputs_by_method.items():
        traces = _manual_method_traces(method_name, method_outputs, residual_df, include_ellipsoids=True)
        method_html = plot_dir / f"{safe_key(method_name)}_3d.html"
        if _write_manual_plotly_html(method_html, f"3D network visualization - {method_name}", traces):
            index_lines.append(f"- {method_name}: {method_html.name}")
        method_rmse_html = plot_dir / f"{safe_key(method_name)}_rmse_3d.html"
        method_rmse_traces = _manual_rmse_traces(
            residual_df,
            method_name=method_name,
            outputs_by_method=outputs_by_method,
            independent_df=independent_df,
        )
        if _write_manual_plotly_html(
            method_rmse_html,
            f"3D point RMSE color scale - {method_name}",
            method_rmse_traces,
        ):
            index_lines.append(f"- {method_name} RMSE colors: {method_rmse_html.name}")
        combined_traces.extend(_manual_method_traces(method_name, method_outputs, None, include_ellipsoids=False))

    if write_combined:
        all_html = output_dir / "all_methods_3d.html"
        _write_manual_plotly_html(all_html, "3D network visualization - all methods", combined_traces)

    selected = selected_method if selected_method in outputs_by_method else None
    selected_traces = []
    if selected:
        selected_traces = _manual_method_traces(selected, outputs_by_method[selected], residual_df, include_ellipsoids=True)
    if not selected_traces:
        selected_traces = combined_traces
    _write_manual_plotly_html(output_dir / "selected_method_3d.html", "3D network visualization - selected method", selected_traces)
    if write_combined:
        _write_manual_plotly_html(output_dir / "method_point_clouds.html", "3D network visualization - all methods", combined_traces)

    all_rmse_traces = _manual_rmse_traces(
        residual_df,
        outputs_by_method=outputs_by_method,
        independent_df=independent_df,
    )
    selected_rmse_traces = _manual_rmse_traces(
        residual_df,
        method_name=selected,
        outputs_by_method=outputs_by_method,
        independent_df=independent_df,
    ) if selected else []
    if write_combined and all_rmse_traces:
        _write_manual_plotly_html(output_dir / "all_methods_rmse_3d.html", "3D point RMSE color scale - all methods", all_rmse_traces)
        _write_manual_plotly_html(output_dir / "method_point_clouds_rmse.html", "3D point RMSE color scale - all methods", all_rmse_traces)
    if selected_rmse_traces:
        _write_manual_plotly_html(output_dir / "selected_method_rmse_3d.html", "3D point RMSE color scale - selected method", selected_rmse_traces)

    (output_dir / "plotly_3d_index.md").write_text("\n".join(index_lines), encoding="utf-8")
    (output_dir / "plotly_html_generated_with_js_fallback.txt").write_text(
        "Python plotly was not installed; HTML was generated with plotly.js CDN fallback.\n",
        encoding="utf-8",
    )
    print(f"OK: 3D Plotly JS fallback outputs saved to {output_dir}")


def make_plotly_3d_outputs(output_dir, outputs_by_method, residual_df=None, selected_method=None, independent_df=None):
    """Generate 3D Plotly outputs with point clouds, residual vectors and error ellipsoids."""
    if not PLOTLY_AVAILABLE:
        make_plotly_3d_outputs_without_python_plotly(output_dir, outputs_by_method, residual_df, selected_method, independent_df)
        return

    plot_dir = output_dir / "plotly_3d_by_method"
    plot_dir.mkdir(parents=True, exist_ok=True)
    index_lines = ["# Plotly 3D outputs", ""]
    write_combined = len(outputs_by_method) > 1

    combined = go.Figure()
    for method_name, method_outputs in outputs_by_method.items():
        method_fig = go.Figure()
        ellipsoids = _add_method_to_3d_figure(
            method_fig, method_name, method_outputs, include_ellipsoids=True
        )
        _add_residual_vectors_3d(method_fig, method_name, residual_df)
        method_html = plot_dir / f"{safe_key(method_name)}_3d.html"
        if _write_3d_figure(
            method_fig,
            method_html,
            f"3D network visualization - {method_name} ({ellipsoids} error ellipsoids)",
        ):
            index_lines.append(f"- {method_name}: {method_html.name}")
        method_rmse_html = plot_dir / f"{safe_key(method_name)}_rmse_3d.html"
        if _write_rmse_3d_figure(
            method_rmse_html,
            f"3D point RMSE color scale - {method_name}",
            residual_df,
            method_name=method_name,
            outputs_by_method=outputs_by_method,
            independent_df=independent_df,
        ):
            index_lines.append(f"- {method_name} RMSE colors: {method_rmse_html.name}")

        _add_method_to_3d_figure(
            combined, method_name, method_outputs, include_ellipsoids=False
        )

    if write_combined:
        _add_residual_vectors_3d(combined, selected_method, residual_df) if selected_method else None
        all_html = output_dir / "all_methods_3d.html"
        _write_3d_figure(combined, all_html, "3D network visualization - all methods")

    selected = selected_method if selected_method in outputs_by_method else None
    selected_fig = go.Figure()
    if selected:
        ellipsoids = _add_method_to_3d_figure(
            selected_fig, selected, outputs_by_method[selected], include_ellipsoids=True
        )
        _add_residual_vectors_3d(selected_fig, selected, residual_df)
        title = f"3D network visualization - {selected} ({ellipsoids} error ellipsoids)"
    else:
        _add_method_to_3d_figure(
            selected_fig, next(iter(outputs_by_method)), outputs_by_method[next(iter(outputs_by_method))],
            include_ellipsoids=True
        ) if outputs_by_method else None
        title = "3D network visualization - selected method"
    selected_html = output_dir / "selected_method_3d.html"
    if not _write_3d_figure(selected_fig, selected_html, title):
        _write_3d_figure(combined, selected_html, "3D network visualization - selected method")

    if write_combined:
        method_point_clouds = output_dir / "method_point_clouds.html"
        _write_3d_figure(combined, method_point_clouds, "3D network visualization - all methods")
        _write_rmse_3d_figure(
            output_dir / "all_methods_rmse_3d.html",
            "3D point RMSE color scale - all methods",
            residual_df,
            outputs_by_method=outputs_by_method,
            independent_df=independent_df,
        )
        _write_rmse_3d_figure(
            output_dir / "method_point_clouds_rmse.html",
            "3D point RMSE color scale - all methods",
            residual_df,
            outputs_by_method=outputs_by_method,
            independent_df=independent_df,
        )
    if selected:
        _write_rmse_3d_figure(
            output_dir / "selected_method_rmse_3d.html",
            f"3D point RMSE color scale - {selected}",
            residual_df,
            method_name=selected,
            outputs_by_method=outputs_by_method,
            independent_df=independent_df,
        )

    (output_dir / "plotly_3d_index.md").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"OK: 3D Plotly outputs saved to {output_dir}")


def _add_plotly_xy_error_ellipses(fig, method_name, station_name, all_ids, coords, uncertainties, visual_scale, showlegend=False):
    if not isinstance(uncertainties, dict):
        return 0
    ellipse_x, ellipse_y, ellipse_text = [], [], []
    count = 0
    for pid, coord in zip(all_ids, coords):
        cov = uncertainties.get(pid)
        if cov is None:
            continue
        axes_info = _covariance_axes(cov, 2)
        if axes_info is None:
            continue
        sigmas, _ = axes_info
        ellipse = _ellipse_xy(coord, cov, scale_factor=visual_scale)
        if ellipse is None:
            continue
        axes_mm = sigmas * ERROR_ELLIPSE_95_2D * 1000.0
        hover = (
            f"Method: {method_name}<br>Station: {station_name}<br>Point: {pid}"
            f"<br>95% XY semi-axes: {axes_mm[0]:.2f}, {axes_mm[1]:.2f} mm"
            f"<br>Visual scale: {visual_scale:.1f}x"
        )
        ex, ey = ellipse
        ellipse_x.extend(ex.tolist() + [None])
        ellipse_y.extend(ey.tolist() + [None])
        ellipse_text.extend([hover] * len(ex) + [""])
        count += 1
    if count:
        fig.add_trace(go.Scatter(
            x=ellipse_x,
            y=ellipse_y,
            mode="lines",
            name=f"{method_name} 95% XY error ellipses (scaled {visual_scale:.0f}x)",
            legendgroup=method_name,
            showlegend=showlegend,
            line=dict(width=1.8, color="#dc2626"),
            opacity=0.7,
            hoverinfo="text",
            text=ellipse_text,
        ))
    return count


def make_plotly_point_clouds(output_dir, outputs_by_method):
    """Generate an interactive 2D Plotly point cloud for all methods."""
    if not PLOTLY_AVAILABLE:
        (output_dir / "plotly_html_not_generated.txt").write_text(
            "plotly is not installed in this Python environment.\n",
            encoding="utf-8",
        )
        return

    fig = go.Figure()
    ellipse_count = 0

    for method_name, outputs in outputs_by_method.items():
        ellipse_scale = _error_geometry_visual_scale(
            outputs,
            dimensions=2,
            confidence_sigma=ERROR_ELLIPSE_95_2D,
        )
        ellipse_legend_shown = False
        if "network" in outputs:
            network_result = outputs["network"]
            transformed_stations = network_result.get("transformed_stations", [])

            for station in transformed_stations:
                if "coords" in station and station["coords"] is not None:
                    coords = np.asarray(station["coords"], dtype=float)
                    if coords.ndim == 2 and coords.shape[1] >= 3:
                        station_name = station.get("name", station.get("station", "Unknown"))
                        all_ids = station.get("all_ids", [f"P{i + 1}" for i in range(len(coords))])
                        xs = coords[:, 0]
                        ys = coords[:, 1]

                        fig.add_trace(go.Scatter(
                            x=xs,
                            y=ys,
                            mode="markers",
                            name=f"{method_name} | {station_name}",
                            legendgroup=method_name,
                            marker=dict(size=6, opacity=0.75),
                            text=[
                                f"{station_name} - {pid}"
                                f"<br>Easting/X: {x:.3f} m"
                                f"<br>Northing/Y: {y:.3f} m"
                                f"<br>Elevation/Z: {z:.3f} m"
                                for pid, x, y, z in zip(all_ids, coords[:, 0], coords[:, 1], coords[:, 2])
                            ],
                            hoverinfo="text",
                        ))

                        uncertainties = station.get("point_uncertainties", {}) or station.get("point_uncertainties_approx", {})
                        added = _add_plotly_xy_error_ellipses(
                            fig,
                            method_name,
                            station_name,
                            all_ids,
                            coords,
                            uncertainties,
                            ellipse_scale,
                            showlegend=not ellipse_legend_shown,
                        )
                        ellipse_count += added
                        ellipse_legend_shown = ellipse_legend_shown or bool(added)

        for station_name, station_outputs in outputs.items():
            if station_name == "network":
                continue
            if isinstance(station_outputs, dict) and "coords" in station_outputs:
                coords = np.asarray(station_outputs.get("coords", []), dtype=float)
                if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] >= 3:
                    all_ids = station_outputs.get("all_ids", [f"P{i + 1}" for i in range(len(coords))])
                    xs = coords[:, 0]
                    ys = coords[:, 1]

                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        name=f"{method_name} | {station_name}",
                        legendgroup=method_name,
                        marker=dict(size=6, opacity=0.75),
                            text=[
                            f"{station_name} - {pid}"
                            f"<br>Easting/X: {x:.3f} m"
                            f"<br>Northing/Y: {y:.3f} m"
                            f"<br>Elevation/Z: {z:.3f} m"
                            for pid, x, y, z in zip(all_ids, coords[:, 0], coords[:, 1], coords[:, 2])
                        ],
                        hoverinfo="text",
                    ))
                    uncertainties = station_outputs.get("point_uncertainties", {}) or station_outputs.get("point_uncertainties_approx", {})
                    added = _add_plotly_xy_error_ellipses(
                        fig,
                        method_name,
                        station_name,
                        all_ids,
                        coords,
                        uncertainties,
                        ellipse_scale,
                        showlegend=not ellipse_legend_shown,
                    )
                    ellipse_count += added
                    ellipse_legend_shown = ellipse_legend_shown or bool(added)

    if len(fig.data) > 0:
        fig.update_layout(
            title=f"Point Clouds by Method (XY){' with error ellipses' if ellipse_count else ''}",
            xaxis_title="Easting / X coordinate (m)",
            yaxis_title="Northing / Y coordinate (m)",
            legend_title="Method | station",
            hovermode="closest",
            margin=dict(l=60, r=30, b=60, t=60),
            template="plotly_white",
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        html_path = output_dir / "method_point_clouds.html"
        fig.write_html(str(html_path), include_plotlyjs="embed")
        print(f"OK: Plotly point cloud saved to {html_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "DATA"))
    parser.add_argument("--out", default=str(ROOT / "results" / "method_audit_DATA"))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            *[key for key, _ in SEQUENTIAL_METHODS],
            *[key for key, _ in NETWORK_METHODS],
        ],
        help="Processing methods to execute. Use 'all' or the new article acronyms.",
    )
    parser.add_argument(
        "--selected-plot-method",
        default=None,
        help="Optional method key to use for selected_method_3d.html when --methods all is used.",
    )
    parser.add_argument(
        "--control-points",
        nargs="+",
        default=None,
        help="Optional list of CP IDs to use. Example: --control-points CP1 CP2 CP3 CP5",
    )
    parser.add_argument(
        "--independent-check-points",
        nargs="+",
        default=[],
        help="Point IDs reserved for independent check RMSE. They are not used as processing controls.",
    )
    parser.add_argument("--min-common-cp", type=int, default=4)
    parser.add_argument("--fixed-scale", action="store_true", help="Force scale factor to 1.0.")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--weight-constraint", type=float, default=1e5)
    parser.add_argument(
        "--apriori-sigma-mm",
        type=float,
        default=3.0,
        help=(
            "A priori standard deviation in millimetres for the global chi-square "
            "variance-factor test. This does not alter the unit weight matrix P=I."
        ),
    )
    parser.add_argument(
        "--use-quaternion-constraint",
        dest="use_quaternion_constraint",
        action="store_true",
        default=True,
        help="Use the q0^2+q1^2+q2^2+q3^2=1 pseudo-observation in quaternion LM.",
    )
    parser.add_argument(
        "--no-quaternion-constraint",
        dest="use_quaternion_constraint",
        action="store_false",
        help="Disable the quaternion norm pseudo-observation in quaternion LM.",
    )
    parser.add_argument("--skip-network", action="store_true")
    parser.add_argument(
        "--outlier-methods",
        nargs="+",
        default=["all"],
        choices=["all", "none", *OUTLIER_METHOD_CHOICES],
        help=(
            "Optional outlier detectors for this run. Use 'all', 'none', "
            "or any combination: baarda mad."
        ),
    )
    parser.add_argument("--outlier-sigma", type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument("--outlier-mad", type=float, default=3.5)
    parser.add_argument("--outlier-mad-min-observations", type=int, default=8)
    parser.add_argument("--outlier-percentile", type=float, default=95.0, help=argparse.SUPPRESS)
    parser.add_argument("--outlier-rms-factor", type=float, default=2.5, help=argparse.SUPPRESS)
    parser.add_argument("--outlier-component-sigma", type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument("--outlier-min-group", type=int, default=4)
    parser.add_argument("--outlier-removal-iterations", type=int, default=20)
    parser.add_argument(
        "--outlier-max-remove-per-iteration",
        type=int,
        default=1,
        help=(
            "Maximum number of station-point observations removed at each "
            "outlier-removal iteration. Default 1 performs sequential removal."
        ),
    )
    parser.add_argument("--outlier-max-condition-number", type=float, default=1e12)
    parser.add_argument("--outlier-numerical-epsilon", type=float, default=1e-12)
    parser.add_argument(
        "--outlier-mode",
        choices=["automatic", "review"],
        default="automatic",
        help="automatic removes admissible candidates; review only reports candidates.",
    )
    parser.add_argument(
        "--outlier-action",
        choices=["report", "remove"],
        default="report",
        help="report only, or remove detected outliers and rerun selected methods.",
    )
    parser.add_argument(
        "--manual-exclusions",
        nargs="*",
        default=[],
        help=(
            "Station-point observations to exclude before reprocessing. "
            "Use STATION:POINT or METHOD:STATION:POINT."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    stations = load_station_data(args.data)
    apply_independent_check_points(stations, args.independent_check_points)
    all_station_rows = []
    all_residual_rows = []
    all_independent_rows = []
    outputs_by_method = defaultdict(dict)
    sequential_methods, network_methods = selected_methods(args)
    manual_exclusions = parse_manual_exclusion_specs(args.manual_exclusions)

    for method_key, method_class in sequential_methods:
        result = run_sequential_method(method_key, method_class, stations, args)
        all_station_rows.extend(result["station_rows"])
        all_residual_rows.extend(result["residual_rows"])
        all_independent_rows.extend(result.get("independent_rows", []))
        outputs_by_method[method_key].update(result["outputs"])

    for method_key, method_class in network_methods:
        result = run_network_method(method_key, method_class, stations, args)
        all_station_rows.extend(result["station_rows"])
        all_residual_rows.extend(result["residual_rows"])
        all_independent_rows.extend(result.get("independent_rows", []))
        outputs_by_method[method_key].update(result["outputs"])

    station_df = pd.DataFrame(all_station_rows)
    residual_df = pd.DataFrame(all_residual_rows)
    outlier_df, outlier_summary_df = evaluate_outliers(residual_df, args)
    outlier_removal_trace_rows = []
    outlier_stop_condition = (
        "USER_DISABLED_AUTOMATIC_EXCLUSION"
        if args.outlier_action != "remove" or args.outlier_mode == "review"
        else ""
    )
    corrected_solution_status = "CORRECTED_SOLUTION_NOT_AVAILABLE"

    if args.outlier_action == "remove" and manual_exclusions:
        run_label = "outliers_removed_manual"
        iteration_rmse_before = residual_global_rmse(residual_df)
        current_residual_rows = []
        current_independent_rows = []

        for method_key, method_class in sequential_methods:
            result = run_sequential_method(
                method_key,
                method_class,
                stations,
                args,
                excluded_points=manual_exclusions,
                run_label=run_label,
            )
            all_station_rows.extend(result["station_rows"])
            all_residual_rows.extend(result["residual_rows"])
            all_independent_rows.extend(result.get("independent_rows", []))
            current_residual_rows.extend(result["residual_rows"])
            current_independent_rows.extend(result.get("independent_rows", []))
            outputs_by_method[f"{method_key}_{run_label}"].update(result["outputs"])

        for method_key, method_class in network_methods:
            result = run_network_method(
                method_key,
                method_class,
                stations,
                args,
                excluded_points=manual_exclusions,
                run_label=run_label,
            )
            all_station_rows.extend(result["station_rows"])
            all_residual_rows.extend(result["residual_rows"])
            all_independent_rows.extend(result.get("independent_rows", []))
            current_residual_rows.extend(result["residual_rows"])
            current_independent_rows.extend(result.get("independent_rows", []))
            outputs_by_method[f"{method_key}_{run_label}"].update(result["outputs"])

        current_residual_df = pd.DataFrame(current_residual_rows)
        iteration_rmse_after = residual_global_rmse(current_residual_df)
        current_outlier_df, _ = evaluate_outliers(current_residual_df, args)
        rank_after = np.nan
        condition_after = np.nan
        if not current_outlier_df.empty:
            if "rank_before" in current_outlier_df:
                ranks = pd.to_numeric(current_outlier_df["rank_before"], errors="coerce").dropna()
                rank_after = float(ranks.max()) if not ranks.empty else np.nan
            if "condition_number_before" in current_outlier_df:
                conditions = pd.to_numeric(current_outlier_df["condition_number_before"], errors="coerce").dropna()
                condition_after = float(conditions.max()) if not conditions.empty else np.nan

        selected_method_keys = [key for key, _ in sequential_methods] + [key for key, _ in network_methods]
        selected_modes = []
        if sequential_methods:
            selected_modes.append("sequential")
        if network_methods:
            selected_modes.append("network")
        total_manual_exclusions = sum(len(points) for points in manual_exclusions.values())

        for (scope_method, station), points in sorted(manual_exclusions.items()):
            methods_triggered = (
                ",".join(selected_method_keys)
                if scope_method == "*"
                else str(scope_method)
            )
            modes_triggered = ",".join(selected_modes)
            for point in sorted(points):
                source_row = {}
                if not residual_df.empty and {"station", "point"}.issubset(residual_df.columns):
                    mask = (
                        residual_df["station"].astype(str).eq(str(station))
                        & residual_df["point"].astype(str).eq(str(point))
                    )
                    if scope_method != "*" and "method" in residual_df.columns:
                        mask &= residual_df["method"].astype(str).eq(str(scope_method))
                    matches = residual_df[mask]
                    if not matches.empty:
                        source_row = matches.iloc[0].to_dict()
                flag_row = {}
                if not outlier_df.empty and {"station", "point"}.issubset(outlier_df.columns):
                    flag_mask = (
                        outlier_df["station"].astype(str).eq(str(station))
                        & outlier_df["point"].astype(str).eq(str(point))
                    )
                    if scope_method != "*" and "method" in outlier_df.columns:
                        flag_mask &= outlier_df["method"].astype(str).eq(str(scope_method))
                    flag_matches = outlier_df[flag_mask]
                    if not flag_matches.empty:
                        flag_row = flag_matches.iloc[0].to_dict()

                entry = {
                    "iteration": 1,
                    "run_label": run_label,
                    "stop_condition": "MANUAL_EXCLUSIONS_APPLIED" if current_residual_rows else "ADJUSTMENT_FAILED",
                    "solution_status": "CORRECTED_SOLUTION_ACCEPTED" if current_residual_rows else "CORRECTED_SOLUTION_NOT_AVAILABLE",
                    "station": station,
                    "point": point,
                    "exclusion_scope": scope_method,
                    "methods_triggered": methods_triggered,
                    "modes_triggered": modes_triggered,
                    "source_run_labels": str(source_row.get("run_label", "initial")),
                    "detection_class": "USER_SELECTED",
                    "n_outlier_flags": flag_row.get("n_outlier_flags", np.nan),
                    "outlier_score": flag_row.get("outlier_score", np.nan),
                    "baarda_score": flag_row.get("baarda_score", source_row.get("baarda_score", np.nan)),
                    "mad_score": flag_row.get("mad_score", np.nan),
                    "w_x": flag_row.get("w_x", source_row.get("w_x", np.nan)),
                    "w_y": flag_row.get("w_y", source_row.get("w_y", np.nan)),
                    "w_z": flag_row.get("w_z", source_row.get("w_z", np.nan)),
                    "critical_component": flag_row.get("critical_component", source_row.get("critical_component", "")),
                    "v_norm": source_row.get("v_norm", np.nan),
                    "v_x_before": source_row.get("vx", np.nan),
                    "v_y_before": source_row.get("vy", np.nan),
                    "v_z_before": source_row.get("vz", np.nan),
                    "residual_norm_before": source_row.get("v_norm", np.nan),
                    "exclusion_reason": "User selected residual row in Results tab",
                    "controls_remaining_at_station": np.nan,
                    "rank_before": flag_row.get("rank_before", np.nan),
                    "rank_after": finite_float(rank_after),
                    "condition_number_before": flag_row.get("condition_number_before", np.nan),
                    "condition_number_after": finite_float(condition_after),
                    "baarda_available": flag_row.get("baarda_available", source_row.get("baarda_available", np.nan)),
                    "baarda_unavailable_reason": flag_row.get("baarda_unavailable_reason", source_row.get("baarda_unavailable_reason", "")),
                    "mad_reliable": flag_row.get("mad_reliable", np.nan),
                    "mad_warning": flag_row.get("mad_warning", ""),
                    "mad_degenerate": flag_row.get("mad_degenerate", np.nan),
                    "available_points_before": np.nan,
                    "minimum_points_required": args.min_common_cp,
                    "available_points_after": np.nan,
                    "new_exclusions_this_iteration": total_manual_exclusions,
                    "cumulative_removed_after_selection": total_manual_exclusions,
                    "residual_rows_before": int(len(residual_df)),
                    "residual_rows_after": int(len(current_residual_df)),
                    "global_rmse_before": iteration_rmse_before,
                    "global_rmse_after": iteration_rmse_after,
                }
                if np.isfinite(iteration_rmse_before) and np.isfinite(iteration_rmse_after):
                    entry["global_rmse_change"] = iteration_rmse_before - iteration_rmse_after
                    entry["global_rmse_improvement_pct"] = (
                        (iteration_rmse_before - iteration_rmse_after) / iteration_rmse_before * 100.0
                        if iteration_rmse_before != 0
                        else np.nan
                    )
                outlier_removal_trace_rows.append(entry)

        outlier_stop_condition = "MANUAL_EXCLUSIONS_APPLIED" if current_residual_rows else "ADJUSTMENT_FAILED"
        corrected_solution_status = (
            "CORRECTED_SOLUTION_ACCEPTED"
            if current_residual_rows
            else "CORRECTED_SOLUTION_NOT_AVAILABLE"
        )

    elif args.outlier_action == "remove" and args.outlier_mode == "automatic":
        if outlier_df.empty:
            outlier_stop_condition = "NO_FLAGGED_OBSERVATIONS"
        cumulative_exclusions = defaultdict(set)
        current_residual_rows = list(all_residual_rows)
        max_iterations = max(1, int(args.outlier_removal_iterations or 1))

        for iteration in range(max_iterations):
            iteration_residual_df = pd.DataFrame(current_residual_rows)
            iteration_outlier_df, _ = evaluate_outliers(iteration_residual_df, args)
            iteration_rmse_before = residual_global_rmse(iteration_residual_df)
            if iteration_outlier_df.empty or not bool(iteration_outlier_df.get("is_outlier_candidate", pd.Series(dtype=bool)).any()):
                outlier_stop_condition = "NO_FLAGGED_OBSERVATIONS"
                break
            selected_exclusions, selected_rows = select_outlier_exclusions(
                iteration_outlier_df,
                args.min_common_cp,
                args.outlier_max_remove_per_iteration,
                candidate_validator=lambda station, point: network_exclusion_preserves_connectivity(
                    stations,
                    cumulative_exclusions,
                    station,
                    point,
                    args.min_common_cp,
                ),
            )
            additions = merge_outlier_exclusions(
                cumulative_exclusions,
                selected_exclusions,
            )
            if additions == 0:
                outlier_stop_condition = "ALL_CANDIDATES_REJECTED_BY_GEOMETRY"
                break

            run_label = "outliers_removed" if iteration == 0 else f"outliers_removed_iter{iteration + 1}"
            trace_start = len(outlier_removal_trace_rows)
            cumulative_removed = sum(len(point_ids) for point_ids in cumulative_exclusions.values())
            for selected in selected_rows:
                entry = dict(selected)
                entry.update({
                    "iteration": iteration + 1,
                    "run_label": run_label,
                    "stop_condition": "",
                    "solution_status": "",
                    "new_exclusions_this_iteration": additions,
                    "cumulative_removed_after_selection": cumulative_removed,
                    "residual_rows_before": int(len(iteration_residual_df)),
                    "global_rmse_before": iteration_rmse_before,
                })
                outlier_removal_trace_rows.append(entry)
            current_residual_rows = []
            for method_key, method_class in sequential_methods:
                result = run_sequential_method(
                    method_key,
                    method_class,
                    stations,
                    args,
                    excluded_points=cumulative_exclusions,
                    run_label=run_label,
                )
                all_station_rows.extend(result["station_rows"])
                all_residual_rows.extend(result["residual_rows"])
                all_independent_rows.extend(result.get("independent_rows", []))
                current_residual_rows.extend(result["residual_rows"])
                outputs_by_method[f"{method_key}_{run_label}"].update(result["outputs"])
            for method_key, method_class in network_methods:
                result = run_network_method(
                    method_key,
                    method_class,
                    stations,
                    args,
                    excluded_points=cumulative_exclusions,
                    run_label=run_label,
                )
                all_station_rows.extend(result["station_rows"])
                all_residual_rows.extend(result["residual_rows"])
                all_independent_rows.extend(result.get("independent_rows", []))
                current_residual_rows.extend(result["residual_rows"])
                outputs_by_method[f"{method_key}_{run_label}"].update(result["outputs"])

            if not current_residual_rows:
                outlier_stop_condition = "ADJUSTMENT_FAILED"
                break
            current_residual_df = pd.DataFrame(current_residual_rows)
            iteration_rmse_after = residual_global_rmse(current_residual_df)
            current_outlier_df, _ = evaluate_outliers(current_residual_df, args)
            rank_after = np.nan
            condition_after = np.nan
            if not current_outlier_df.empty:
                if "rank_before" in current_outlier_df:
                    ranks = pd.to_numeric(current_outlier_df["rank_before"], errors="coerce").dropna()
                    rank_after = float(ranks.max()) if not ranks.empty else np.nan
                if "condition_number_before" in current_outlier_df:
                    conditions = pd.to_numeric(current_outlier_df["condition_number_before"], errors="coerce").dropna()
                    condition_after = float(conditions.max()) if not conditions.empty else np.nan
            for entry in outlier_removal_trace_rows[trace_start:]:
                entry["residual_rows_after"] = int(len(current_residual_df))
                entry["global_rmse_after"] = iteration_rmse_after
                entry["rank_after"] = finite_float(rank_after)
                entry["condition_number_after"] = finite_float(condition_after)
                if np.isfinite(iteration_rmse_before) and np.isfinite(iteration_rmse_after):
                    entry["global_rmse_change"] = iteration_rmse_before - iteration_rmse_after
                    entry["global_rmse_improvement_pct"] = (
                        (iteration_rmse_before - iteration_rmse_after) / iteration_rmse_before * 100.0
                        if iteration_rmse_before != 0
                        else np.nan
                    )
        else:
            outlier_stop_condition = "MAXIMUM_CYCLES_REACHED"

        if outlier_removal_trace_rows:
            if outlier_stop_condition in {"ADJUSTMENT_FAILED", "NETWORK_DISCONNECTED", "INSUFFICIENT_MATRIX_RANK"}:
                corrected_solution_status = "CORRECTED_SOLUTION_NOT_AVAILABLE"
            elif outlier_stop_condition == "MAXIMUM_CYCLES_REACHED":
                corrected_solution_status = "CORRECTED_SOLUTION_WITH_WARNINGS"
            else:
                corrected_solution_status = "CORRECTED_SOLUTION_ACCEPTED"
            for entry in outlier_removal_trace_rows:
                entry["stop_condition"] = outlier_stop_condition
                entry["solution_status"] = corrected_solution_status

    station_df = pd.DataFrame(all_station_rows)
    residual_df = pd.DataFrame(all_residual_rows)
    independent_df = pd.DataFrame(all_independent_rows)
    outlier_df, outlier_summary_df = evaluate_outliers(residual_df, args)
    outlier_trace_df = outlier_trace_dataframe(outlier_removal_trace_rows)
    matrices_df = pd.DataFrame(matrix_inventory(outputs_by_method))
    variables_df = pd.DataFrame(variable_inventory(outputs_by_method))

    station_df.to_csv(output_dir / "station_outputs.csv", index=False)
    residual_df.to_csv(output_dir / "residuals.csv", index=False)
    independent_df.to_csv(output_dir / "independent_check_points.csv", index=False)
    matrices_df.to_csv(output_dir / "matrix_inventory.csv", index=False)
    variables_df.to_csv(output_dir / "variable_inventory.csv", index=False)
    outlier_df.to_csv(output_dir / "outlier_flags.csv", index=False)
    outlier_summary_df.to_csv(output_dir / "outlier_summary.csv", index=False)
    outlier_trace_df.to_csv(output_dir / "outlier_removal_trace.csv", index=False)
    save_full_numeric_outputs(output_dir, outputs_by_method)
    write_method_notes(output_dir)
    write_outlier_catalog(output_dir, args)
    write_outlier_diagnostics(output_dir, residual_df, outlier_df, outlier_summary_df, outlier_trace_df, args)
    write_run_configuration(output_dir, args)
    write_article_quality_outputs(output_dir, outputs_by_method, station_df, residual_df)
    write_accuracy_quality_outputs(output_dir, outputs_by_method)
    write_initial_corrected_outlier_package(
        output_dir,
        station_df,
        residual_df,
        independent_df,
        outlier_df,
        outlier_summary_df,
        outlier_trace_df,
        corrected_solution_status,
        outlier_stop_condition,
    )

    method_summary = []
    summary_group_cols = ["method", "run_label"] if "run_label" in station_df.columns else ["method"]
    for keys, group in station_df.groupby(summary_group_cols, dropna=False):
        if len(summary_group_cols) == 2:
            method_key, run_label = keys
        else:
            method_key = keys
            run_label = ""
        ok = group[group["success"] == True]  # noqa: E712
        if not outlier_df.empty:
            outlier_mask = outlier_df["method"] == method_key
            if "run_label" in outlier_df.columns:
                outlier_mask &= outlier_df["run_label"].astype(str) == str(run_label)
            outlier_candidates = int(outlier_df[outlier_mask]["is_outlier_candidate"].sum())
        else:
            outlier_candidates = 0
        method_summary.append({
            "method": method_key,
            "run_label": run_label,
            "mode": ", ".join(sorted(group["mode"].dropna().unique())),
            "stations_attempted": len(group),
            "stations_success": len(ok),
            "stations_failed": len(group) - len(ok),
            "mean_rmse": ok["rmse"].mean() if "rmse" in ok else np.nan,
            "max_rmse": ok["rmse"].max() if "rmse" in ok else np.nan,
            "mean_residual": ok["mean_residual"].mean() if "mean_residual" in ok else np.nan,
            "max_residual": ok["max_residual"].max() if "max_residual" in ok else np.nan,
            "icp_points": int(ok["n_icp_points"].sum()) if "n_icp_points" in ok else 0,
            "icp_rmse": float(np.sqrt(np.nanmean(ok["icp_rmse"].astype(float) ** 2))) if "icp_rmse" in ok and ok["icp_rmse"].notna().any() else np.nan,
            "icp_max_residual": ok["icp_max_residual"].max() if "icp_max_residual" in ok else np.nan,
            "total_time": ok["elapsed_time_total"].sum() if "elapsed_time_total" in ok else np.nan,
            "sigma0_mean": ok["sigma0"].mean() if "sigma0" in ok else np.nan,
            "outlier_candidates": outlier_candidates,
        })
    pd.DataFrame(method_summary).to_csv(output_dir / "method_summary.csv", index=False)

    with (output_dir / "outputs_full_summary.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(outputs_by_method), f, indent=2)

    make_plots(output_dir, station_df, residual_df, outlier_df)
    selected_for_3d = args.selected_plot_method or (
        args.methods[0] if len(args.methods) == 1 and args.methods[0] != "all" else None
    )
    if args.outlier_action == "remove" and selected_for_3d:
        filtered_keys = [
            key for key in outputs_by_method
            if key.startswith(f"{selected_for_3d}_outliers_removed")
        ]
        if filtered_keys:
            selected_for_3d = filtered_keys[-1]
    make_plotly_3d_outputs(output_dir, outputs_by_method, residual_df, selected_for_3d, independent_df)

    print(f"OK: exported method audit to {output_dir}")
    print(f"Stations: {len(stations)}")
    print(f"Rows: station_outputs={len(station_df)}, residuals={len(residual_df)}")
    print(f"Outlier candidates: {int(outlier_df['is_outlier_candidate'].sum()) if not outlier_df.empty else 0}")


if __name__ == "__main__":
    main()
