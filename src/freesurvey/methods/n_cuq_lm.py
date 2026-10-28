"""N-CUQ-LM: fully coupled network unit-quaternion LM adjustment."""

from __future__ import annotations

import math
import time

import numpy as np

from freesurvey.Registration_Module import (
    Method3_GlobalQuaternionsLM_ObsEquations,
    Schut1973_SpaceTime,
)


class N_CUQ_LM(Method3_GlobalQuaternionsLM_ObsEquations):
    method_code = "N-CUQ-LM"
    article_name = "Network-constrained unit-quaternion Levenberg-Marquardt adjustment"
    is_fully_coupled_network = True

    @staticmethod
    def _matrix_to_quaternion(rotation):
        matrix = np.asarray(rotation, dtype=float)
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            quaternion = np.array([
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ])
        else:
            diagonal = np.diag(matrix)
            index = int(np.argmax(diagonal))
            if index == 0:
                scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
                quaternion = np.array([
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ])
            elif index == 1:
                scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
                quaternion = np.array([
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ])
            else:
                scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
                quaternion = np.array([
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ])
        quaternion /= np.linalg.norm(quaternion)
        if quaternion[0] < 0:
            quaternion *= -1.0
        return quaternion

    @classmethod
    def _initial_station_parameters(cls, source_matrices, target_matrices):
        parameters = []
        for source, target in zip(source_matrices, target_matrices):
            scale, rotation, translation, _ = Schut1973_SpaceTime.solve(source, target)
            quaternion = cls._matrix_to_quaternion(rotation)
            parameters.append(
                np.concatenate(
                    ([float(scale)], quaternion, np.asarray(translation, dtype=float))
                )
            )
        return np.asarray(parameters, dtype=float)

    @staticmethod
    def _point_order(point_ids):
        order = []
        seen = set()
        for station_ids in point_ids:
            for point_id in station_ids:
                key = str(point_id)
                if key not in seen:
                    seen.add(key)
                    order.append(key)
        return order

    @classmethod
    def _initial_point_coordinates(
        cls,
        source_matrices,
        target_matrices,
        point_ids,
        station_parameters,
        point_order,
        initial_point_coordinates,
    ):
        accumulators = {point_id: [] for point_id in point_order}
        supplied = initial_point_coordinates or {}
        for station_index, station_ids in enumerate(point_ids):
            params = station_parameters[station_index]
            scale = params[0]
            rotation = cls._quat_to_matrix(params[1:5])
            translation = params[5:8]
            for local_index, point_id in enumerate(station_ids):
                key = str(point_id)
                if key in supplied:
                    accumulators[key].append(np.asarray(supplied[key], dtype=float))
                if local_index < len(target_matrices[station_index]):
                    accumulators[key].append(
                        np.asarray(target_matrices[station_index][local_index], dtype=float)
                    )
                accumulators[key].append(
                    scale
                    * (rotation @ np.asarray(source_matrices[station_index][local_index], dtype=float))
                    + translation
                )
        return np.asarray(
            [
                np.mean(np.vstack(accumulators[point_id]), axis=0)
                for point_id in point_order
            ],
            dtype=float,
        )

    @classmethod
    def _build_coupled_system(
        cls,
        source_matrices,
        point_ids,
        active_masks,
        station_parameters,
        point_coordinates,
        point_offsets,
        free_station_offsets,
        use_quaternion_constraint,
        weight_constraint,
    ):
        n_unknowns = len(free_station_offsets) * 8 + len(point_offsets) * 3
        physical_rows = []
        physical_misclosures = []
        coordinate_meta = []
        point_meta = []

        for station_index, (source, station_ids, active_mask) in enumerate(
            zip(source_matrices, point_ids, active_masks)
        ):
            params = station_parameters[station_index]
            scale = params[0]
            quaternion = params[1:5]
            translation = params[5:8]
            for local_index, point_id in enumerate(station_ids):
                if not active_mask[local_index]:
                    continue
                key = str(point_id)
                point_index = point_offsets[key]
                target = point_coordinates[point_index]
                residual, station_jacobian = cls._compute_residuals_and_jacobian_for_point(
                    np.asarray(source[local_index], dtype=float),
                    target,
                    scale,
                    quaternion,
                    translation,
                )
                row = np.zeros((3, n_unknowns), dtype=float)
                if station_index in free_station_offsets:
                    station_offset = free_station_offsets[station_index]
                    row[:, station_offset:station_offset + 8] = station_jacobian
                point_offset = len(free_station_offsets) * 8 + point_index * 3
                row[:, point_offset:point_offset + 3] = -np.eye(3)
                physical_rows.append(row)
                physical_misclosures.append(residual)
                point_meta.append(
                    {
                        "station_index": station_index,
                        "point_index": local_index,
                        "point_id": key,
                        "v_norm": float(np.linalg.norm(residual)),
                    }
                )
                for component in range(3):
                    coordinate_meta.append(
                        {
                            "station_index": station_index,
                            "point_index": local_index,
                            "point_id": key,
                            "component": component,
                        }
                    )

        A_physical = (
            np.vstack(physical_rows)
            if physical_rows
            else np.zeros((0, n_unknowns), dtype=float)
        )
        L_physical = (
            np.concatenate(physical_misclosures)
            if physical_misclosures
            else np.zeros(0, dtype=float)
        )
        all_rows = list(physical_rows)
        all_misclosures = list(physical_misclosures)

        if use_quaternion_constraint:
            for station_index, station_offset in free_station_offsets.items():
                quaternion = station_parameters[station_index, 1:5]
                row = np.zeros((1, n_unknowns), dtype=float)
                row[0, station_offset + 1:station_offset + 5] = (
                    2.0 * quaternion * weight_constraint
                )
                all_rows.append(row)
                all_misclosures.append(
                    np.asarray(
                        [(1.0 - float(quaternion @ quaternion)) * weight_constraint],
                        dtype=float,
                    )
                )

        A = np.vstack(all_rows) if all_rows else A_physical.copy()
        L = np.concatenate(all_misclosures) if all_misclosures else L_physical.copy()
        return A, L, A_physical, L_physical, coordinate_meta, point_meta

    @staticmethod
    def _removal_preserves_coupled_network(
        point_ids,
        active_masks,
        candidate_station,
        candidate_local_index,
        min_points_per_station,
    ):
        retained_by_station = []
        for station_index, (station_ids, active_mask) in enumerate(
            zip(point_ids, active_masks)
        ):
            retained = {
                str(point_id)
                for local_index, (point_id, is_active) in enumerate(
                    zip(station_ids, active_mask)
                )
                if is_active
                and not (
                    station_index == candidate_station
                    and local_index == candidate_local_index
                )
            }
            retained_by_station.append(retained)

        point_counts = {}
        for retained in retained_by_station:
            for point_id in retained:
                point_counts[point_id] = point_counts.get(point_id, 0) + 1
        network_points = {
            point_id for point_id, count in point_counts.items() if count >= 2
        }
        station_network_points = [
            retained.intersection(network_points)
            for retained in retained_by_station
        ]
        minimum = max(1, int(min_points_per_station))
        if any(len(points) < minimum for points in station_network_points):
            return False

        connected = {0}
        pending = set(range(1, len(station_network_points)))
        while pending:
            newly_connected = {
                station_index
                for station_index in pending
                if any(
                    station_network_points[station_index].intersection(
                        station_network_points[connected_index]
                    )
                    for connected_index in connected
                )
            }
            if not newly_connected:
                return False
            connected.update(newly_connected)
            pending.difference_update(newly_connected)
        return True

    @staticmethod
    def _select_coupled_outlier(
        A_physical,
        L_physical,
        coordinate_meta,
        point_meta,
        point_ids,
        active_masks,
        threshold,
        min_points_per_station,
    ):
        if not point_meta or A_physical.size == 0:
            return None
        scores = {}
        test_name = "baarda"
        try:
            dof = A_physical.shape[0] - np.linalg.matrix_rank(A_physical)
            if dof > 0:
                qxx = np.linalg.pinv(A_physical.T @ A_physical)
                qvv = np.eye(A_physical.shape[0]) - A_physical @ qxx @ A_physical.T
                qvv_diag = np.diag(qvv)
                sigma0 = math.sqrt(float(np.sum(L_physical**2) / dof))
                valid = np.isfinite(qvv_diag) & (qvv_diag > 0) & (sigma0 > 0)
                standardized = np.zeros_like(L_physical, dtype=float)
                standardized[valid] = (
                    np.abs(L_physical[valid])
                    / (sigma0 * np.sqrt(qvv_diag[valid]))
                )
                for index, meta in enumerate(coordinate_meta):
                    key = (int(meta["station_index"]), int(meta["point_index"]))
                    scores[key] = max(scores.get(key, 0.0), float(standardized[index]))
        except Exception:
            scores = {}

        if not scores:
            test_name = "mad"
            norms = np.asarray([row["v_norm"] for row in point_meta], dtype=float)
            finite = norms[np.isfinite(norms)]
            if finite.size < 3:
                return None
            median = float(np.median(finite))
            mad = float(np.median(np.abs(finite - median)))
            robust_sigma = 1.4826 * mad if mad > 0 else float(np.std(finite))
            if robust_sigma <= 0 or not np.isfinite(robust_sigma):
                return None
            for row in point_meta:
                key = (int(row["station_index"]), int(row["point_index"]))
                scores[key] = abs(float(row["v_norm"]) - median) / robust_sigma

        candidates = []
        for row in point_meta:
            station_index = int(row["station_index"])
            local_index = int(row["point_index"])
            score = scores.get((station_index, local_index), 0.0)
            if not np.isfinite(score) or score <= float(threshold):
                continue
            active_count = int(np.sum(active_masks[station_index]))
            if active_count - 1 < int(min_points_per_station):
                continue
            if not N_CUQ_LM._removal_preserves_coupled_network(
                point_ids,
                active_masks,
                station_index,
                local_index,
                min_points_per_station,
            ):
                continue
            candidates.append(
                {
                    "station_index": station_index,
                    "point_index": local_index,
                    "point_id": row["point_id"],
                    "test": test_name,
                    "statistic": float(score),
                    "threshold": float(threshold),
                    "v_norm": float(row["v_norm"]),
                    "available_points_before": active_count,
                    "available_points_after": active_count - 1,
                }
            )
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (item["statistic"], item["v_norm"]),
            reverse=True,
        )
        return candidates[0]

    @classmethod
    def solve(
        cls,
        S_list,
        T_input,
        max_iter=50,
        tol=1e-8,
        verbose=False,
        weight_constraint=1e5,
        use_quaternion_constraint=True,
        iterative_outlier_removal=False,
        outlier_threshold=3.0,
        outlier_min_points=4,
        point_ids=None,
        station_names=None,
        initial_point_coordinates=None,
        datum_station_index=0,
        linear_solver='legacy_normal',
        initial_station_parameters=None,
    ):
        if point_ids is None:
            return super().solve(
                S_list,
                T_input,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                weight_constraint=weight_constraint,
                use_quaternion_constraint=use_quaternion_constraint,
                iterative_outlier_removal=iterative_outlier_removal,
                outlier_threshold=outlier_threshold,
                outlier_min_points=outlier_min_points,
            )

        started = time.time()
        targets = T_input if isinstance(T_input, list) else [T_input] * len(S_list)
        source_matrices = [np.asarray(source, dtype=float).copy() for source in S_list]
        target_matrices = [np.asarray(target, dtype=float).copy() for target in targets]
        station_point_ids = [[str(value) for value in ids] for ids in point_ids]
        if not (
            len(source_matrices)
            == len(target_matrices)
            == len(station_point_ids)
        ):
            raise ValueError("Station source, target and point-id lists must have equal length")
        for source, target, ids in zip(
            source_matrices, target_matrices, station_point_ids
        ):
            if len(source) != len(target) or len(source) != len(ids):
                raise ValueError("Each station must have matching source, target and point-id rows")

        station_names = list(station_names or [f"station_{i + 1}" for i in range(len(S_list))])
        station_parameters = cls._initial_station_parameters(
            source_matrices, target_matrices
        )
        if initial_station_parameters is not None:
            station_parameters = np.asarray(initial_station_parameters, dtype=float).copy()
            if station_parameters.shape != (len(source_matrices), 8) or not np.isfinite(station_parameters).all():
                raise ValueError('Invalid warm-start station parameters')
            qnorm = np.linalg.norm(station_parameters[:, 1:5], axis=1)
            if np.any(qnorm <= 0) or np.any(station_parameters[:, 0] <= 0):
                raise ValueError('Warm-start rotations/scales must be valid')
            station_parameters[:, 1:5] /= qnorm[:, None]
        point_order = cls._point_order(station_point_ids)
        point_offsets = {point_id: index for index, point_id in enumerate(point_order)}
        point_coordinates = cls._initial_point_coordinates(
            source_matrices,
            target_matrices,
            station_point_ids,
            station_parameters,
            point_order,
            initial_point_coordinates,
        )
        if initial_station_parameters is not None:
            point_coordinates = np.asarray([initial_point_coordinates[p] for p in point_order], dtype=float).copy()
        datum_station_index = int(datum_station_index)
        if not 0 <= datum_station_index < len(source_matrices):
            raise ValueError("datum_station_index is outside the station list")
        free_station_indices = [
            index for index in range(len(source_matrices))
            if index != datum_station_index
        ]
        free_station_offsets = {
            station_index: offset * 8
            for offset, station_index in enumerate(free_station_indices)
        }
        active_masks = [
            np.ones(len(source), dtype=bool) for source in source_matrices
        ]
        outlier_trace = []
        iteration_history = []
        converged = False
        stop_reason = 'MAXIMUM_ITERATIONS'
        lm_lambda = 1e-3
        lm_nu = 2.0
        lm_trials = []
        if int(max_iter) < 1 or tol <= 0:
            raise ValueError('max_iter must be positive and tol must be > 0')

        for iteration in range(int(max_iter)):
            A, L, A_phys_step, L_phys_step, _, _ = cls._build_coupled_system(
                source_matrices,
                station_point_ids,
                active_masks,
                station_parameters,
                point_coordinates,
                point_offsets,
                free_station_offsets,
                use_quaternion_constraint,
                weight_constraint,
            )
            normal = A.T @ A
            right_hand_side = A.T @ L
            damping = (
                1e-6
                * np.trace(normal)
                / max(1, len(normal))
                * np.eye(len(normal))
            )
            lm_info = {}
            if linear_solver == 'adaptive_lm':
                # Constrained LM in column-equilibrated tangent coordinates.
                # Quaternion constraints are exact linearized conditions, not
                # pseudo-observations contributing to the damping magnitude.
                from scipy.linalg import null_space
                if not use_quaternion_constraint:
                    raise ValueError('adaptive_lm requires quaternion constraints')
                C = A[A_phys_step.shape[0]:]
                c = L[A_phys_step.shape[0]:]
                Z = null_space(C)
                particular = np.linalg.lstsq(C, c, rcond=None)[0]
                reduced = A_phys_step @ Z
                scales = np.linalg.norm(reduced, axis=0)
                if np.any(scales == 0):
                    raise ValueError('Unobservable reduced network parameter')
                B = reduced / scales
                U, singular, Vt = np.linalg.svd(B, full_matrices=False)
                if np.min(singular) <= max(B.shape)*np.finfo(float).eps*singular[0]:
                    raise ValueError('Rank deficient reduced network')
                rhs = L_phys_step - A_phys_step @ particular
                projected_rhs = U.T @ rhs
                objective = float(L_phys_step @ L_phys_step)
                gradient_relative = float(np.linalg.norm(B.T @ L_phys_step, ord=np.inf) /
                                          max(np.linalg.norm(L_phys_step), 1e-15))
                accepted = False
                point_base = len(free_station_offsets) * 8
                for trial in range(40):
                    used_lambda = lm_lambda
                    # SVD solves (B'B + lambda I) z = B' rhs without
                    # explicitly forming the ill-conditioned normal matrix.
                    step = Vt.T @ ((singular/(singular**2 + used_lambda))*projected_rhs)
                    delta = particular + Z @ (step/scales)
                    candidate_stations = station_parameters.copy()
                    for station_index, offset in free_station_offsets.items():
                        candidate_stations[station_index] += delta[offset:offset+8]
                        candidate_stations[station_index,1:5] /= np.linalg.norm(candidate_stations[station_index,1:5])
                    candidate_points = point_coordinates + delta[point_base:].reshape(-1,3)
                    _, _, _, candidate_v, _, _ = cls._build_coupled_system(
                        source_matrices, station_point_ids, active_masks, candidate_stations,
                        candidate_points, point_offsets, free_station_offsets, False, weight_constraint)
                    objective_after = float(candidate_v @ candidate_v)
                    linear_change = A_phys_step @ delta
                    predicted = float(2*L_phys_step@linear_change-linear_change@linear_change)
                    actual = objective-objective_after
                    rho = actual/predicted if predicted > 0 else -np.inf
                    # At the numerical floor do not confuse a damped tiny step
                    # with convergence: require independent stationarity too.
                    roundoff_stationary = (np.linalg.norm(delta)<tol and gradient_relative<1e-7
                        and abs(actual)<=1e-13*max(1.,objective))
                    accepted = bool((rho > 1e-4 and actual > 0) or roundoff_stationary)
                    lm_info = {'lambda':used_lambda,'gain_ratio':float(rho),
                               'predicted_reduction':predicted,'actual_reduction':actual,
                               'objective_after':objective_after,'gradient_relative':gradient_relative,
                               'trials':trial+1,'roundoff_stationary':bool(roundoff_stationary)}
                    lm_trials.append({'iteration':iteration+1,'trial':trial+1,
                                      'accepted':accepted,**lm_info})
                    if accepted:
                        lm_lambda = max(1e-15, used_lambda*max(1/3,1-(2*rho-1)**3)) if np.isfinite(rho) and rho>0 else used_lambda
                        lm_nu = 2.0
                        break
                    lm_lambda = min(1e15, used_lambda*lm_nu)
                    lm_nu = min(1e12, 2*lm_nu)
                if not accepted:
                    stop_reason = 'LM_NO_ACCEPTABLE_STEP'
                    break
                damping = used_lambda*np.eye(len(normal))
            elif linear_solver == 'constrained_scaled_svd':
                # Solve quaternion constraints exactly at each linearization.
                # Column equilibration avoids damping metres and quaternion
                # components by a trace dominated by the norm pseudo-equations.
                from scipy.linalg import null_space
                C = A[A_phys_step.shape[0]:]
                c = L[A_phys_step.shape[0]:]
                Z = null_space(C) if C.size else np.eye(A.shape[1])
                particular = np.linalg.lstsq(C, c, rcond=None)[0] if C.size else np.zeros(A.shape[1])
                reduced = A_phys_step @ Z
                column_norm = np.linalg.norm(reduced, axis=0)
                if np.any(column_norm == 0):
                    raise ValueError('Unobservable reduced network parameter')
                B = reduced / column_norm
                rhs = L_phys_step - A_phys_step @ particular
                step, _, rank, _ = np.linalg.lstsq(B, rhs, rcond=None)
                if rank != B.shape[1]:
                    raise ValueError('Rank deficient reduced network')
                delta = particular + Z @ (step / column_norm)
                damping = np.zeros_like(normal)
            elif linear_solver == 'legacy_normal':
                try:
                    delta = np.linalg.solve(normal + damping, right_hand_side)
                except np.linalg.LinAlgError:
                    delta = np.linalg.pinv(normal + damping) @ right_hand_side
            else:
                raise ValueError('Unknown linear_solver')

            new_station_parameters = station_parameters.copy()
            for station_index, offset in free_station_offsets.items():
                new_station_parameters[station_index] += delta[offset:offset + 8]
            point_base = len(free_station_offsets) * 8
            new_point_coordinates = point_coordinates + delta[point_base:].reshape(-1, 3)
            if linear_solver == 'adaptive_lm':
                new_station_parameters, new_point_coordinates = candidate_stations, candidate_points
            if linear_solver == 'constrained_scaled_svd':
                # Backtrack on the physical objective; norm is enforced exactly.
                current_objective = float(L_phys_step @ L_phys_step)
                accepted = False
                for backtrack in range(25):
                    fraction = 0.5**backtrack
                    candidate_stations = station_parameters.copy()
                    for station_index, offset in free_station_offsets.items():
                        candidate_stations[station_index] += fraction*delta[offset:offset+8]
                        candidate_stations[station_index,1:5] /= np.linalg.norm(candidate_stations[station_index,1:5])
                    candidate_points = point_coordinates + fraction*delta[point_base:].reshape(-1,3)
                    _, _, _, candidate_v, _, _ = cls._build_coupled_system(
                        source_matrices, station_point_ids, active_masks, candidate_stations,
                        candidate_points, point_offsets, free_station_offsets, False, weight_constraint)
                    if float(candidate_v @ candidate_v) <= current_objective + 1e-14:
                        new_station_parameters, new_point_coordinates = candidate_stations, candidate_points
                        delta *= fraction
                        accepted = True
                        break
                if not accepted:
                    stop_reason = 'LINE_SEARCH_FAILED'
                    break
            delta_norm = float(np.linalg.norm(delta))
            iteration_history.append({
                'iteration': iteration + 1, 'increment_norm': delta_norm,
                'objective_before': float(L @ L),
                'damping_diagonal': float(damping[0, 0]),
                'gradient_norm_before': float(np.linalg.norm(right_hand_side)),
                **lm_info,
            })
            if not np.isfinite(delta_norm):
                stop_reason = 'NONFINITE_INCREMENT'
                break

            if iterative_outlier_removal:
                _, _, A_physical, L_physical, coordinate_meta, point_meta = (
                    cls._build_coupled_system(
                        source_matrices,
                        station_point_ids,
                        active_masks,
                        new_station_parameters,
                        new_point_coordinates,
                        point_offsets,
                        free_station_offsets,
                        False,
                        weight_constraint,
                    )
                )
                candidate = cls._select_coupled_outlier(
                    A_physical,
                    L_physical,
                    coordinate_meta,
                    point_meta,
                    station_point_ids,
                    active_masks,
                    outlier_threshold,
                    outlier_min_points,
                )
                if candidate is not None:
                    station_index = candidate["station_index"]
                    local_index = candidate["point_index"]
                    active_masks[station_index][local_index] = False
                    candidate["iteration"] = iteration + 1
                    candidate["station"] = station_names[station_index]
                    outlier_trace.append(candidate)
                    station_parameters = new_station_parameters
                    point_coordinates = new_point_coordinates
                    if verbose:
                        print(
                            "[OUTLIER] Removed network link "
                            f"{station_names[station_index]}/{candidate['point_id']} "
                            f"inside coupled LM iteration {iteration + 1}"
                        )
                    continue

            station_parameters = new_station_parameters
            point_coordinates = new_point_coordinates
            if delta_norm < tol and (linear_solver != 'adaptive_lm' or lm_info['gradient_relative'] < 1e-7):
                converged = True
                stop_reason = 'INCREMENT_TOLERANCE'
                break

        A, L, A_physical, L_physical, coordinate_meta, point_meta = (
            cls._build_coupled_system(
                source_matrices,
                station_point_ids,
                active_masks,
                station_parameters,
                point_coordinates,
                point_offsets,
                free_station_offsets,
                use_quaternion_constraint,
                weight_constraint,
            )
        )
        effective_parameters = len(free_station_offsets) * 7 + len(point_offsets) * 3
        dof = max(1, A_physical.shape[0] - effective_parameters)
        sigma0_sq = float(np.sum(L_physical**2) / dof)
        coupled_covariance = sigma0_sq * cls._covariance_from_design(A)
        if linear_solver in ('constrained_scaled_svd', 'adaptive_lm'):
            from scipy.linalg import null_space
            C = A[A_physical.shape[0]:]
            Z = null_space(C) if C.size else np.eye(A.shape[1])
            coupled_covariance = sigma0_sq * Z @ cls._covariance_from_design(A_physical @ Z) @ Z.T

        comparison_indices = [
            index
            for index, meta in enumerate(coordinate_meta)
            if int(meta["station_index"]) != datum_station_index
        ]
        observation_labels = [
            (
                f"{station_names[int(coordinate_meta[index]['station_index'])]}."
                f"{coordinate_meta[index]['point_id']}."
                f"{('X', 'Y', 'Z')[int(coordinate_meta[index]['component'])]}"
            )
            for index in comparison_indices
        ]
        observation_design = A_physical[comparison_indices, :]
        network_observation_covariance = (
            observation_design @ coupled_covariance @ observation_design.T
        )
        stationwise_observation_covariance = np.zeros_like(
            network_observation_covariance
        )
        comparison_positions = {
            physical_index: output_index
            for output_index, physical_index in enumerate(comparison_indices)
        }
        for station_index, station_offset in free_station_offsets.items():
            station_physical_indices = [
                index
                for index in comparison_indices
                if int(coordinate_meta[index]["station_index"]) == station_index
            ]
            if not station_physical_indices:
                continue
            station_rows = [
                comparison_positions[index] for index in station_physical_indices
            ]
            station_design = A_physical[
                station_physical_indices,
                station_offset:station_offset + 8,
            ]
            station_covariance = sigma0_sq * cls._covariance_from_design(
                station_design
            )
            fitted_covariance = (
                station_design @ station_covariance @ station_design.T
            )
            stationwise_observation_covariance[np.ix_(
                station_rows,
                station_rows,
            )] = fitted_covariance

        station_covariance = np.zeros(
            (len(source_matrices) * 8, len(source_matrices) * 8),
            dtype=float,
        )
        for station_i, offset_i in free_station_offsets.items():
            output_i = station_i * 8
            for station_j, offset_j in free_station_offsets.items():
                output_j = station_j * 8
                station_covariance[
                    output_i:output_i + 8,
                    output_j:output_j + 8,
                ] = coupled_covariance[
                    offset_i:offset_i + 8,
                    offset_j:offset_j + 8,
                ]

        results = []
        for params in station_parameters:
            results.append(
                (
                    float(params[0]),
                    cls._quat_to_matrix(params[1:5]),
                    params[5:8].copy(),
                )
            )

        station_parameter_names = [
            f"{station_name}.{parameter}"
            for station_name in station_names
            for parameter in ("scale", "q0", "q1", "q2", "q3", "tx", "ty", "tz")
        ]
        coupled_parameter_names = [
            f"{station_names[index]}.{parameter}"
            for index in free_station_indices
            for parameter in ("scale", "q0", "q1", "q2", "q3", "tx", "ty", "tz")
        ]
        coupled_parameter_names.extend(
            f"point_{point_id}.{component}"
            for point_id in point_order
            for component in ("x", "y", "z")
        )
        # Exact tangent restriction for statistical diagnostics. Quaternion
        # normalization constraints are not counted as physical observations.
        from scipy.linalg import null_space
        constraints = A[A_physical.shape[0]:, :]
        tangent_basis = null_space(constraints) if constraints.size else np.eye(A.shape[1])
        statistical_design = A_physical @ tangent_basis
        diagnostics = {
            'iterations': len(iteration_history),
            'iteration_history': iteration_history,
            'lm_trial_history': lm_trials,
            'solution_type': 'iterative',
            'converged': converged,
            'stop_reason': stop_reason,
            'max_iterations': int(max_iter),
            'tolerance': float(tol),
            'final_increment_norm': iteration_history[-1]['increment_norm'] if iteration_history else np.nan,
            'final_objective': float(L @ L),
            'damping_coefficient': (lm_lambda if linear_solver == 'adaptive_lm' else
                                    0.0 if linear_solver == 'constrained_scaled_svd' else 1e-6),
            'lm_initial_lambda': 1e-3 if linear_solver == 'adaptive_lm' else None,
            'lm_acceptance_ratio': 1e-4 if linear_solver == 'adaptive_lm' else None,
            'lm_gradient_tolerance': 1e-7 if linear_solver == 'adaptive_lm' else None,
            'linear_solver': linear_solver,
            'solver_description': ('Gauss-Newton with exact linearized quaternion constraints, column equilibration, SVD and backtracking'
                                   if linear_solver == 'constrained_scaled_svd' else
                                   'Levenberg-Marquardt with adaptive gain-ratio damping, exact linearized quaternion constraints and scaled SVD solve'
                                   if linear_solver == 'adaptive_lm' else 'legacy trace-damped normal equations'),
            'statistical_design_matrix': statistical_design,
            'statistical_parameter_basis': tangent_basis,
            'quaternion_constraint_matrix': constraints,
            'degrees_of_freedom': int(A_physical.shape[0] - np.linalg.matrix_rank(statistical_design)),
            'dof': int(A_physical.shape[0] - np.linalg.matrix_rank(statistical_design)),
            "design_matrix": A_physical,
            "observation_vector": L_physical,
            "observation_residuals": L_physical,
            "normal_matrix": A.T @ A,
            "correlation_matrix": cls._correlation_from_covariance(station_covariance),
            "covariance_param_names": station_parameter_names,
            "active_observation_masks": active_masks,
            "active_observation_indices": [
                np.flatnonzero(mask).astype(int).tolist() for mask in active_masks
            ],
            "iterative_outlier_trace": outlier_trace,
            "iterative_outlier_count": len(outlier_trace),
            "coupled_design_matrix": A_physical,
            "coupled_covariance": coupled_covariance,
            "coupled_correlation_matrix": cls._correlation_from_covariance(
                coupled_covariance
            ),
            "coupled_parameter_names": coupled_parameter_names,
            "coupled_covariance_param_names": coupled_parameter_names,
            "coupled_correlation_matrix_param_names": coupled_parameter_names,
            "network_observation_covariance_matrix": network_observation_covariance,
            "network_observation_correlation_matrix": cls._correlation_from_covariance(
                network_observation_covariance
            ),
            "network_observation_correlation_matrix_param_names": observation_labels,
            "stationwise_observation_covariance_matrix": (
                stationwise_observation_covariance
            ),
            "stationwise_observation_correlation_matrix": (
                cls._correlation_from_covariance(
                    stationwise_observation_covariance
                )
            ),
            "stationwise_observation_correlation_matrix_param_names": (
                observation_labels
            ),
            "observation_comparison_design_matrix": observation_design,
            "shared_point_ids": point_order,
            "shared_point_coordinates": point_coordinates,
            "datum_station_index": datum_station_index,
            "datum_station_name": station_names[datum_station_index],
            "fully_coupled_network": True,
            "coordinate_metadata": coordinate_meta,
            "point_metadata": point_meta,
        }
        return (
            results,
            sigma0_sq,
            station_covariance,
            L_physical,
            time.time() - started,
            diagnostics,
        )

    @staticmethod
    def _covariance_from_design(design_matrix):
        design = np.asarray(design_matrix, dtype=float)
        if design.size == 0:
            return np.zeros((design.shape[1], design.shape[1]), dtype=float)
        _, singular_values, right_vectors = np.linalg.svd(
            design,
            full_matrices=False,
        )
        tolerance = (
            max(design.shape)
            * np.finfo(float).eps
            * singular_values[0]
            if singular_values.size
            else 0.0
        )
        inverse_squared = np.zeros_like(singular_values)
        valid = singular_values > tolerance
        inverse_squared[valid] = 1.0 / singular_values[valid] ** 2
        covariance = (right_vectors.T * inverse_squared) @ right_vectors
        return 0.5 * (covariance + covariance.T)

    @staticmethod
    def _correlation_from_covariance(covariance):
        covariance = np.asarray(covariance, dtype=float)
        sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
        denominator = np.outer(sigma, sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.divide(
                covariance,
                denominator,
                out=np.zeros_like(covariance),
                where=denominator > 0,
            )
        correlation = np.nan_to_num(
            correlation,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        correlation = np.clip(correlation, -1.0, 1.0)
        positive_variance = sigma > 0
        correlation[np.diag_indices_from(correlation)] = positive_variance.astype(float)
        return correlation
