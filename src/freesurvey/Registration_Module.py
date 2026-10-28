"""
================================================================================
                   3D GEODETIC NETWORK ADJUSTMENT - COMPLETE BENCHMARK
================================================================================
"""
import numpy as np
import time
import logging
import math
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None
try:
    from scipy.optimize import least_squares
except ModuleNotFoundError:
    least_squares = None
try:
    from scipy.spatial.transform import Rotation
except ModuleNotFoundError:
    class Rotation:
        """Minimal fallback for Rotation.from_matrix(...).as_quat()."""

        def __init__(self, quat_xyzw):
            self._quat_xyzw = np.asarray(quat_xyzw, dtype=float)

        @classmethod
        def from_matrix(cls, matrix):
            matrix = np.asarray(matrix, dtype=float)
            trace = np.trace(matrix)
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2.0
                w = 0.25 * s
                x = (matrix[2, 1] - matrix[1, 2]) / s
                y = (matrix[0, 2] - matrix[2, 0]) / s
                z = (matrix[1, 0] - matrix[0, 1]) / s
            else:
                diag = np.diag(matrix)
                idx = int(np.argmax(diag))
                if idx == 0:
                    s = np.sqrt(1.0 + matrix[0, 0] -
                                matrix[1, 1] - matrix[2, 2]) * 2.0
                    w = (matrix[2, 1] - matrix[1, 2]) / s
                    x = 0.25 * s
                    y = (matrix[0, 1] + matrix[1, 0]) / s
                    z = (matrix[0, 2] + matrix[2, 0]) / s
                elif idx == 1:
                    s = np.sqrt(1.0 + matrix[1, 1] -
                                matrix[0, 0] - matrix[2, 2]) * 2.0
                    w = (matrix[0, 2] - matrix[2, 0]) / s
                    x = (matrix[0, 1] + matrix[1, 0]) / s
                    y = 0.25 * s
                    z = (matrix[1, 2] + matrix[2, 1]) / s
                else:
                    s = np.sqrt(1.0 + matrix[2, 2] -
                                matrix[0, 0] - matrix[1, 1]) * 2.0
                    w = (matrix[1, 0] - matrix[0, 1]) / s
                    x = (matrix[0, 2] + matrix[2, 0]) / s
                    y = (matrix[1, 2] + matrix[2, 1]) / s
                    z = 0.25 * s
            quat = np.array([x, y, z, w], dtype=float)
            norm = np.linalg.norm(quat)
            return cls(quat / norm if norm else quat)

        def as_quat(self):
            return self._quat_xyzw
try:
    from scipy.linalg import eigh, svd, solve
except ModuleNotFoundError:
    eigh = np.linalg.eigh
    svd = np.linalg.svd
    solve = np.linalg.solve

logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 2: METHOD 1 - SEQUENTIAL SVD (PROCRUSTES ANALYSIS)
# ============================================================================

class Method1_SequentialSVD:
    method_code = "SVD-P"
    article_name = "SVD-based Orthogonal Procrustes transformation"
    reference = "Arun, Huang and Blostein (1987); Umeyama (1991)."
    """
    Sequential Singular Value Decomposition (Procrustes Analysis) Method.

    This is the most fundamental approach to the orthogonal Procrustes problem:
    Find orthogonal matrix R that minimizes ||T - k*S*R^T||².

    Mathematical Formulation:
    -------------------------
    Given centered coordinates: Sc = S - μ_S,  Tc = T - μ_T

    The rotation matrix R is found by SVD of the cross-covariance matrix:
        U * Σ * V^T = Sc^T * Tc
        R = V * U^T

    The scale factor k is derived from:
        k = trace(Tc^T * Sc * R^T) / trace(Sc^T * Sc)

    Translation vector: t = μ_T - k * μ_S * R^T

    Advantages:
    -----------
    - Simple, direct solution (non-iterative)
    - Guaranteed orthogonal rotation matrix
    - Computationally efficient

    Limitations:
    ------------
    - Assumes errors only in source coordinates
    - No statistical rigor (no variance estimation)
    - Sensitive to outliers

    Reference: Schönemann, P.H. (1966). A generalized solution of the 
               orthogonal Procrustes problem.
    """

    @staticmethod
    def solve(S, T):
        """
        Solve 3D similarity transformation using Sequential SVD.

        Parameters:
        -----------
        S : ndarray (n, 3)
            Source coordinates (Local Frame - LF)
        T : ndarray (n, 3)
            Target coordinates (Vessel Frame - VF)

        Returns:
        --------
        scale : float
            Estimated scale factor (k)
        R : ndarray (3, 3)
            Estimated rotation matrix (orthogonal, det=+1)
        translation : ndarray (3,)
            Estimated translation vector
        elapsed_time : float
            Computation time in seconds
        """
        start_time = time.time()

        # Step 1: Center the coordinates (remove translations)
        # This isolates rotation and scale from translation
        mean_S = np.mean(S, axis=0)  # Centroid of source points
        mean_T = np.mean(T, axis=0)  # Centroid of target points
        Sc = S - mean_S  # Centered source coordinates
        Tc = T - mean_T  # Centered target coordinates

        # Step 2: Compute cross-covariance matrix
        # H = Sc^T * Tc  (3x3 matrix)
        H = Sc.T @ Tc

        # Step 3: Singular Value Decomposition of H
        # H = U * Σ * V^T
        U, sigma, Vt = svd(H)

        # Step 4: Compute rotation matrix
        # R = V * U^T
        R = Vt.T @ U.T

        # Step 5: Ensure proper rotation (det = +1, no reflection)
        # If determinant is -1, we have a reflection which is physically invalid
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1  # Flip the last row of V^T
            R = Vt.T @ U.T

        # Step 6: Compute scale factor using least squares
        # For each point: Tc_i = k * Sc_i * R^T
        # Minimizing sum ||Tc_i - k*Sc_i*R^T||²
        # Solution: k = sum(Tc_i · (Sc_i * R^T)) / sum(||Sc_i||²)
        scale = np.sum(Tc * (Sc @ R.T)) / np.sum(Sc**2)

        # Step 7: Compute translation vector
        # t = μ_T - k * μ_S * R^T
        translation = mean_T - scale * (mean_S @ R.T)

        elapsed_time = time.time() - start_time

        return scale, R, translation, elapsed_time

# ============================================================================
# SECTION 3: METHOD 2 - SCHUT (1973) SPACE-TIME QUATERNION METHOD
# ============================================================================

class Schut1973_SpaceTime:
    method_code = "SCH-Q"
    article_name = "Schut quaternion-based spatial similarity transformation"
    reference = "Schut (1973); Horn (1987)."
    """
    Schut's (1973) Space-Time Quaternion Method for 3D Transformations.

    This elegant method solves the 3D similarity transformation using quaternion
    algebra and eigenvalue decomposition of a 4x4 matrix.

    Mathematical Derivation:
    ------------------------
    The rotation quaternion q = [q0, q1, q2, q3] satisfies:
        R(q) = (q0²+q1²-q2²-q3²)   2(q1q2-q0q3)     2(q1q3+q0q2)
               2(q1q2+q0q3)    (q0²-q1²+q2²-q3²)   2(q2q3-q0q1)
               2(q1q3-q0q2)      2(q2q3+q0q1)    (q0²-q1²-q2²+q3²)

    Key Insight: The rotation parameters can be found by solving:
        N * q = λ_max * q

    Where N is a 4x4 symmetric matrix built from the cross-covariance.

    Advantages:
    -----------
    - Exact linear solution (no iterations)
    - Handles scale estimation optimally
    - Theoretically rigorous

    Reference: Schut, G.H. (1973). On the choice of parameters in the 
               three-dimensional similarity transformation.
    """

    @staticmethod
    def solve(S, T):
        """
        Solve 3D similarity transformation using Schut's eigenvalue method.

        Parameters:
        -----------
        S : ndarray (n, 3)
            Source coordinates (Local Frame)
        T : ndarray (n, 3)
            Target coordinates (Vessel Frame)

        Returns:
        --------
        scale : float
            Estimated scale factor
        R : ndarray (3, 3)
            Estimated rotation matrix
        translation : ndarray (3,)
            Estimated translation vector
        elapsed_time : float
            Computation time in seconds
        """
        start_time = time.time()

        # Step 1: Center coordinates
        mean_S = np.mean(S, axis=0)
        mean_T = np.mean(T, axis=0)
        Sc = S - mean_S
        Tc = T - mean_T

        # Step 2: Compute optimal scale factor
        # Minimizing sum ||Tc - k*Sc||² yields k = sqrt(||Tc||²/||Sc||²)
        scale = np.sqrt(np.sum(Tc**2) / np.sum(Sc**2))

        # Step 3: Scale the source coordinates
        Sc_scaled = Sc * scale

        # Step 4: Build cross-covariance matrix M = (scale*Sc)^T * Tc
        M = Sc_scaled.T @ Tc

        # Step 5: Construct the 4x4 symmetric matrix N (Schut's Equation 12)
        # N represents the quaternion algebra in matrix form
        N = np.zeros((4, 4))

        # Diagonal elements
        N[0, 0] = M[0, 0] + M[1, 1] + M[2, 2]  # trace of M
        N[1, 1] = M[0, 0] - M[1, 1] - M[2, 2]
        N[2, 2] = -M[0, 0] + M[1, 1] - M[2, 2]
        N[3, 3] = -M[0, 0] - M[1, 1] + M[2, 2]

        # Off-diagonal elements (symmetric)
        N[0, 1] = N[1, 0] = M[1, 2] - M[2, 1]
        N[0, 2] = N[2, 0] = M[2, 0] - M[0, 2]
        N[0, 3] = N[3, 0] = M[0, 1] - M[1, 0]
        N[1, 2] = N[2, 1] = M[0, 1] + M[1, 0]
        N[1, 3] = N[3, 1] = M[2, 0] + M[0, 2]
        N[2, 3] = N[3, 2] = M[1, 2] + M[2, 1]

        # Step 6: Eigenvalue decomposition (N is symmetric, use eigh)
        eigenvalues, eigenvectors = eigh(N)

        # Step 7: The eigenvector corresponding to the largest eigenvalue
        # gives the rotation quaternion
        q = eigenvectors[:, np.argmax(eigenvalues)]

        # Step 8: Normalize quaternion (ensure unit length)
        q = q / np.linalg.norm(q)
        q0, q1, q2, q3 = q

        # Step 9: Convert quaternion to rotation matrix
        R = np.array([
            [q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0**2-q1**2-q2**2+q3**2]
        ])

        # Step 10: Compute translation
        translation = mean_T - scale * (mean_S @ R.T)

        elapsed_time = time.time() - start_time

        return scale, R, translation, elapsed_time

# ============================================================================
# SECTION 4: METHOD 3 - STęPIEŃ (2020) QUASI-SIMILARITY TRANSFORMATION (Q-ST)
# ============================================================================

class Stepien2020_QST:
    method_code = "Q-ST"
    article_name = "Quasi-Similarity Transformation"
    reference = "Stepien et al. (2020)."
    """Affine-capable implementation of the staged Q-ST procedure.

    Pairwise-distance scale is filtered bilaterally at 2 sigma_D; scaled
    source coordinates and targets are centred; the free 3x3 map is fitted
    by least squares. The affine correction is evaluated at every fit.
    No SVD orthogonalization is applied. For centred full-rank LS data,
    the correction is redundant up to numerical precision. The initial
    scalar scale is not independently identifiable from the free matrix.
    Reference: Stepien et al. (2020), doi:10.3390/s20226497.
    """

    @staticmethod
    def solve(S, T):
        """
        Fit the affine-capable Q-ST variant documented for this experiment.

        Parameters:
        -----------
        S : ndarray (n, 3)
            Source coordinates (Local Frame)
        T : ndarray (n, 3)
            Target coordinates (Vessel Frame)

        Returns:
        --------
        scale : float
            Preliminary distance-based scalar; not identifiable independently of the free matrix
        R : ndarray (3, 3)
            Free quasi-rotation matrix; orthogonality is not enforced
        translation : ndarray (3,)
            Estimated translation vector
        stats : dict
            Statistical information about the adjustment
        elapsed_time : float
            Computation time in seconds
        """
        start_time = time.time()
        n = len(S)

        # ==============================================================
        # PHASE 1: SCALE ESTIMATION USING PAIRWISE DISTANCES
        # ==============================================================
        # Create all unique point pairs (n choose 2 combinations)
        d_S = []  # Distances in source system
        d_T = []  # Distances in target system

        for i in range(n):
            for j in range(i + 1, n):
                d_S.append(np.linalg.norm(S[j] - S[i]))
                d_T.append(np.linalg.norm(T[j] - T[i]))

        # Convert to column vectors for least squares
        D_LF_all = np.array(d_S, dtype=float).reshape(-1, 1)
        D_VF_all = np.array(d_T, dtype=float).reshape(-1, 1)

        # Least squares solution: k = (A^T * A)^-1 * A^T * L
        # This minimizes sum (k * d_S - d_T)²
        scale = (np.linalg.pinv(D_LF_all.T @ D_LF_all)
                 @ D_LF_all.T @ D_VF_all)[0, 0]
        distance_residuals_all = D_LF_all * scale - D_VF_all
        if len(distance_residuals_all) > 1:
            sigma_D = float(np.sqrt((distance_residuals_all.T @ distance_residuals_all)
                            [0, 0] / (len(distance_residuals_all) - 1)))
        else:
            sigma_D = 0.0
        scale_pair_mask = np.ones(len(D_LF_all), dtype=bool)
        D_LF_used = D_LF_all
        D_VF_used = D_VF_all
        if sigma_D > 0:
            scale_pair_mask = np.abs(
                distance_residuals_all.ravel()) < 2.0 * sigma_D
            if np.count_nonzero(scale_pair_mask) >= 1 and np.count_nonzero(~scale_pair_mask) > 0:
                D_LF_used = D_LF_all[scale_pair_mask]
                D_VF_used = D_VF_all[scale_pair_mask]
                scale = (np.linalg.pinv(D_LF_used.T @ D_LF_used)
                         @ D_LF_used.T @ D_VF_used)[0, 0]
        distance_residuals = D_LF_used * scale - D_VF_used

        # ==============================================================
        # PHASE 2: QUASI-ROTATION MATRIX ESTIMATION
        # ==============================================================
        # Center coordinates to remove translation effect
        S_scaled = scale * S
        # Stage 2
        C_LF_scaled = np.mean(S_scaled, axis=0)
        C_LF = np.mean(S, axis=0)  # Source centroid
        C_VF = np.mean(T, axis=0)  # Target centroid
        # Stage 3
        S_c = S_scaled - C_LF_scaled
        T_c = T - C_VF

        # Estágio 4: Solução explícita das equações normais (igual à Equação 25 do artigo)
        # Modelo: T_c = S_c * B  =>  B = (S_c^T * S_c)^-1 * S_c^T * T_c
        B = np.linalg.pinv(S_c.T @ S_c) @ S_c.T @ T_c
        QR = B.T
        initial_centroid_residuals = T_c - S_c @ QR.T

        # Equation (41): linear correction to QR plus a centroid translation
        # correction. For four non-coplanar fitting points this can zero the
        # adjustment-point deviations; for larger sets it gives the LS correction.
        A_corr = np.zeros((3 * n, 12), dtype=float)
        L_corr = initial_centroid_residuals.ravel()
        for row, point in enumerate(S_c):
            base = 3 * row
            x, y, z = point
            A_corr[base, 0] = 1.0
            A_corr[base, 3:6] = [x, y, z]
            A_corr[base + 1, 1] = 1.0
            A_corr[base + 1, 6:9] = [x, y, z]
            A_corr[base + 2, 2] = 1.0
            A_corr[base + 2, 9:12] = [x, y, z]
        correction, _, _, _ = np.linalg.lstsq(A_corr, L_corr, rcond=None)
        centroid_correction = correction[:3]
        QR = QR + correction[3:].reshape(3, 3)

        # ==============================================================
        # PHASE 4/5: KEEP THE QUASI-ROTATION MATRIX
        # ==============================================================
        # Q-ST does not force QR to the nearest orthogonal matrix.
        # QR = U * Σ * V^T  =>  R = U * V^T (orthogonal)
        R = QR

        # Q-ST keeps QR as solved; non-orthogonality is reported below.

        translation = C_VF - C_LF_scaled @ R.T + centroid_correction

        # ==============================================================
        # STATISTICAL EVALUATION (A POSTERIORI)
        # ==============================================================
        # Compute residuals: V = transformed - observed
        transformed = scale * (S @ R.T) + translation
        residuals = transformed - T

        # The free affine map has nine linear elements AND three translations.
        # Centroid estimation consumes three degrees of freedom as well.
        design_rank = int(np.linalg.matrix_rank(A_corr))
        dof = 3 * n - design_rank
        sigma0_sq = float(np.sum(residuals**2) / dof) if dof > 0 else np.nan
        orthogonality_matrix = np.eye(3) - R.T @ R

        stats = {
            'iterations': 0,
            'solution_type': 'direct',
            'converged': None,
            'stop_reason': 'DIRECT_SOLUTION',
            'rank_a': design_rank,
            'dof': dof,
            'degrees_of_freedom': dof,
            'variance_estimable': bool(dof > 0 and design_rank == 12),
            'geometry_full_rank': bool(design_rank == 12),
            'sigma0_sq': sigma0_sq,  # A posteriori variance factor
            # RMS error
            'rmse': np.sqrt(np.mean(np.sum(residuals**2, axis=1))),
            'max_residual': np.max(np.abs(residuals)),  # Maximum residual
            'distance_design_matrix': D_LF_used,
            'distance_observation_vector': D_VF_used,
            'distance_residuals': distance_residuals,
            'distance_residuals_all': distance_residuals_all,
            'distance_sigma_D': sigma_D,
            'distance_pair_mask': scale_pair_mask,
            'scale_pairs_removed': int(np.count_nonzero(~scale_pair_mask)),
            'source_scaled': S_scaled,
            'source_centroid_scaled': C_LF_scaled,
            'target_centroid': C_VF,
            'source_centered_scaled': S_c,
            'target_centered': T_c,
            'initial_centroid_residuals': initial_centroid_residuals,
            'qst_correction_design_matrix': A_corr,
            'qst_correction_vector': correction,
            'qst_centroid_correction': centroid_correction,
            'quasi_rotation_matrix': R,
            'orthogonality_matrix': orthogonality_matrix,
            'orthogonality_error': float(np.linalg.norm(orthogonality_matrix)),
            'zero_deviation_max_abs': float(np.max(np.abs(residuals))),
        }

        elapsed_time = time.time() - start_time

        return scale, R, translation, stats, elapsed_time

# ============================================================================
# SECTION 4B: TOTAL FREE STATION (TFS)
# ============================================================================

class TotalFreeStation_TFS:
    method_code = "TFS"
    article_name = "Total Free Station fixed-scale free-station transformation"
    reference = "Stepien (2019)."
    """
    Total Free Station (TFS) station-wise transformation.

    The TFS formulation used here follows the offshore free-station premise that
    the local station frame is shifted to the centre of gravity of the available
    control points, rotated into the vessel frame, and kept at physical scale
    lambda = 1.0. In the article workflow more than three common control points
    can be present, so the rotation is obtained as the proper rigid least-squares
    solution after centring both systems.
    """

    @staticmethod
    def solve(S, T):
        """
        Solve a fixed-scale rigid transformation using the TFS free-station
        centre-of-gravity model.

        Parameters
        ----------
        S : ndarray (n, 3)
            Source coordinates in the local station frame.
        T : ndarray (n, 3)
            Target coordinates in the vessel/reference frame.

        Returns
        -------
        scale : float
            Fixed scale factor, equal to 1.0.
        R : ndarray (3, 3)
            Proper rotation matrix.
        translation : ndarray (3,)
            Translation vector from local to target frame.
        stats : dict
            Diagnostic outputs for the shared article/GUI reporting pipeline.
        elapsed_time : float
            Computation time in seconds.
        """
        start_time = time.time()
        S = np.asarray(S, dtype=float)
        T = np.asarray(T, dtype=float)
        if S.shape != T.shape or S.ndim != 2 or S.shape[1] != 3:
            raise ValueError("TFS requires source and target arrays with shape (n, 3).")
        if len(S) < 3:
            raise ValueError("TFS requires at least three homologous control points.")

        source_centroid = np.mean(S, axis=0)
        target_centroid = np.mean(T, axis=0)
        source_centered = S - source_centroid
        target_centered = T - target_centroid

        h_matrix = source_centered.T @ target_centered
        U, singular_values, Vt = svd(h_matrix)
        R = Vt.T @ U.T
        reflection_corrected = False
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
            reflection_corrected = True

        scale = 1.0
        translation = target_centroid - source_centroid @ R.T
        transformed = scale * (S @ R.T) + translation
        residuals = transformed - T
        residual_norms = np.linalg.norm(residuals, axis=1)
        dof = max(1, 3 * len(S) - 6)
        residual_sum_squares = float(np.sum(residual_norms ** 2))
        sigma0_sq = residual_sum_squares / dof
        orthogonality_matrix = R.T @ R - np.eye(3)

        stats = {
            "method_code": TotalFreeStation_TFS.method_code,
            "article_name": TotalFreeStation_TFS.article_name,
            "tfs_strategy": "center_of_gravity_fixed_scale_rigid_ls",
            "tfs_fixed_scale": True,
            "tfs_reference_scale": 1.0,
            "fixed_scale_applied": True,
            "fixed_scale_value": 1.0,
            "source_centroid": source_centroid,
            "target_centroid": target_centroid,
            "source_centered": source_centered,
            "target_centered": target_centered,
            "cross_covariance_h": h_matrix,
            "svd_u": U,
            "svd_singular_values": singular_values,
            "svd_vt": Vt,
            "reflection_corrected": reflection_corrected,
            "rotation_determinant": float(np.linalg.det(R)),
            "orthogonality_matrix": orthogonality_matrix,
            "orthogonality_error": float(np.linalg.norm(orthogonality_matrix)),
            "scale_deviation": 0.0,
            "transformed_control_points": transformed,
            "residuals_xyz": residuals,
            "residual_norms": residual_norms,
            "observation_residuals": residuals.ravel(),
            "residual_sum_squares": residual_sum_squares,
            "degrees_of_freedom": dof,
            "dof": dof,
            "sigma0_sq": sigma0_sq,
            "sigma0": float(np.sqrt(sigma0_sq)),
            "rmse": float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))),
            "mean_residual": float(np.mean(residual_norms)),
            "std_residual": float(np.std(residual_norms)),
            "max_residual": float(np.max(residual_norms)),
        }

        elapsed_time = time.time() - start_time
        return scale, R, translation, stats, elapsed_time

# ============================================================================
# SECTION 5: METHOD 4 - GLOBAL QUATERNIONS (OBSERVATION EQUATIONS)
# ============================================================================


# Assumindo que Schut1973_SpaceTime já esteja importado no seu escopo
# from sua_biblioteca import Schut1973_SpaceTime

class Method3_GlobalQuaternionsLM_ObsEquations:
    """
    Internal quaternion observation-equation engine used by N-CUQ-LM.
    """

    is_global_lm = True
    parameterization = "quaternion"

    @staticmethod
    def _quat_to_matrix(q):
        """Converte quatérnio para matriz de rotação."""
        q = q / np.linalg.norm(q)
        q0, q1, q2, q3 = q
        return np.array([
            [q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0**2-q1**2-q2**2+q3**2]
        ])

    @staticmethod
    def _compute_residuals_and_jacobian_for_point(S_pt, T_pt, scale, q, trans):
        """Calcula vetor de resíduos e Jacobiana local (3x8) para um único ponto."""
        R = Method3_GlobalQuaternionsLM_ObsEquations._quat_to_matrix(q)
        q_norm = q / np.linalg.norm(q)
        q0, q1, q2, q3 = q_norm

        S_rot = R @ S_pt
        T_calc = scale * S_rot + trans

        # Resíduo (L)
        residual = T_pt - T_calc

        # Jacobiana local
        J = np.zeros((3, 8))
        J[:, 0] = S_rot

        dR_dq0 = 2 * np.array([[q0, -q3, q2], [q3, q0, -q1], [-q2, q1, q0]])
        dR_dq1 = 2 * np.array([[q1, q2, q3], [q2, -q1, -q0], [q3, q0, -q1]])
        dR_dq2 = 2 * np.array([[-q2, q1, q0], [q1, q2, q3], [-q0, q3, -q2]])
        dR_dq3 = 2 * np.array([[-q3, -q0, q1], [q0, -q3, q2], [q1, q2, q3]])

        J[:, 1] = scale * (dR_dq0 @ S_pt)
        J[:, 2] = scale * (dR_dq1 @ S_pt)
        J[:, 3] = scale * (dR_dq2 @ S_pt)
        J[:, 4] = scale * (dR_dq3 @ S_pt)

        J[:, 5:8] = np.eye(3)

        return residual, J

    @classmethod
    def _build_lm_system(cls, S_work, T_work, params, use_quaternion_constraint,
                         weight_constraint):
        n_stations = len(S_work)
        n_total_params = n_stations * 8
        A_list = []
        L_list = []
        A_physical = []
        L_physical = []
        coordinate_meta = []
        point_meta = []

        for i in range(n_stations):
            S_i = S_work[i]
            T_i = T_work[i]
            n_pts = min(len(S_i), len(T_i))
            p = params[i*8: (i+1)*8]
            scale = p[0]
            q = p[1:5]
            trans = p[5:8]

            for j in range(n_pts):
                res, J_local = cls._compute_residuals_and_jacobian_for_point(
                    S_i[j], T_i[j], scale, q, trans)
                A_row = np.zeros((3, n_total_params))
                A_row[:, i*8: (i+1)*8] = J_local
                A_list.append(A_row)
                L_list.append(res)
                A_physical.append(A_row)
                L_physical.append(res)
                point_meta.append({
                    "station_index": i,
                    "point_index": j,
                    "v_norm": float(np.linalg.norm(res)),
                })
                for component in range(3):
                    coordinate_meta.append({
                        "station_index": i,
                        "point_index": j,
                        "component": component,
                    })

            if use_quaternion_constraint:
                q_norm_sq = np.sum(q**2)
                res_constraint = np.array([1.0 - q_norm_sq])
                J_constraint_local = np.zeros((1, 8))
                J_constraint_local[0, 1:5] = 2 * q
                A_constraint_row = np.zeros((1, n_total_params))
                A_constraint_row[0, i*8:(i+1)*8] = J_constraint_local
                A_list.append(A_constraint_row * weight_constraint)
                L_list.append(res_constraint * weight_constraint)

        A = np.vstack(A_list) if A_list else np.zeros((0, n_total_params))
        L = np.concatenate(L_list) if L_list else np.zeros(0)
        A_phys = np.vstack(A_physical) if A_physical else np.zeros((0, n_total_params))
        L_phys = np.concatenate(L_physical) if L_physical else np.zeros(0)
        return A, L, A_phys, L_phys, coordinate_meta, point_meta

    @staticmethod
    def _select_iterative_outlier(A_phys, L_phys, coordinate_meta, point_meta,
                                  S_work, threshold, min_points_per_station):
        if not point_meta:
            return None
        scores = {}
        test_name = "baarda"
        if A_phys.size and L_phys.size == A_phys.shape[0]:
            try:
                dof = A_phys.shape[0] - A_phys.shape[1]
                if dof > 0:
                    qxx = np.linalg.pinv(A_phys.T @ A_phys)
                    qvv = np.eye(A_phys.shape[0]) - A_phys @ qxx @ A_phys.T
                    qvv_diag = np.diag(qvv)
                    sigma0 = math.sqrt(float(np.sum(L_phys ** 2) / dof))
                    valid = np.isfinite(qvv_diag) & (qvv_diag > 0) & (sigma0 > 0)
                    w = np.zeros_like(L_phys, dtype=float)
                    w[valid] = np.abs(L_phys[valid]) / (sigma0 * np.sqrt(qvv_diag[valid]))
                    for idx, meta in enumerate(coordinate_meta):
                        key = (int(meta["station_index"]), int(meta["point_index"]))
                        scores[key] = max(scores.get(key, 0.0), float(w[idx]))
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
            point_index = int(row["point_index"])
            score = scores.get((station_index, point_index), 0.0)
            if not np.isfinite(score) or score <= float(threshold):
                continue
            if len(S_work[station_index]) - 1 < int(min_points_per_station):
                continue
            candidates.append({
                "station_index": station_index,
                "point_index": point_index,
                "test": test_name,
                "statistic": float(score),
                "threshold": float(threshold),
                "v_norm": float(row["v_norm"]),
                "available_points_before": int(len(S_work[station_index])),
                "available_points_after": int(len(S_work[station_index]) - 1),
            })
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item["statistic"], item["v_norm"]), reverse=True)
        return candidates[0]

    @classmethod
    def solve(cls, S_list, T_input, max_iter=50, tol=1e-8, verbose=False,
              weight_constraint=1e5, use_quaternion_constraint=True,
              iterative_outlier_removal=False, outlier_threshold=3.0,
              outlier_min_points=4):
        """
        weight_constraint: Peso aplicado na pseudo-observação de ortogonalidade.
        """
        start_time = time.time()
        n_stations = len(S_list)
        is_t_list = isinstance(T_input, list)
        S_work = []
        T_work = []
        active_indices = []
        for i in range(n_stations):
            T_current = T_input[i] if is_t_list else T_input
            n_min = min(len(S_list[i]), len(T_current))
            S_work.append(np.asarray(S_list[i][:n_min], dtype=float).copy())
            T_work.append(np.asarray(T_current[:n_min], dtype=float).copy())
            active_indices.append(list(range(n_min)))
        outlier_trace = []

        # 1. Chute inicial com Schut
        params_list = []
        for i in range(n_stations):
            S_i = S_work[i]
            T_i = T_work[i]

            scale_init, R_init, trans_init, _ = Schut1973_SpaceTime.solve(
                S_i, T_i)
            r = Rotation.from_matrix(R_init)
            quat_xyzw = r.as_quat()
            q0, q1, q2, q3 = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]

            if q0 < 0:
                q0, q1, q2, q3 = -q0, -q1, -q2, -q3

            params_list.extend([scale_init, q0, q1, q2, q3,
                               trans_init[0], trans_init[1], trans_init[2]])

        params = np.array(params_list)
        n_total_params = n_stations * 8

        # 2. Ajuste por Mínimos Quadrados (Equações Normais)
        for iteration in range(max_iter):
            A, L, _, _, _, _ = cls._build_lm_system(
                S_work,
                T_work,
                params,
                use_quaternion_constraint,
                weight_constraint,
            )

            # Equações Normais
            N = A.T @ A
            U = A.T @ L

            # Regularização Tikhonov/Marquardt leve
            damping = 1e-6 * np.trace(N) / max(1, len(N)) * np.eye(len(N))

            try:
                delta = np.linalg.solve(N + damping, U)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(N + damping) @ U

            params_new = params + delta
            delta_norm = np.linalg.norm(delta)

            if iterative_outlier_removal:
                _, _, A_phys, L_phys, coordinate_meta, point_meta = cls._build_lm_system(
                    S_work,
                    T_work,
                    params_new,
                    False,
                    weight_constraint,
                )
                candidate = cls._select_iterative_outlier(
                    A_phys,
                    L_phys,
                    coordinate_meta,
                    point_meta,
                    S_work,
                    outlier_threshold,
                    outlier_min_points,
                )
                if candidate is not None:
                    station_index = candidate["station_index"]
                    point_index = candidate["point_index"]
                    original_index = active_indices[station_index][point_index]
                    candidate["iteration"] = iteration + 1
                    candidate["original_point_index"] = int(original_index)
                    outlier_trace.append(candidate)
                    S_work[station_index] = np.delete(S_work[station_index], point_index, axis=0)
                    T_work[station_index] = np.delete(T_work[station_index], point_index, axis=0)
                    del active_indices[station_index][point_index]
                    params = params_new
                    if verbose:
                        print(
                            "[OUTLIER] Removed station "
                            f"{station_index}, point {original_index} "
                            f"inside LM iteration {iteration + 1}"
                        )
                    continue

            if delta_norm < tol:
                params = params_new
                break

            params = params_new

        A, L, A_phys, L_phys, coordinate_meta, point_meta = cls._build_lm_system(
            S_work,
            T_work,
            params,
            use_quaternion_constraint,
            weight_constraint,
        )

        # 3. Extração dos Resultados
        results = []
        for i in range(n_stations):
            p = params[i*8:(i+1)*8]
            scale = p[0]
            q = p[1:5]
            trans = p[5:8]

            # Uma vez que forçamos q a ter norma 1 no ajustamento, R sairá estritamente ortogonal.
            R = cls._quat_to_matrix(q)
            results.append((scale, R, trans))

        # Cálculo da variância de referência (sigma0_sq)
        # Cuidado: a pseudo-observação infla L. Para fins de estatística, o ideal seria re-calcular
        # L apenas com as observações físicas, mas mantive o fluxo da matriz global.
        dof = max(1, len(L) - len(params))
        sigma0_sq = np.sum(L**2) / dof

        try:
            cov_matrix = sigma0_sq * np.linalg.pinv(A.T @ A)
        except np.linalg.LinAlgError:
            cov_matrix = np.eye(len(params)) * 1e-6

        elapsed_time = time.time() - start_time

        diagnostics = {
            "design_matrix": A_phys,
            "observation_vector": L_phys,
            "active_observation_indices": active_indices,
            "active_observation_masks": [
                np.asarray([
                    idx in set(active)
                    for idx in range(min(len(S_list[i]), len(T_input[i] if is_t_list else T_input)))
                ], dtype=bool)
                for i, active in enumerate(active_indices)
            ],
            "iterative_outlier_trace": outlier_trace,
            "iterative_outlier_count": len(outlier_trace),
        }
        return results, sigma0_sq, cov_matrix, L, elapsed_time, diagnostics


class Method4_GlobalEulerLM_ObsEquations:
    method_code = "EA-LM"
    article_name = "Euler-angle Levenberg-Marquardt global similarity adjustment"
    reference = "Levenberg (1944); Marquardt (1963); Helmert 7-parameter transformation literature."
    """Global LM using Euler angles instead of quaternions."""
    is_global_lm = True
    parameterization = "euler_zyx"

    @staticmethod
    def _euler_to_matrix(angles):
        omega, phi, kappa = angles
        co, so = np.cos(omega), np.sin(omega)
        cp, sp = np.cos(phi), np.sin(phi)
        ck, sk = np.cos(kappa), np.sin(kappa)
        rx = np.array([[1, 0, 0], [0, co, -so], [0, so, co]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[ck, -sk, 0], [sk, ck, 0], [0, 0, 1]])
        return rz @ ry @ rx

    @staticmethod
    def _matrix_to_euler(matrix):
        matrix = np.asarray(matrix, dtype=float)
        phi = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
        cp = np.cos(phi)
        if abs(cp) > 1e-12:
            omega = np.arctan2(matrix[2, 1], matrix[2, 2])
            kappa = np.arctan2(matrix[1, 0], matrix[0, 0])
        else:
            omega = 0.0
            kappa = np.arctan2(-matrix[0, 1], matrix[1, 1])
        return np.array([omega, phi, kappa], dtype=float)

    @classmethod
    def _transform_point(cls, S_pt, params):
        scale = params[0]
        angles = params[1:4]
        trans = params[4:7]
        return scale * (cls._euler_to_matrix(angles) @ S_pt) + trans

    @classmethod
    def _compute_residuals_and_jacobian_for_point(cls, S_pt, T_pt, params):
        calc = cls._transform_point(S_pt, params)
        residual = T_pt - calc
        J = np.zeros((3, 7))
        for idx in range(7):
            eps = 1e-6
            if idx == 0:
                eps = max(1e-7, abs(params[idx]) * 1e-7)
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[idx] += eps
            p_minus[idx] -= eps
            J[:, idx] = (
                cls._transform_point(S_pt, p_plus) -
                cls._transform_point(S_pt, p_minus)
            ) / (2.0 * eps)
        return residual, J

    @classmethod
    def solve(cls, S_list, T_input, max_iter=50, tol=1e-8, verbose=False):
        start_time = time.time()
        n_stations = len(S_list)
        is_t_list = isinstance(T_input, list)

        params_list = []
        for i in range(n_stations):
            T_current = T_input[i] if is_t_list else T_input
            n_min = min(len(S_list[i]), len(T_current))
            S_i = S_list[i][:n_min]
            T_i = T_current[:n_min]
            scale_init, R_init, trans_init, _ = Schut1973_SpaceTime.solve(
                S_i, T_i)
            angles_init = cls._matrix_to_euler(R_init)
            params_list.extend([
                scale_init,
                angles_init[0], angles_init[1], angles_init[2],
                trans_init[0], trans_init[1], trans_init[2],
            ])

        params = np.array(params_list, dtype=float)
        n_total_params = n_stations * 7
        A = np.zeros((0, n_total_params))
        L = np.zeros(0)

        for iteration in range(max_iter):
            A_list = []
            L_list = []
            for i in range(n_stations):
                S_i = S_list[i]
                T_i = T_input[i] if is_t_list else T_input
                n_pts = min(len(S_i), len(T_i))
                p = params[i*7:(i+1)*7]
                for j in range(n_pts):
                    res, J_local = cls._compute_residuals_and_jacobian_for_point(
                        S_i[j], T_i[j], p)
                    A_row = np.zeros((3, n_total_params))
                    A_row[:, i*7:(i+1)*7] = J_local
                    A_list.append(A_row)
                    L_list.append(res)

            A = np.vstack(A_list)
            L = np.concatenate(L_list)
            N = A.T @ A
            U = A.T @ L
            damping = 1e-6 * np.trace(N) / max(1, len(N)) * np.eye(len(N))
            try:
                delta = np.linalg.solve(N + damping, U)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(N + damping) @ U

            params_new = params + delta
            if np.linalg.norm(delta) < tol:
                params = params_new
                break
            params = params_new

        results = []
        for i in range(n_stations):
            p = params[i*7:(i+1)*7]
            scale = p[0]
            R = cls._euler_to_matrix(p[1:4])
            trans = p[4:7]
            results.append((scale, R, trans))

        dof = max(1, len(L) - len(params))
        sigma0_sq = np.sum(L**2) / dof
        try:
            cov_matrix = sigma0_sq * np.linalg.pinv(A.T @ A)
        except np.linalg.LinAlgError:
            cov_matrix = np.eye(len(params)) * 1e-6

        elapsed_time = time.time() - start_time
        return results, sigma0_sq, cov_matrix, L, elapsed_time


Method4_GlobalEulerLM = Method4_GlobalEulerLM_ObsEquations
# ============================================================================
# SECTION 7: METHOD 5 - TAO ET AL. (2020) CONSTRAINED MULTIVARIATE TLS
# ============================================================================


class Tao2020_CMTLS_Rigorous:
    method_code = "CMTLS"
    article_name = "Constrained Multivariate Total Least Squares for 3D similarity transformation"
    reference = "Tao, Hua and Feng (2020)."

    @staticmethod
    def solve(S, T, max_iter=30, tol=1e-6):
        start_time = time.time()
        n = S.shape[0]
        I3 = np.eye(3)
        In_1 = np.eye(n - 1)
        G = np.zeros((n - 1, n))
        for i in range(n - 1):
            G[i, 0] = -1
            G[i, i + 1] = 1
        A = G @ S
        Y = G @ T
        Q_S = np.eye(3 * n)
        Q_T = np.eye(3 * n)
        kron_I3_G = np.kron(I3, G)
        Q_A = kron_I3_G @ Q_S @ kron_I3_G.T
        Q_Y = kron_I3_G @ Q_T @ kron_I3_G.T

        k_init, R_init, _, _ = Schut1973_SpaceTime.solve(S, T)
        xi = k_init * R_init.T
        E_A = np.zeros(((n - 1), 3))
        N_inv = None
        C = None
        Q_y = None
        Q_y_inv = None
        N = None
        W = None
        W0 = None
        lagrange = None
        delta_vec_xi = None
        vec_res = None
        converged = False
        iteration = 0
        iteration_history = []
        stop_reason = 'MAXIMUM_ITERATIONS'
        if max_iter < 1 or tol <= 0:
            raise ValueError('max_iter must be positive and tol must be > 0')

        for iteration in range(max_iter):
            try:
                kron_xi_In1 = np.kron(xi.T, In_1)
                Q_y = Q_Y + kron_xi_In1 @ Q_A @ kron_xi_In1.T
                Q_y_inv = np.linalg.pinv(Q_y)
                A_j = A - E_A
                kron_I3_Aj = np.kron(I3, A_j)
                N = kron_I3_Aj.T @ Q_y_inv @ kron_I3_Aj
                N_inv = np.linalg.pinv(N)
                vec_res = (Y - A @ xi).flatten('F').reshape(-1, 1)
                W = kron_I3_Aj.T @ Q_y_inv @ vec_res
                xi_f = xi.flatten('F')
                C = np.zeros((5, 9))
                W0 = np.zeros((5, 1))
                C[0, 0:3] = 2 * xi_f[0:3]
                C[0, 3:6] = -2 * xi_f[3:6]
                W0[0, 0] = np.sum(xi_f[0:3]**2) - np.sum(xi_f[3:6]**2)
                C[1, 3:6] = 2 * xi_f[3:6]
                C[1, 6:9] = -2 * xi_f[6:9]
                W0[1, 0] = np.sum(xi_f[3:6]**2) - np.sum(xi_f[6:9]**2)
                C[2, 0:3] = xi_f[3:6]
                C[2, 3:6] = xi_f[0:3]
                W0[2, 0] = np.dot(xi_f[0:3], xi_f[3:6])
                C[3, 0:3] = xi_f[6:9]
                C[3, 6:9] = xi_f[0:3]
                W0[3, 0] = np.dot(xi_f[0:3], xi_f[6:9])
                C[4, 3:6] = xi_f[6:9]
                C[4, 6:9] = xi_f[3:6]
                W0[4, 0] = np.dot(xi_f[3:6], xi_f[6:9])

                lagrange = np.linalg.pinv(C @ N_inv @ C.T)
                delta_vec_xi = N_inv @ (W + C.T @ lagrange @
                                        (-W0 - C @ N_inv @ W))
                xi_new = xi + delta_vec_xi.reshape((3, 3), order='F')
                vec_E_A = - \
                    Q_A @ kron_xi_In1.T @ Q_y_inv @ (
                        vec_res - kron_I3_Aj @ delta_vec_xi)
                E_A = vec_E_A.reshape(((n - 1), 3), order='F')

                delta_norm = float(np.linalg.norm(delta_vec_xi))
                iteration_history.append({'iteration': iteration + 1,
                                          'increment_norm': delta_norm,
                                          'constraint_norm_before': float(np.linalg.norm(W0))})
                if delta_norm <= tol:
                    stop_reason = 'INCREMENT_TOLERANCE'

                    xi = xi_new
                    converged = True
                    break
                xi = xi_new
            except np.linalg.LinAlgError:
                stop_reason = 'LINEAR_ALGEBRA_FAILURE'
                break

        scale = np.sqrt(np.sum(xi[:, 0]**2))
        R = (xi / scale).T
        translation = np.mean(T - scale * (S @ R.T), axis=0)

        if N_inv is not None and C is not None:
            try:
                Q_xi = N_inv - \
                    N_inv @ C.T @ np.linalg.pinv(C @ N_inv @ C.T) @ C @ N_inv
            except Exception:
                Q_xi = np.eye(9) * 1e-6
        else:
            Q_xi = np.eye(9) * 1e-6

        transformed = scale * (S @ R.T) + translation
        residuals = transformed - T
        residual_norms = np.linalg.norm(residuals, axis=1)
        rmse = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
        dof = max(1, 3 * n - 7)
        residual_sum_squares = float(np.sum(residual_norms ** 2))
        sigma0_sq = residual_sum_squares / dof

        stats = {
            'method_code': Tao2020_CMTLS_Rigorous.method_code,
            'article_name': Tao2020_CMTLS_Rigorous.article_name,
            'converged': converged,
            'iterations': len(iteration_history),
            'solution_type': 'iterative',
            'stop_reason': stop_reason,
            'iteration_history': iteration_history,
            'final_increment_norm': iteration_history[-1]['increment_norm'] if iteration_history else np.nan,
            'max_iterations': max_iter,
            'tolerance': tol,
            'degrees_of_freedom': dof,
            'dof': dof,
            'sigma0_sq': sigma0_sq,
            'sigma0': float(np.sqrt(sigma0_sq)),
            'rmse': float(rmse),
            'mean_residual': float(np.mean(residual_norms)),
            'std_residual': float(np.std(residual_norms)),
            'max_residual': float(np.max(residual_norms)),
            'max_component_residual': float(np.max(np.abs(residuals))),
            'residual_sum_squares': residual_sum_squares,
            'transformed_control_points': transformed,
            'residuals_xyz': residuals,
            'residual_norms': residual_norms,
            'observation_residuals': residuals.ravel(),
            'source_centroid': np.mean(S, axis=0),
            'target_centroid': np.mean(T, axis=0),
            'G_matrix': G,
            'A_matrix': A,
            'Y_matrix': Y,
            'Q_S': Q_S,
            'Q_T': Q_T,
            'Q_A': Q_A,
            'Q_Y': Q_Y,
            'Q_y': Q_y,
            'Q_y_inverse': Q_y_inv,
            'normal_matrix': N,
            'N_matrix': N,
            'N_inverse': N_inv,
            'constraint_matrix': C,
            'constraint_misclosure': W0,
            'normal_rhs': W,
            'lagrange_multipliers': lagrange,
            'delta_xi': delta_vec_xi,
            'E_A': E_A,
            'xi_matrix': xi,
            'Q_xi': Q_xi,
            'covariance': Q_xi,
            'correlation_matrix': create_correlation_matrix(Q_xi),
        }
        elapsed_time = time.time() - start_time
        return scale, R, translation, Q_xi, stats, elapsed_time
# ============================================================================
# SECTION 8: UTILITY FUNCTIONS
# ============================================================================


def evaluate_rmse(S, T, scale, R, translation):
    transformed = scale * (S @ R.T) + translation
    diff = transformed - T
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def create_correlation_matrix(cov_matrix):
    std_devs = np.sqrt(np.clip(np.diag(cov_matrix), 0.0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
        corr_matrix[np.isnan(corr_matrix)] = 0
        corr_matrix[np.isinf(corr_matrix)] = 0
    return corr_matrix


def _matrix_inverse(matrix, allow_pseudoinverse=True):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return matrix
    try:
        condition = np.linalg.cond(matrix)
        if not np.isfinite(condition) or condition > 1e12:
            if allow_pseudoinverse:
                return np.linalg.pinv(matrix)
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        if allow_pseudoinverse:
            return np.linalg.pinv(matrix)
        raise


def _weight_matrix(P, n_obs):
    if P is None:
        return np.eye(n_obs)
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        return np.diag(P)
    return P


def _default_observation_metadata(n_obs):
    components = ("X", "Y", "Z")
    return [
        {
            "observation_index": i,
            "point_id": f"obs_{i // 3}",
            "station_id": "",
            "component": components[i % 3],
            "method": "",
            "scenario": "",
            "original_index": i,
        }
        for i in range(n_obs)
    ]


def _metadata_value(metadata, key, default=""):
    if metadata is None:
        return default
    return metadata.get(key, default)


def _baarda_empty_diagnostics(enabled, threshold, message):
    return {
        "enabled": bool(enabled),
        "threshold": threshold,
        "iterations": [],
        "removed_observations": [],
        "removed_points": [],
        "final_max_w": np.nan,
        "final_sigma0_hat": np.nan,
        "converged": False,
        "message": message,
        "outlier_report_rows": [],
        "final_residual_rows": [],
        "active_observation_indices": [],
        "residual_convention": "v = A @ X - L",
    }


def baarda_data_snooping_adjustment(
    A,
    L,
    P=None,
    observation_metadata=None,
    solve_function=None,
    threshold=3.0,
    max_iterations=10,
    min_observations_per_point=4,
    remove_strategy="observation",
    use_posterior_sigma=True,
    sigma_apriori=1.0,
    allow_pseudoinverse=True,
    residual_vector=None,
):
    """
    Perform iterative Baarda data snooping based on standardized residuals.

    Residual sign convention used here is ``v = A @ X - L`` for the default
    linear least-squares solver. If ``residual_vector`` or ``solve_function`` is
    supplied, the caller's residual sign convention is preserved.

    Returns:
        adjusted_result, baarda_diagnostics
    """
    A = np.asarray(A, dtype=float)
    L = np.asarray(L, dtype=float).reshape(-1)
    if A.ndim != 2 or A.shape[0] != L.size:
        msg = "Baarda Data Snooping not available because A and L have incompatible shapes."
        logger.warning(msg)
        return None, _baarda_empty_diagnostics(True, threshold, msg)

    n_obs = A.shape[0]
    P_full = _weight_matrix(P, n_obs)
    if P_full.shape != (n_obs, n_obs):
        msg = "Baarda Data Snooping not available because P has incompatible shape."
        logger.warning(msg)
        return None, _baarda_empty_diagnostics(True, threshold, msg)

    metadata = observation_metadata or _default_observation_metadata(n_obs)
    if len(metadata) != n_obs:
        msg = "Baarda Data Snooping not available because observation metadata length does not match A."
        logger.warning(msg)
        return None, _baarda_empty_diagnostics(True, threshold, msg)

    if not np.isfinite(A).all() or not np.isfinite(L).all() or not np.isfinite(P_full).all():
        finite_rows = np.isfinite(A).all(axis=1) & np.isfinite(L)
        finite_rows &= np.isfinite(P_full).all(axis=0) & np.isfinite(P_full).all(axis=1)
        if not finite_rows.any():
            msg = "Baarda Data Snooping not available because all observations are NaN or infinite."
            logger.warning(msg)
            return None, _baarda_empty_diagnostics(True, threshold, msg)
        logger.warning("Baarda Data Snooping ignored NaN/infinite observations.")
    else:
        finite_rows = np.ones(n_obs, dtype=bool)

    active = finite_rows.copy()
    removed_observations = []
    removed_points = []
    iteration_rows = []
    report_rows = []
    final_rows = []
    adjusted_result = None
    final_max_w = np.nan
    final_sigma0_hat = np.nan
    converged = False
    message = "Maximum iterations reached."

    def solve_active(active_mask):
        active_idx = np.flatnonzero(active_mask)
        A_i = A[active_idx]
        L_i = L[active_idx]
        P_i = P_full[np.ix_(active_idx, active_idx)]
        if solve_function is not None:
            result = solve_function(A_i, L_i, P_i, active_idx)
            X_i = np.asarray(result.get("X", np.array([])), dtype=float).reshape(-1)
            v_i = np.asarray(result.get("residuals", A_i @ X_i - L_i), dtype=float).reshape(-1)
            Qxx_i = result.get("Qxx")
        else:
            N_i = A_i.T @ P_i @ A_i
            Qxx_i = _matrix_inverse(N_i, allow_pseudoinverse=allow_pseudoinverse)
            X_i = Qxx_i @ A_i.T @ P_i @ L_i
            v_i = A_i @ X_i - L_i
            result = {}

        if Qxx_i is None:
            N_i = A_i.T @ P_i @ A_i
            Qxx_i = _matrix_inverse(N_i, allow_pseudoinverse=allow_pseudoinverse)

        Qll_i = _matrix_inverse(P_i, allow_pseudoinverse=allow_pseudoinverse)
        Qvv_i = Qll_i - A_i @ Qxx_i @ A_i.T
        dof_i = int(A_i.shape[0] - A_i.shape[1])
        if dof_i <= 0:
            return {
                **result,
                "X": X_i,
                "residuals": v_i,
                "Qxx": Qxx_i,
                "Qvv": Qvv_i,
                "P": P_i,
                "active_indices": active_idx,
                "dof": dof_i,
                "sigma0_hat": np.nan,
                "w": np.full(A_i.shape[0], np.nan),
                "redundancy_numbers": np.full(A_i.shape[0], np.nan),
            }

        sigma_sq = float(v_i.T @ P_i @ v_i / dof_i)
        sigma_hat = float(np.sqrt(max(sigma_sq, 0.0))) if use_posterior_sigma else float(sigma_apriori)
        qvv_diag = np.diag(Qvv_i)
        w_i = np.full(A_i.shape[0], np.nan)
        valid = np.isfinite(v_i) & np.isfinite(qvv_diag) & (qvv_diag > 0) & (sigma_hat > 0)
        if np.any(~valid):
            logger.warning("Baarda Data Snooping ignored observations with invalid Qvv or sigma.")
        w_i[valid] = v_i[valid] / (sigma_hat * np.sqrt(qvv_diag[valid]))
        if np.allclose(P_i, np.diag(np.diag(P_i))):
            redundancy = np.diag(P_i) * qvv_diag
        else:
            redundancy = np.full(A_i.shape[0], np.nan)
        return {
            **result,
            "X": X_i,
            "residuals": v_i,
            "Qxx": Qxx_i,
            "Qvv": Qvv_i,
            "P": P_i,
            "active_indices": active_idx,
            "dof": dof_i,
            "sigma0_hat": sigma_hat,
            "w": w_i,
            "redundancy_numbers": redundancy,
        }

    for iteration in range(int(max_iterations) + 1):
        adjusted_result = solve_active(active)
        active_indices = adjusted_result["active_indices"]
        dof = adjusted_result["dof"]
        if dof <= 0:
            message = "Baarda Data Snooping not executed because degrees of freedom <= 0."
            logger.warning(message)
            break

        w = adjusted_result["w"]
        abs_w = np.abs(w)
        if np.all(~np.isfinite(abs_w)):
            message = "Baarda Data Snooping stopped because no finite standardized residuals were available."
            logger.warning(message)
            break

        local_max = int(np.nanargmax(abs_w))
        obs_idx = int(active_indices[local_max])
        meta = metadata[obs_idx]
        point_id = _metadata_value(meta, "point_id", "")
        component = _metadata_value(meta, "component", "")
        station_id = _metadata_value(meta, "station_id", "")
        method = _metadata_value(meta, "method", "")
        scenario = _metadata_value(meta, "scenario", "")
        max_w = float(abs_w[local_max])
        final_max_w = max_w
        final_sigma0_hat = float(adjusted_result["sigma0_hat"])
        qvv_value = float(np.diag(adjusted_result["Qvv"])[local_max])
        redundancy_value = float(adjusted_result["redundancy_numbers"][local_max])
        residual_value = float(adjusted_result["residuals"][local_max])
        decision = "kept"
        reason = "below_threshold"
        remove_indices = []

        if max_w > threshold and remove_strategy != "none":
            if not point_id:
                decision = "not_removed_minimum_observations"
                reason = "missing_observation_metadata"
            elif remove_strategy == "point":
                remove_indices = [
                    i for i, m in enumerate(metadata)
                    if active[i] and _metadata_value(m, "point_id", None) == point_id
                ]
                active_points_after = sum(
                    1 for i, m in enumerate(metadata)
                    if active[i] and i not in remove_indices and _metadata_value(m, "point_id", None) == point_id
                )
                if active_points_after < 0:
                    remove_indices = []
            else:
                same_point_active = [
                    i for i, m in enumerate(metadata)
                    if active[i] and _metadata_value(m, "point_id", None) == point_id
                ]
                if len(same_point_active) > 1 and len(same_point_active) - 1 < min_observations_per_point:
                    decision = "not_removed_minimum_observations"
                    reason = "minimum_observations_per_point"
                else:
                    remove_indices = [obs_idx]

            if remove_indices:
                for remove_idx in remove_indices:
                    active[remove_idx] = False
                    removed_meta = metadata[remove_idx]
                    removed_observations.append({
                        "observation_index": int(remove_idx),
                        "point_id": _metadata_value(removed_meta, "point_id", ""),
                        "station_id": _metadata_value(removed_meta, "station_id", ""),
                        "component": _metadata_value(removed_meta, "component", ""),
                        "iteration": int(iteration),
                    })
                if remove_strategy == "point" and point_id not in removed_points:
                    removed_points.append(point_id)
                decision = "removed"
                reason = "threshold_exceeded"
        elif max_w > threshold:
            decision = "kept"
            reason = "remove_strategy_none"

        iteration_record = {
            "iteration": int(iteration),
            "num_observations": int(active.sum()),
            "dof": int(dof),
            "sigma0_hat": final_sigma0_hat,
            "max_abs_w": max_w,
            "max_w_observation_index": obs_idx,
            "point_id": point_id,
            "component": component,
            "raw_residual": residual_value,
            "standardized_residual_w": float(w[local_max]),
            "qvv": qvv_value,
            "redundancy_number": redundancy_value,
            "decision": decision,
        }
        iteration_rows.append(iteration_record)
        report_rows.append({
            "iteration": int(iteration),
            "method": method,
            "scenario": scenario,
            "observation_index": obs_idx,
            "point_id": point_id,
            "station_id": station_id,
            "component": component,
            "residual": residual_value,
            "qvv": qvv_value,
            "redundancy_number": redundancy_value,
            "standardized_residual_w": float(w[local_max]),
            "threshold": float(threshold),
            "removed": decision == "removed",
            "reason": reason,
        })

        if decision != "removed":
            converged = max_w <= threshold
            message = "Converged: no standardized residual exceeded the threshold." if converged else reason
            break

    if adjusted_result is not None:
        active_indices = adjusted_result["active_indices"]
        qvv_diag = np.diag(adjusted_result["Qvv"])
        removed_set = {row["observation_index"] for row in removed_observations}
        for local_i, obs_idx in enumerate(active_indices):
            meta = metadata[int(obs_idx)]
            w_val = adjusted_result["w"][local_i]
            final_rows.append({
                "method": _metadata_value(meta, "method", ""),
                "scenario": _metadata_value(meta, "scenario", ""),
                "observation_index": int(obs_idx),
                "point_id": _metadata_value(meta, "point_id", ""),
                "station_id": _metadata_value(meta, "station_id", ""),
                "component": _metadata_value(meta, "component", ""),
                "residual": float(adjusted_result["residuals"][local_i]),
                "qvv": float(qvv_diag[local_i]),
                "redundancy_number": float(adjusted_result["redundancy_numbers"][local_i]),
                "standardized_residual_w": float(w_val) if np.isfinite(w_val) else np.nan,
                "is_outlier_candidate": bool(np.isfinite(w_val) and abs(w_val) > threshold),
                "was_removed": int(obs_idx) in removed_set,
            })

    diagnostics = {
        "enabled": True,
        "threshold": threshold,
        "iterations": iteration_rows,
        "removed_observations": removed_observations,
        "removed_points": removed_points,
        "final_max_w": float(final_max_w) if np.isfinite(final_max_w) else np.nan,
        "final_sigma0_hat": float(final_sigma0_hat) if np.isfinite(final_sigma0_hat) else np.nan,
        "converged": bool(converged),
        "message": message,
        "outlier_report_rows": report_rows,
        "final_residual_rows": final_rows,
        "active_observation_indices": [int(i) for i in np.flatnonzero(active)],
        "residual_convention": "v = A @ X - L",
    }
    return adjusted_result, diagnostics


def propagate_point_covariance(point_local, params, cov_params, quat_to_matrix_func):
    eps = 1e-6

    def transform_point(pt, p):
        k = p[0]
        quat = p[1:5]
        trans = p[5:8]
        R_m = quat_to_matrix_func(quat)
        return k * (R_m @ pt) + trans
    J = np.zeros((3, 8))
    for i in range(8):
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        f_plus = transform_point(point_local, p_plus)
        f_minus = transform_point(point_local, p_minus)
        J[:, i] = (f_plus - f_minus) / (2 * eps)
    return J @ cov_params @ J.T



