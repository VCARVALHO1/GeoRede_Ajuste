"""Baarda local w-tests and reliability for a specified linearized stochastic model.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm, chi2
from scipy.optimize import brentq


def noncentral_shift(critical_value, power=0.8):
    if not 0.5 < power < 1:
        raise ValueError('Detection power must be between 0.5 and 1')
    return float(brentq(lambda d: norm.sf(critical_value-d)+norm.cdf(-critical_value-d)-power,
                       0., critical_value+12.))


def affine_design(source):
    """Centred affine coefficients, followed by translation at the source centroid."""
    points=np.asarray(source,float); centered=points-points.mean(axis=0)
    A=np.zeros((3*len(points),12))
    for i,p in enumerate(centered):
        for c in range(3):
            A[3*i+c,3*c:3*c+3]=p
            A[3*i+c,9+c]=1.
    return A


def similarity_design(source, scale, rotation, fixed_scale=False):
    """Local proper-rotation tangent, translation at centroid, optional scale."""
    points=np.asarray(source,float); centered=points-points.mean(axis=0)
    A=np.zeros((3*len(points),6 if fixed_scale else 7))
    for i,p in enumerate(centered):
        rp=np.asarray(rotation)@p; x,y,z=scale*rp
        skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        A[3*i:3*i+3,:3]=-skew
        A[3*i:3*i+3,3:6]=np.eye(3)
        if not fixed_scale:A[3*i:3*i+3,6]=rp
    return A


def baarda_reliability(A, residuals, *, P=None, sigma_apriori=0.002,
                      alpha_local=None, critical_value=3., alpha_global=0.05,
                      power=0.8, metadata=None, parameter_basis=None,
                      solver_converged=True, scope='linearized_model',
                      stationarity_tolerance=1e-3):
    A=np.asarray(A,float); v=np.asarray(residuals,float).reshape(-1)
    if A.ndim!=2 or A.shape[0]!=v.size or not np.isfinite(A).all() or not np.isfinite(v).all():
        raise ValueError('Finite, dimensionally consistent design and residuals required')
    if sigma_apriori<=0 or not 0<alpha_global<1:raise ValueError('Invalid sigma or significance')
    m,p=A.shape
    P=np.eye(m) if P is None else np.asarray(P,float)
    if P.shape!=(m,m) or not np.allclose(P,P.T):raise ValueError('P must be symmetric')
    W=np.linalg.cholesky(P).T
    Q=np.linalg.solve(P,np.eye(m))
    B=W@A
    scales=np.linalg.norm(B,axis=0)
    if np.any(scales==0):raise ValueError('Unobservable parameter columns; use an identifiable parameterization')
    U,s,Vt=np.linalg.svd(B/scales,full_matrices=False)
    threshold=max(B.shape)*np.finfo(float).eps*s[0]
    rank=int(np.count_nonzero(s>threshold)); dof=m-rank
    if rank<p:raise ValueError(f'Rank deficient parameterization: {rank}/{p}')
    Bplus=(Vt.T/s)@U.T/scales[:,None]
    gain=Bplus@W
    Qxx=Bplus@Bplus.T
    # Orthogonal projector in whitened space avoids cancellation in I-AQxxA'.
    Winv=np.linalg.solve(W,np.eye(m))
    Qvv=Winv@(np.eye(m)-U@U.T)@Winv.T; Qvv=(Qvv+Qvv.T)/2
    S=P@Qvv@P; S=(S+S.T)/2
    sii=np.diag(S)
    estimable=sii>max(1.,float(np.max(np.diag(P))))*1e-10
    redundancy=np.diag(Qvv@P)
    whiten_v=W@v
    projected=U.T@whiten_v
    stationarity=float(np.linalg.norm(projected)/max(np.linalg.norm(whiten_v),sigma_apriori*np.sqrt(m)))
    eligible=bool(solver_converged is not False and dof>0 and stationarity<=stationarity_tolerance)
    reason=('SOLVER_NOT_CONVERGED' if solver_converged is False else
            'NO_REDUNDANCY' if dof<=0 else
            'RESIDUALS_NOT_ORTHOGONAL_TO_MODEL' if stationarity>stationarity_tolerance else 'OK')
    alpha_local=float(2*norm.sf(critical_value)) if alpha_local is None else float(alpha_local)
    if not 0<alpha_local<1:raise ValueError('Invalid local significance')
    critical_value=float(norm.isf(alpha_local/2))
    family_critical=float(norm.isf(alpha_global/(2*max(1,int(estimable.sum())))))
    delta=noncentral_shift(critical_value,power)
    delta_family=noncentral_shift(family_critical,power)
    w=np.full(m,np.nan); mdb=np.full(m,np.nan); mdb_family=np.full(m,np.nan)
    w[estimable]=(P@v)[estimable]/(sigma_apriori*np.sqrt(sii[estimable]))
    mdb[estimable]=sigma_apriori*delta/np.sqrt(sii[estimable])
    mdb_family[estimable]=sigma_apriori*delta_family/np.sqrt(sii[estimable])
    effects=gain*mdb[None,:]
    original_effects=effects if parameter_basis is None else np.asarray(parameter_basis)@effects
    external_standardized=np.linalg.norm(W@A@effects,axis=0)/sigma_apriori
    statistic=float(v@P@v/sigma_apriori**2)
    global_test={
        'statistic':statistic,'degrees_of_freedom':dof,'alpha':alpha_global,
        'upper_critical':float(chi2.isf(alpha_global,dof)) if dof>0 else None,
        'bilateral_lower':float(chi2.ppf(alpha_global/2,dof)) if dof>0 else None,
        'bilateral_upper':float(chi2.isf(alpha_global/2,dof)) if dof>0 else None,
        'p_upper':float(chi2.sf(statistic,dof)) if dof>0 else None,
        'p_lower':float(chi2.cdf(statistic,dof)) if dof>0 else None,
        'reject_upper':bool(dof>0 and statistic>chi2.isf(alpha_global,dof)),
        'below_bilateral_lower':bool(dof>0 and statistic<chi2.ppf(alpha_global/2,dof)),
        'eligible_for_inference':eligible,
    }
    rows=[]
    for i in range(m):
        row=dict(metadata[i]) if metadata and i<len(metadata) else {}
        row.update(observation_index=i,residual=float(v[i]),redundancy_number=float(redundancy[i]),
                   qvv=float(Qvv[i,i]),sensitivity_sii=float(sii[i]),w=float(w[i]),
                   p_local=float(2*norm.sf(abs(w[i]))),critical_value=critical_value,
                   alpha_local=alpha_local,alpha_family=alpha_global,family_critical=family_critical,
                   detected_local=bool(eligible and estimable[i] and abs(w[i])>critical_value),
                   detected_bonferroni=bool(eligible and estimable[i] and abs(w[i])>family_critical),
                   detectable=bool(estimable[i]),mdb_m=float(mdb[i]),mdb_bonferroni_m=float(mdb_family[i]),
                   external_standardized=float(external_standardized[i]),power=power,
                   eligible_for_inference=eligible,validity_reason=reason)
        rows.append(row)
    return {'scope':scope,'rank':rank,'degrees_of_freedom':dof,'n_observations':m,
            'sigma_apriori_m':sigma_apriori,'alpha_local':alpha_local,'critical_value':critical_value,
            'power':power,'noncentrality':delta**2,'eligible_for_inference':eligible,
            'validity_reason':reason,'stationarity_relative':stationarity,
            'condition_number_scaled':float(s[0]/s[-1]),'redundancy_sum':float(redundancy.sum()),
            'global_test':global_test,'observations':rows,'Qvv':Qvv,'Qxx':Qxx,
            'parameter_bias_at_mdb':original_effects,'gain':gain}
