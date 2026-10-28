"""Reproduce article workflows and export controlled reliability comparisons.

Run through python run.py using the documented environment.
Each output folder must be new. Input data and reference results are read-only.
"""
from __future__ import annotations
import argparse,sys,json,hashlib,platform,copy
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--source-root',type=Path,default=ROOT)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--data',type=Path,default=ROOT/'data')
parser.add_argument('--profile',choices=['legacy','baarda_mad','baarda_only','network_baarda_mad'],default='network_baarda_mad')
parser.add_argument('--max-iter',type=int,default=50)
parser.add_argument('--station-max-iter',type=int,default=25,help='Separate iteration limit for station-wise methods; defaults to 25.')
parser.add_argument('--network-tol',type=float,default=1e-8,help='N-LM increment tolerance; station-wise tolerance is unchanged.')
parser.add_argument('--linear-solver',choices=['legacy_normal','constrained_scaled_svd','adaptive_lm'],default='adaptive_lm')
parser.add_argument('--methods',nargs='+',default=['SVD-P','SCH-Q','Q-ST','CMTLS','TFS','N-CUQ-LM'])
parser.add_argument('--sigma-mm',type=float,default=2.)
parser.add_argument('--power',type=float,default=.8)
parser.add_argument('--critical-value',type=float,default=3.)
parser.add_argument('--alpha-global',type=float,default=.05)
parser.add_argument('--max-cycles',type=int,default=20)
parser.add_argument('--network-geometry',choices=['chained','coupled'],default='coupled')
cli=parser.parse_args()
sys.path.insert(0,str(cli.source_root.resolve()))
sys.path.insert(0,str(cli.source_root.resolve()/'src'))
import analyze_methods_data as an
import outlier_statistics as oscreen
if cli.profile!='legacy':
    from freesurvey.reliability import baarda_reliability,affine_design,similarity_design

DATA=cli.data.resolve()
if cli.out.exists():raise RuntimeError('Output already exists; choose a new folder to preserve traceability')
cli.out.mkdir(parents=True)
base=json.loads((cli.source_root.resolve()/'config/base.json').read_text(encoding='utf8'))
base.update(data=str(DATA),out=str(cli.out),methods=list(cli.methods),max_iter=cli.max_iter,use_quaternion_constraint=True,
            outlier_sigma=cli.critical_value,outlier_percentile=95.,outlier_rms_factor=2.5,
            outlier_component_sigma=3.,min_common_cp=4,control_points=None,
            statistical_profile=cli.profile,network_linear_solver=cli.linear_solver,apriori_sigma_mm=cli.sigma_mm,apriori_sigma_m=cli.sigma_mm/1000,
            local_test_critical_value=cli.critical_value,global_test_alpha=cli.alpha_global,
            detection_power=cli.power,source_root=str(cli.source_root),
            local_multiple_testing='per-observation; Bonferroni diagnostics also exported',
            global_gate_for_removal=cli.profile=='baarda_only')
base['network_geometry']=cli.network_geometry
base['station_max_iter']=cli.station_max_iter if cli.station_max_iter is not None else cli.max_iter
base['network_tol']=cli.network_tol if cli.network_tol is not None else base['tol']
base['network_trial_policy']='retain >=4 total shared controls per station; connected graph; full rank; convergence; valid Baarda; rollback on failure' if cli.network_geometry=='coupled' else 'chained initialization reachability'
base['screening_by_method']={m:('mad' if cli.profile=='network_baarda_mad' and m!='N-CUQ-LM' else ('baarda' if cli.profile=='baarda_only' else 'baarda,mad')) for m in cli.methods}
if cli.profile!='legacy':
    base['global_variance_test_note']='A priori sigma normalizes global/local tests and MDB; adjustment weights remain P=I.'
args=SimpleNamespace(**base)
classes={'SVD-P':an.SVD_P,'SCH-Q':an.SCH_Q,'Q-ST':an.Q_ST,'CMTLS':an.CMTLS,'TFS':an.TFS,'N-CUQ-LM':an.N_CUQ_LM}
if not all(cls.__module__.startswith('freesurvey.') for cls in classes.values()):
    raise RuntimeError('Full numerical method imports are required. Install the documented dependencies; fallback implementations are not accepted for article reproduction.')
stations=an.load_station_data(DATA)
an.apply_independent_check_points(stations,[])
manifest={'configuration':base,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
          'data_sha256':{str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.rglob('*') if p.is_file()},
          'source_sha256':{str(p.resolve().relative_to(cli.source_root.resolve())):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in list((cli.source_root/'src').rglob('*.py'))+[cli.source_root/'src/analyze_methods_data.py',cli.source_root/'src/outlier_statistics.py',Path(__file__)]}}
(cli.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')

def dump(path,value):
    def clean(x):
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,(np.integer,np.floating,np.bool_)):x=x.item()
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def network_output(result):
    return next(iter(result['outputs'].values()))

def enrich(result,method,folder,cycle):
    """Attach model-specific tests without changing the transformation estimates."""
    diagnostics=[];summary=[];effects={}
    if method=='N-CUQ-LM':
        obj=network_output(result)
        A=obj['statistical_design_matrix']
        # Kernel stores target-minus-transformed closures. Export v = transformed-minus-target.
        v=-np.asarray(obj['observation_residuals'],dtype=float)
        names=[s['station'] for s in obj['stations']]
        meta=[{'station':names[int(x['station_index'])],'point':str(x['point_id']),
               'component':('X','Y','Z')[int(x['component'])]} for x in obj['coordinate_metadata']]
        groups=[('NETWORK',obj,A,v,meta,obj['statistical_parameter_basis'],f'coupled_network_P_identity_sigma{cli.sigma_mm:g}mm')]
    else:
        groups=[]
        for station,obj in result['outputs'].items():
            if obj.get('is_reference'):continue
            source=obj['input_source_coordinates']
            A=affine_design(source) if method=='Q-ST' else similarity_design(source,obj['scale'],obj['rotation_matrix'],method=='TFS')
            fitrows=[x for x in result['residual_rows'] if x['station']==station]
            meta=[{'station':station,'point':str(x['point']),'component':c} for x in fitrows for c in ('X','Y','Z')]
            scope='conditional_fixed_accumulated_reference; reference_uncertainty_not_propagated'
            if method=='CMTLS':scope+='; similarity_linearization_not_full_TLS_stochastic_model'
            groups.append((station,obj,A,obj['residuals_xyz'].ravel(),meta,None,scope))
    lookup={}
    for station,obj,A,v,meta,basis,scope in groups:
        try:
            r=baarda_reliability(A,v,sigma_apriori=cli.sigma_mm/1000,
                  critical_value=cli.critical_value,alpha_global=cli.alpha_global,power=cli.power,
                  metadata=meta,parameter_basis=basis,solver_converged=obj.get('converged'),scope=scope)
        except ValueError as exc:
            summary.append({'station':station,'cycle':cycle,'eligible_for_inference':False,'validity_reason':str(exc)})
            continue
        r['post_selection']=cycle>0
        r['scope']=scope
        rows=r.pop('observations');diagnostics.extend([{'cycle':cycle,'method':method,**x} for x in rows])
        for key in ['Qvv','Qxx','parameter_bias_at_mdb','gain']:effects[f'{station}__{key}']=r.pop(key)
        summary.append({'station':station,'cycle':cycle,**r})
        for d in rows:
            key=(d['station'],d['point']);entry=lookup.setdefault(key,{'score':np.nan,'flag':False})
            gate=cli.profile!='baarda_only' or r['global_test']['reject_upper']
            value=abs(d['w']) if d['eligible_for_inference'] and gate else np.nan
            if np.isfinite(value) and (not np.isfinite(entry['score']) or value>entry['score']):entry['score']=value
            entry['flag']|=d['detected_local'] and gate
            entry['w_'+d['component'].lower()]=d['w']
            entry['eligible']=d['eligible_for_inference'];entry['reason']=d['validity_reason']
        obj['reliability_summary']=r
    for row in result['residual_rows']:
        entry=lookup.get((row['station'],str(row['point'])),{})
        row.update(baarda_score=entry.get('score',np.nan),flag_baarda=entry.get('flag',False),
                   baarda_available=entry.get('eligible',False),baarda_unavailable_reason=entry.get('reason','NO_MODEL'))
        for c in ('x','y','z'):row['w_'+c]=entry.get('w_'+c,np.nan)
    pd.DataFrame(diagnostics).to_csv(folder/'baarda_observations.csv',index=False)
    dump(folder/'baarda_summary.json',summary)
    np.savez_compressed(folder/'reliability_arrays.npz',**effects)
    return summary

def common_icp(result,method):
    if method=='N-CUQ-LM':
        return result['independent_rows']
    sets=network_output(result)['transformed_stations'] if method=='N-CUQ-LM' else [dict(v,name=k) for k,v in result['outputs'].items()]
    pools=defaultdict(list)
    for st in sets:
        name=st.get('name',st.get('station',''))
        for pid in st.get('icp_ids',[]):
            if pid in st['all_ids']:pools[str(pid)].append((name,np.asarray(st['coords'])[st['all_ids'].index(pid)]))
    rows=[]
    for pid,items in pools.items():
        if len(items)<2:continue
        for i,(station,coord) in enumerate(items):
            ref=np.mean([x[1] for j,x in enumerate(items) if j!=i],axis=0);v=coord-ref
            rows.append({'station':station,'point':pid,'vx':v[0],'vy':v[1],'vz':v[2],'v_norm':float(np.linalg.norm(v))})
    return rows

def record(result,method,folder,cycle):
    folder.mkdir(parents=True)
    use_baarda=cli.profile!='legacy' and not (cli.profile=='network_baarda_mad' and method!='N-CUQ-LM')
    diagnostic_summary=enrich(result,method,folder,cycle) if use_baarda else []
    if cli.profile=='network_baarda_mad' and method!='N-CUQ-LM':
        for row in result['residual_rows']:
            row.update(flag_baarda=False,baarda_score=np.nan,baarda_available=False,baarda_unavailable_reason='NOT_APPLIED_MAD_ONLY')
            for component in ('x','y','z'):row['w_'+component]=np.nan
    for key in ['station_rows','residual_rows','independent_rows']:
        pd.DataFrame(result.get(key,[])).to_csv(folder/(key+'.csv'),index=False)
    an.save_full_numeric_outputs(folder,{method:result['outputs']})
    dump(folder/'outputs_full_summary.json',an.jsonable(result['outputs']))
    icp=common_icp(result,method);pd.DataFrame(icp).to_csv(folder/'common_icp.csv',index=False)
    norms=np.array([x['v_norm'] for x in result['residual_rows']])
    iteration_rows=[];histories=[]
    objs=[('NETWORK',network_output(result))] if method=='N-CUQ-LM' else result['outputs'].items()
    for name,o in objs:
        if o.get('is_reference'):continue
        iteration_rows.append({'method':method,'station':name,'cycle':cycle,
             **{k:o.get(k) for k in ['iterations','solution_type','converged','stop_reason','max_iterations','tolerance','final_increment_norm','dof','degrees_of_freedom','sigma0_sq']}})
        histories.extend([{'method':method,'station':name,'cycle':cycle,**x} for x in o.get('iteration_history',[])])
    pd.DataFrame(iteration_rows).to_csv(folder/'solver_iterations.csv',index=False)
    pd.DataFrame(histories).to_csv(folder/'iteration_history.csv',index=False)
    if method=='N-CUQ-LM':
        pd.DataFrame(network_output(result).get('lm_trial_history',[])).to_csv(folder/'lm_trials.csv',index=False)
    summary={'method':method,'cycle':cycle,'control_rmse_mm':float(np.sqrt(np.mean(norms**2))*1000),
             'n_control_rows':len(norms),'n_icp_common':len(icp),
             'common_icp_rmse_mm':float(np.sqrt(np.mean([r['v_norm']**2 for r in icp]))*1000) if icp else None,
             'n_successful_stations':sum(bool(x.get('success',True)) for x in result['station_rows']),
             'solver_status':iteration_rows,
             'statistical_validity':[{'station':x['station'],'eligible':x['eligible_for_inference'],'reason':x['validity_reason']} for x in diagnostic_summary]}
    dump(folder/'summary.json',summary)
    return summary

all_summaries=[]
for method in cli.methods:
    args.max_iter=cli.max_iter if method=='N-CUQ-LM' else base['station_max_iter']
    args.tol=base['network_tol'] if method=='N-CUQ-LM' else base['tol']
    method_dir=cli.out/method.replace('-','_');method_dir.mkdir()
    run=an.run_network_method if method=='N-CUQ-LM' else an.run_sequential_method
    exclusions=defaultdict(set);trace=[];stop='MAXIMUM_CYCLES_REACHED';cached_result=None;trial_log=[]
    args.initial_network_result=None
    if cli.profile=='baarda_only':args.outlier_methods=['baarda']
    elif cli.profile=='network_baarda_mad' and method!='N-CUQ-LM':args.outlier_methods=['mad']
    else:args.outlier_methods=['all']
    for cycle in range(cli.max_cycles+1):
        print(f'RUN {cli.profile} {method} cycle={cycle}',flush=True)
        if cached_result is None:
            result=run(method,classes[method],stations,args,excluded_points=exclusions,run_label=f'cycle_{cycle:02d}')
        else:
            result=cached_result;cached_result=None

        if not result['residual_rows']:raise RuntimeError(f'Adjustment failed: {method} cycle {cycle}: {result}')
        folder=method_dir/f'cycle_{cycle:02d}'
        summary=record(result,method,folder,cycle);all_summaries.append(summary)
        flags,_=oscreen.evaluate_outliers(pd.DataFrame(result['residual_rows']),args)
        flags.to_csv(folder/'outlier_flags.csv',index=False)
        if cycle==cli.max_cycles:break
        accepted_trial={}
        def validate_candidate(st,pt):
            if method!='N-CUQ-LM' or cli.network_geometry!='coupled':
                return oscreen.network_exclusion_preserves_connectivity(stations,exclusions,st,pt,args.min_common_cp)
            pre=oscreen.coupled_network_exclusion_precheck(stations,exclusions,st,pt,args.min_common_cp)
            log={'cycle_selected':cycle,'station':st,'point':pt,'precheck':pre}
            if not pre['accepted']:
                trial_log.append(log);return pre
            trial_exclusions=copy.deepcopy(exclusions);trial_exclusions[('*',st)].add(pt)
            args.initial_network_result=network_output(result)
            try:
                trial=run(method,classes[method],stations,args,excluded_points=trial_exclusions,run_label=f'cycle_{cycle+1:02d}')
                trial_folder=method_dir/'trials'/f'from_{cycle:02d}_{st}_{pt}'
                trial_folder.mkdir(parents=True,exist_ok=True)
                if trial.get('outputs'):
                    trial_obj=network_output(trial)
                    dump(trial_folder/'solver_diagnostics.json', {k:trial_obj.get(k) for k in
                         ['converged','iterations','stop_reason','final_increment_norm','final_objective','iteration_history','lm_trial_history']})
                check=oscreen.coupled_solution_validation(trial,[s['name'] for s in stations])
                if check['accepted']:
                    trial_folder=method_dir/'trials'/f'from_{cycle:02d}_{st}_{pt}'
                    trial_folder.mkdir(parents=True,exist_ok=True)
                    diag=enrich(trial,method,trial_folder,cycle+1)
                    if not diag or not all(d.get('eligible_for_inference',False) for d in diag):
                        check.update(accepted=False,reason='INVALID_STATISTICAL_DIAGNOSTICS')
                if check['accepted']:
                    accepted_trial['result']=trial
                log['trial']=check
            except Exception as exc:
                check={'accepted':False,'reason':'TRIAL_EXCEPTION','detail':str(exc)};log['trial']=check
            finally:
                args.initial_network_result=None
            trial_log.append(log)
            dump(method_dir/'candidate_trials.json',trial_log)
            return check
        selected,rows=oscreen.select_outlier_exclusions(flags,args.min_common_cp,1,candidate_validator=validate_candidate)
        if accepted_trial:
            cached_result=accepted_trial['result']
        dump(method_dir/'candidate_trials.json',trial_log)
        if not oscreen.merge_outlier_exclusions(exclusions,selected):
            stop='NO_ADMISSIBLE_EXCLUSION' if (not flags.empty and flags['is_outlier_candidate'].any()) else 'NO_FLAGGED_OBSERVATIONS'
            break
        trace.extend([{'cycle_selected':cycle,**r} for r in rows])
    dump(method_dir/'final.json',{'final_cycle':cycle,'stop_reason':stop,'exclusions':{str(k):sorted(v) for k,v in exclusions.items()},'summary':summary})
    pd.DataFrame(trace).to_csv(method_dir/'exclusion_trace.csv',index=False)
dump(cli.out/'all_summaries.json',all_summaries)
print('COMPLETE '+str(cli.out),flush=True)
