#!/usr/bin/env python3
"""Run all twelve projects on a new, labelled, post-lockbox synthetic track.

No network calls, no trading, no original lockbox access, no result overwriting.
See protocol/PORTFOLIO_RESEARCH_V1.md for design and interpretation restrictions.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.metadata, json, os, platform, sys, time, traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from wharton_lab.research.causal_portfolio import *
from wharton_lab.models.m01 import M01RegimeDistributionalGBM
from wharton_lab.models.m01.config import M01Config
from wharton_lab.models.m02 import M02FactorResidualRanker
from wharton_lab.models.m02.config import M02Config
from wharton_lab.models.m03 import M03ConditionalNonlinearFactor
from wharton_lab.models.m03.config import M03Config
from wharton_lab.models.m03.factors import pca_factors, fit_ar1
from wharton_lab.models.m04 import M04FinancialAPEN
from wharton_lab.models.m04.config import M04Config
from wharton_lab.models.m05 import QAPENModel
from wharton_lab.models.m05.config import M05Config
from wharton_lab.models.m06 import FIJEPAModel
from wharton_lab.models.m06.config import M06Config
from wharton_lab.models.m07 import EigenJEPAModel
from wharton_lab.models.m07.config import M07Config
from wharton_lab.models.m08 import GraphFIJEPAModel
from wharton_lab.models.m08.config import M08Config
from wharton_lab.models.m09 import M09Model, M09Config
from wharton_lab.models.m10 import M10Model, M10Config
from wharton_lab.models.m11 import M11Model, M11Config
from wharton_lab.models.m11.accounting import simulate_ledger, empirical_cvar
from wharton_lab.models.m12 import M12Model, M12Config

UPSTREAM='8e3d790dd513a93b7d80d5726a8b2708e8615b71'
ROOT=Path(__file__).resolve().parents[1]

def json_default(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path): return str(x)
    raise TypeError(type(x).__name__)

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=json_default,allow_nan=False)+'\n')

def mse_losses(pred,y):
    z=(np.asarray(pred)-np.asarray(y))**2
    return z if z.ndim==1 else z.mean(axis=tuple(range(1,z.ndim)))

def ridge_predictions(a,y,b):
    return Ridge(alpha=1).fit(a.reshape(len(a),-1),y).predict(b.reshape(len(b),-1))

def pinball_losses(y,p,taus):
    err=y[:,None]-p
    return np.maximum(taus*err,(taus-1)*err).mean(axis=1)

def common_point(r,train,test):
    a=point_features(r,train); b=point_features(r,test)
    y=r[train,0]; truth=r[test,0]
    return a,b,y,truth

def evaluate(mid,ablate,d):
    r=d.normalized; train=d.train_origins; test=d.test_origins
    a,b,y,truth=common_point(r,train,test)
    raw={'train_origins':train,'test_origins':test,'truth':truth}
    info={'model_title':MODEL_TITLES[mid],'data_scale':d.scale,'train_rows':len(train),'eval_origins':len(test),
          'direction':'lower','task':'point_forecast','primary_metric':'mse_normalized_return',
          'baseline_name':'information_matched_ridge','extra_metrics':{}}
    rng=np.random.default_rng(d.seed+600)
    m=None
    if mid=='M01':
        cfg=M01Config(n_estimators=30,max_depth=2,min_samples_leaf=12,vol_threshold=1.4,
                      use_regimes=not ablate,random_state=d.seed)
        m=M01RegimeDistributionalGBM(cfg).fit(a,y)
        pred=m.predict_quantiles(b); taus=np.asarray(cfg.quantiles)
        # Fit/calibration split is entirely inside training; no holdout outcomes used.
        cut=int(.8*len(train)); base=Ridge(alpha=1).fit(a[:cut],y[:cut])
        residual=y[cut:]-base.predict(a[cut:])
        ref=base.predict(b)[:,None]+np.quantile(residual,taus)[None,:]
        historical=np.broadcast_to(np.quantile(y,taus),(len(test),len(taus)))
        losses=pinball_losses(truth,pred,taus); base_losses=pinball_losses(truth,ref,taus)
        raw.update(prediction=pred,baseline_prediction=ref,historical_quantiles=historical,quantiles=taus,
                   historical_losses=pinball_losses(truth,historical,taus))
        info.update(task='distributional_forecast',primary_metric='mean_pinball',baseline_name='ridge_calibration_quantiles')
        info['extra_metrics'].update(empirical_quantile_pinball=float(raw['historical_losses'].mean()),
            coverage_90=float(np.mean((truth>=pred[:,0])&(truth<=pred[:,-1]))),quantile_crossings=int(np.sum(np.diff(pred,axis=1)<0)))
    elif mid=='M02':
        xp,beta=panel_features(r,train); xq,betaq=panel_features(r,test)
        future_market=r[train].mean(axis=1); test_market=r[test].mean(axis=1)
        train_resid=r[train]-beta*future_market[:,None]
        truth=r[test]-betaq*test_market[:,None]
        cfg=M02Config(rank_epochs=20,rank_hidden=12,max_pairs_per_date=10,random_state=d.seed)
        m=M02FactorResidualRanker(cfg)
        kwargs={} if ablate else {'betas':beta.reshape(-1,1),
            'realized_future_factors':np.repeat(future_market,5)[:,None]}
        m.fit(xp.reshape(-1,5),r[train].ravel(),date_ids=np.repeat(train,5),**kwargs)
        pred=m.predict(xq.reshape(-1,5)).reshape(-1,5)
        ref=ridge_predictions(xp.reshape(-1,5),train_resid.ravel(),xq.reshape(-1,5)).reshape(-1,5)
        lag=r[test-1]-betaq*r[test-1].mean(axis=1)[:,None]
        losses=-np.array([spearmanr(t,p).statistic for t,p in zip(truth,pred)])
        base_losses=-np.array([spearmanr(t,p).statistic for t,p in zip(truth,ref)])
        raw.update(truth=truth,prediction=pred,baseline_prediction=ref,lag_residual=lag,features_train=xp,features_test=xq,
                   train_labels=train_resid,beta_train=beta,beta_test=betaq,raw_return_truth=r[test])
        info.update(task='cross_sectional_ranking',primary_metric='negative_mean_spearman',baseline_name='residual_ridge_ranker')
        info['extra_metrics']['mean_spearman']=float(-losses.mean())
        info['extra_metrics']['lag_residual_spearman']=float(np.mean([spearmanr(t,p).statistic for t,p in zip(truth,lag)]))
    elif mid=='M03':
        predictions=[]; references=[]; means=[]; records=[]
        cfg=M03Config(n_factors=2,mlp_hidden=16,mlp_epochs=40,random_state=d.seed)
        for t in test:
            chars,_=panel_features(r,np.array([t])); panel=r[:t]
            m=M03ConditionalNonlinearFactor(cfg).fit(chars[0],r[t-1],returns_panel=panel,
                    loading_target='legacy_single_snapshot' if ablate else 'historical_ols')
            predictions.append(m.predict(chars[0]))
            factors,_=pca_factors(panel,2)
            ff=np.array([fit_ar1(factors[:,j])[0]+fit_ar1(factors[:,j])[1]*factors[-1,j] for j in range(2)])
            coef=np.linalg.lstsq(np.column_stack([np.ones(len(factors)),factors]),panel,rcond=None)[0]
            references.append(np.r_[1,ff]@coef); means.append(panel.mean(axis=0)); records.append(t-1)
        pred=np.asarray(predictions); ref=np.asarray(references); truth=r[test]
        losses=mse_losses(pred,truth); base_losses=mse_losses(ref,truth)
        raw.update(truth=truth,prediction=pred,baseline_prediction=ref,historical_mean=np.array(means),train_end_per_origin=np.array(records))
        info.update(task='rolling_factor_forecast',baseline_name='PCA_AR1_linear_loadings',
                    training_mode='expanding refit at every origin, including previously observed test returns; no future returns')
    elif mid in ('M04','M05','M06','M08','M09','M12'):
        if mid=='M04':
            cfg=M04Config(horizon_steps=1,maturity_delay=0,ridge_alpha=1,random_state=d.seed)
            cut=int(.7*len(train)); m=M04FinancialAPEN(cfg).fit(a[:cut],y[:cut])
            base_time=datetime(2020,1,1)
            if not ablate:
                for i in range(cut,len(train)):
                    # These are calibration observations, all matured before outer evaluation.
                    when=base_time+timedelta(days=int(train[i]))
                    m.log_prediction(a[i:i+1],y[i:i+1],when)
                    m.advance_time(when+timedelta(days=1))
            pred=m.predict(b,as_of=base_time+timedelta(days=int(test[0])))
            raw['same_backbone_prediction']=m._backbone_predict(b)
            info['extra_metrics']['matured_memory_episodes']=len(m._memory.episodes)
            info['extra_metrics']['backbone_train_rows']=cut
        elif mid=='M05':
            cfg=M05Config(max_experts=4,top_k=2,expert_types=('linear',) if ablate else ('linear','gaussian'),random_state=d.seed)
            m=QAPENModel(cfg).fit(a,y); pred=m.predict(b)
            info['extra_metrics']['active_experts']=len(m._specs)
            info['extra_metrics']['packed_payload_bytes_excluding_python_overhead']=m._memory.blob_nbytes()
            # No assumption that one batch tests expert birth/merge/prune effectiveness.
            info['training_mode']='one training batch; dynamic expert lifecycle effectiveness not established'
        elif mid=='M06':
            a=windows(r,train,8); b=windows(r,test,8)
            cfg=M06Config(seq_len=8,input_dim=5,hidden_dim=16,latent_dim=8,horizons=(1,5),epochs=12,
                          batch_size=32,random_state=d.seed,device='cpu')
            m=FIJEPAModel(cfg).fit(a,y,future_windows={h:future_paths(r,train,h) for h in cfg.horizons},
                                 predictive_coeff=0 if ablate else 1)
            pred=m.predict(b)
            info['extra_metrics']['internal_validation_mse']=m.validation_loss_
            info['training_mode']='internal validation diagnostic only; horizon-overlap training rows purged'
        elif mid=='M08':
            a=windows(r,train,8).transpose(0,2,1)[:,:,:,None]
            b=windows(r,test,8).transpose(0,2,1)[:,:,:,None]
            y=r[train]; truth=r[test]
            cfg=M08Config(n_nodes=5,seq_len=8,node_dim=1,hidden_dim=16,latent_dim=8,epochs=25,random_state=d.seed,device='cpu')
            m=GraphFIJEPAModel(cfg).fit(a,y,history_returns=r[:d.cutoff],no_edges=ablate)
            pred=m.predict(b)
            raw['adjacency']=m.frozen_adjacency()
            info['training_mode']='supervised node forecasts with training-history graph frozen before test; not a JEPA objective'
        elif mid=='M09':
            cfg=M09Config(groupdro_steps=10,groupdro_eta=0 if ablate else .3,ridge_alpha=.01,vol_threshold=1.4,random_state=d.seed)
            m=M09Model(cfg).fit(a,y); pred=m.predict(b)
            raw['group_weights']=m.group_weights_
            info['training_mode']='fixed random fusion features; weighted ridge; causal four-group labels'
        elif mid=='M12':
            a=r[train-1]; b=r[test-1]
            cfg=M12Config(n_assets=5,latent_dim=4,no_memory=ablate,random_state=d.seed)
            m=M12Model(cfg).fit(a,y,scenario_targets=future_paths(r,train,5).reshape(len(train),-1),
                    scenario_config={"noise_scale":1.0,"n_ode_steps":32,"ridge_alpha":.1})
            pred=m.predict(b)
            context=m._encode_market(b[:1])
            scenario=m.scenario_head.sample_paths(context,n_paths=64)*d.scale
            if np.min(scenario)<-1: raise ValueError('Generated scenario has an impossible simple return; planner not called')
            m.planner=M11Model(M11Config(allow_cash=True,horizon=5,solver="finite_candidates"))
            L=np.array([.2,.2,.2,.2,.23])
            allocation=m.plan(scenario,L,cash=1)
            heldout=future_paths(d.returns,test[::5],5)
            seq=m.planner._last_result.full_horizon_weights
            ledger=simulate_ledger(seq,heldout,1,np.zeros(5),L,.0005)
            raw.update(stack_scenarios=scenario,stack_weights=seq,stack_unpaid=ledger['unpaid'])
            info['extra_metrics']['composed_planner_cvar_unpaid']=empirical_cvar(ledger['unpaid'].sum(axis=1))
            info['extra_metrics']['first_action_risky_weight']=float(allocation.sum())
            info['training_mode']='composed, not jointly trained; random fixed latent dynamics; fresh full target paths, not duplicated scalar labels'
        ref=ridge_predictions(a,y,b)
        losses=mse_losses(pred,truth); base_losses=mse_losses(ref,truth)
        mean=np.broadcast_to(np.mean(y,axis=0),np.shape(truth))
        raw.update(truth=truth,prediction=pred,baseline_prediction=ref,historical_mean=mean,
                   historical_mean_losses=mse_losses(mean,truth))
        info['extra_metrics']['historical_mean_mse']=float(raw['historical_mean_losses'].mean())
        info['extra_metrics']['historical_mean_skill']=float(1-losses.mean()/raw['historical_mean_losses'].mean())
    elif mid=='M07':
        a=windows(r,train,20); b=windows(r,test,20)
        cov_train=covariance_targets(future_paths(r,train,5)); truth=covariance_targets(future_paths(r,test,5))
        cfg=M07Config(windows=(20,),n_assets=5,hidden_dim=24,latent_dim=12,epochs=35,random_state=d.seed,device='cpu')
        m=EigenJEPAModel(cfg).fit(np.zeros_like(a) if ablate else a,np.zeros(len(train)),future_covariances={20:cov_train})
        pred=m.predict_covariances(np.zeros_like(b) if ablate else b,20)
        ref=covariance_targets(b)
        # Training-global covariance is a serious baseline for stationary worlds.
        global_cov=np.cov(r[:d.cutoff].T)
        constant=np.broadcast_to(global_cov,truth.shape)
        losses=np.linalg.norm(pred-truth,axis=(1,2))
        base_losses=np.linalg.norm(ref-truth,axis=(1,2))
        raw.update(truth=truth,prediction=pred,baseline_prediction=ref,global_covariance=constant,
                   global_covariance_losses=np.linalg.norm(constant-truth,axis=(1,2)))
        info.update(task='forward_realized_covariance',primary_metric='frobenius_error_normalized_covariance',baseline_name='trailing_20_sample_covariance')
        info['extra_metrics']['global_covariance_error']=float(raw['global_covariance_losses'].mean())
        info['extra_metrics']['min_prediction_eigenvalue']=float(np.linalg.eigvalsh(pred).min())
        info['training_mode']='explicit five-step future realized sample covariance; conditional supervised SPD regression, not proof of JEPA pretraining'
    elif mid=='M10':
        target=future_paths(r,train,5).reshape(len(train),-1)
        truth=future_paths(r,test,5).reshape(len(test),-1)
        cfg=M10Config(n_assets=5,path_steps=5,n_ode_steps=32,context_dim=4,training_tau_samples=4,noise_scale=1,
                      ridge_alpha=.1,random_state=d.seed)
        m=M10Model(cfg).fit(np.zeros_like(a) if ablate else a,target)
        pred=np.stack([m.sample_paths(np.zeros_like(row[None]) if ablate else row[None],32).reshape(32,-1) for row in b])
        rb=Ridge(alpha=1).fit(a,target); residual=target-rb.predict(a); means=rb.predict(b)
        indices=rng.integers(0,len(train),(len(test),32))
        ref=means[:,None,:]+residual[indices]
        bootstrap=target[indices]
        losses=energy_losses(pred,truth); base_losses=energy_losses(ref,truth)
        raw.update(truth=truth,prediction=pred,baseline_prediction=ref,bootstrap_prediction=bootstrap,
                   bootstrap_losses=energy_losses(bootstrap,truth))
        info.update(task='joint_path_distribution',primary_metric='energy_score_normalized_path',baseline_name='conditional_ridge_residual_bootstrap')
        info['extra_metrics']['unconditional_bootstrap_energy']=float(raw['bootstrap_losses'].mean())
        info['training_mode']='affine time-dependent flow field; not a universal density estimator; no Gaussianity assumption for bootstrap'
    elif mid=='M11':
        pool=future_paths(d.returns,train,5)
        scenarios=pool[rng.integers(0,len(pool),64)]
        heldout=future_paths(d.returns,test[::5],5)
        L=np.array([.2,.2,.2,.2,.23])
        cfg=M11Config(horizon=5,allow_cash=not ablate,transaction_cost_bps=5,max_weight=.4,solver="finite_candidates")
        m=M11Model(cfg).fit(scenarios,np.zeros(len(scenarios)),liability_schedule=L,cash=1)
        m.predict(scenarios)
        w=m._last_result.full_horizon_weights
        ledger=simulate_ledger(w,heldout,1,np.zeros(5),L,.0005)
        cash_ledger=simulate_ledger(np.zeros_like(w),heldout,1,np.zeros(5),L,.0005)
        equal=simulate_ledger(np.full_like(w,.2),heldout,1,np.zeros(5),L,.0005)
        losses=ledger['unpaid'].sum(axis=1); base_losses=cash_ledger['unpaid'].sum(axis=1)
        raw.update(truth=heldout,prediction=w,baseline_prediction=np.zeros_like(w),unpaid=ledger['unpaid'],
                   liabilities=L,wealth=ledger['wealth_after'],fees=ledger['trade_fee']+ledger['liquidation_fee'],
                   identity_residual=ledger['identity_residual'],equal_weight_losses=equal['unpaid'].sum(axis=1),
                   scenario_returns=scenarios,test_origins=test[::5])
        info.update(task='liability_policy_stress',primary_metric='cvar95_unpaid_currency',baseline_name='cash_only',
                    eval_origins=len(heldout),reduction='cvar95',train_rows=len(scenarios))
        info['extra_metrics'].update(equal_weight_cvar=empirical_cvar(raw['equal_weight_losses']),
            mean_unpaid=float(losses.mean()),max_accounting_residual=float(np.max(np.abs(ledger['identity_residual']))),
            risky_first_weight=float(w[0].sum()),optimizer_success=bool(m._last_result.optimizer_success),
            policy_method=m._last_result.method,verified_candidates=m._last_result.candidate_count)
        info['training_mode']='one open-loop horizon plan fitted to training bootstrap; evaluated on reset-state nonoverlapping heldout windows, not live receding trading'
    else: raise ValueError(mid)
    raw['model_losses']=np.asarray(losses); raw['baseline_losses']=np.asarray(base_losses)
    for key,value in raw.items():
        if np.asarray(value).dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite actual output in {key}')
    info['config']=m.config_dict()
    reduce=empirical_cvar if info.get('reduction')=='cvar95' else np.mean
    info['primary_value']=float(reduce(losses)); info['baseline_value']=float(reduce(base_losses))
    info['paired_difference']=info['primary_value']-info['baseline_value']
    return m,raw,info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(DEFAULT_SEEDS))
    p.add_argument('--worlds',nargs='+',choices=WORLDS,default=list(WORLDS))
    p.add_argument('--models',nargs='+',choices=list(MODEL_TITLES),default=list(MODEL_TITLES))
    p.add_argument('--n-test',type=int,default=60)
    args=p.parse_args(); out=args.output
    if out.exists() and any(out.iterdir()): raise SystemExit('Refusing to overwrite an existing evidence directory')
    for sub in ('raw','datasets','checkpoints'): (out/sub).mkdir(parents=True,exist_ok=True)
    source_paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__) , ROOT/'protocol/PORTFOLIO_RESEARCH_V1.md']
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_paths}
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    env={'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),'upstream_sha':UPSTREAM,
         'packages':{x:importlib.metadata.version(x) for x in ('numpy','scipy','scikit-learn','pandas','torch','pytest')},
         'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}
    protocol={'label':'exploratory_synthetic_post_lockbox','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'worlds':args.worlds,'seeds':args.seeds,'models':args.models,'n_test':args.n_test,'variants':['full','one_predeclared_ablation'],
        'source_fingerprint':fingerprint,'source_hashes':hashes,'expected_cells':len(args.worlds)*len(args.seeds)*len(args.models)*2,
        'command':sys.argv,'no_market_data':True,'no_legacy_lockbox_access':True,'no_hyperparameter_selection':True}
    write_json(out/'protocol_snapshot.json',protocol); write_json(out/'environment.json',env)
    rows=[]; started=time.perf_counter()
    for world in args.worlds:
        for seed in args.seeds:
            data=make_dataset(world,seed,n_test=args.n_test)
            np.savez_compressed(out/'datasets'/f'{world}_{seed}.npz',returns=data.returns,train_origins=data.train_origins,
                                test_origins=data.test_origins,scale=np.array(data.scale))
            for mid in args.models:
                for ablate in (False,True):
                    variant=ABLATIONS[mid] if ablate else 'full'; key=f'{mid}__{world}__{seed}__{variant}'
                    row={'cell_id':key,'model':mid,'world':world,'seed':seed,'variant':variant,
                         'source_fingerprint':fingerprint,'dataset_sha256':data.sha256,'status':'started'}
                    t=time.perf_counter()
                    try:
                        m,raw,info=evaluate(mid,ablate,data)
                        path=out/'raw'/f'{key}.npz'; np.savez_compressed(path,**raw)
                        # Save a representative checkpoint for each model/variant (not all seeds).
                        if seed==args.seeds[0] and world==args.worlds[0]:
                            m.save(out/'checkpoints'/f'{mid}__{variant}.pkl')
                        row.update(info,status='completed',artifact=str(path.relative_to(out)),
                                   artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    except Exception as exc:
                        row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                        (out/'raw'/f'{key}.error.txt').write_text(traceback.format_exc())
                    row['elapsed_seconds']=time.perf_counter()-t
                    rows.append(row)
                    with (out/'cells.jsonl').open('a') as f: f.write(json.dumps(row,default=json_default,allow_nan=False)+'\n')
                    print(f'{len(rows)}/{protocol["expected_cells"]} {key} {row["status"]} {row.get("primary_value",row.get("error"))}',flush=True)
    write_json(out/'results.json',rows)
    summary={'completed':sum(x['status']=='completed' for x in rows),'failed':sum(x['status']=='failed' for x in rows),
             'expected_cells':protocol['expected_cells'],'elapsed_seconds':time.perf_counter()-started,
             'source_fingerprint':fingerprint,'finished_at_utc':datetime.now(timezone.utc).isoformat(),
             'status':'completed' if all(x['status']=='completed' for x in rows) else 'completed_with_failures'}
    write_json(out/'execution_summary.json',summary); print(json.dumps(summary),flush=True)
    return 0 if summary['failed']==0 else 1

if __name__=='__main__': raise SystemExit(main())
