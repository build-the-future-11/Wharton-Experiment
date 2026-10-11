"""Development-only chronological evaluation, baselines, artifacts, and inference."""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,platform,sys,time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .data import load_panel,samples,matured_indices,digest
from .model import Config,FinalModel

def environment_versions():
    import importlib.metadata
    versions={}
    for name in ('numpy','scipy','scikit-learn','torch','xgboost'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return versions

def source_revision():
    root=Path(__file__).parent.parent
    files=sorted([*root.glob('wharton_final/*.py'),root/'SPEC.md',root/'pyproject.toml',root/'evidence'/'SOURCE_RECOVERY.json',root/'configs'/'recovered-laura-gao-source.json'])
    hashes={str(p.relative_to(root)):digest(p) for p in files if p.is_file()}
    return {'files':hashes,'sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()}

def evaluate(manifest, config, *, first_test_row, test_count, stride, output):
    """Fit anew at each origin. No selection or confirmatory interpretation."""
    if test_count<1 or stride<config.horizon:raise ValueError('test labels must be disjoint; stride >= horizon')
    panel=load_panel(manifest);s=samples(panel,config.context,config.horizon,config.reporting_delay_seconds)
    origins=[first_test_row+i*stride for i in range(test_count)]
    row_lookup={int(r):i for i,r in enumerate(s.rows)}
    if any(row not in row_lookup for row in origins):raise ValueError('requested origin lacks complete context and label')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();revision=source_revision()
    metadata={'status':'running','role':'exploratory_development','evidence_class':'correctness_only' if panel.revision['contract']['data_kind']=='synthetic_fixture' else 'exploratory_market','source_revision':revision,
        'data_revision':panel.revision,'config':asdict(config),'origins':origins,
        'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'packages':environment_versions()}}
    (out/'receipt.json').write_text(json.dumps(metadata,indent=2)+'\n')
    records=[]
    try:
        for row in origins:
            i=row_lookup[row];asof=s.decisions[i];idx=matured_indices(s,asof)
            model=FinalModel(config).fit(s,asof)
            model.asset_names=panel.assets
            model.data_revision=panel.revision
            model.source_revision=revision
            pred=model.predict(s.context[i:i+1],asof)[0]
            # Strong all-matured baselines; matched-fit contrasts are recorded separately.
            x=s.context[idx].reshape(len(idx),-1);scaler=StandardScaler().fit(x)
            ridge=Ridge(alpha=1).fit(scaler.transform(x),s.labels[idx])
            base=ridge.predict(scaler.transform(s.context[i:i+1].reshape(1,-1)))[0]
            mean=s.labels[idx].mean(0)
            persistence=np.expm1(config.horizon*np.log1p(s.context[i,-1]))
            fit=model.train_indices
            matched_x=s.context[fit].reshape(len(fit),-1)
            matched_scaler=StandardScaler().fit(matched_x)
            matched_ridge=Ridge(alpha=1).fit(matched_scaler.transform(matched_x),s.labels[fit])
            matched_pred=matched_ridge.predict(matched_scaler.transform(s.context[i:i+1].reshape(1,-1)))[0]
            contrasts={'ridge_matched_fit':matched_pred.tolist(),
                'candidate_no_memory':model.predict(s.context[i:i+1],asof,use_memory=False)[0].tolist()}
            if config.head=='xgboost':
                from xgboost import XGBRegressor
                raw_trees=[]
                for asset in range(len(panel.assets)):
                    tree=XGBRegressor(n_estimators=64,max_depth=3,learning_rate=.05,subsample=1,colsample_bytree=1,random_state=config.seed,n_jobs=1,objective='reg:squarederror')
                    tree.fit(matched_x,s.labels[fit,asset])
                    raw_trees.append(float(tree.predict(s.context[i:i+1].reshape(1,-1))[0]))
                contrasts['xgboost_matched_fit']=raw_trees
            records.append({'row':row,'as_of':float(asof),'label_matures':float(s.maturity[i]),
                'fit_rows':model.train_rows,'matched_baseline_rows':model.train_rows,'calibration_rows':model.calibration_rows,'baseline_rows':s.rows[idx].tolist(),
                'truth':s.labels[i].tolist(),'candidate':pred.tolist(),'ridge':base.tolist(),
                'historical_mean':mean.tolist(),'persistence':persistence.tolist(),'losses':model.losses,**contrasts})
            model.save(out/f'model-{row}.pkl')
            (out/'predictions.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        truth=np.array([r['truth'] for r in records]);metrics={}
        mean_mse=float(np.mean((truth-np.array([r['historical_mean'] for r in records]))**2))
        names=['candidate','ridge','historical_mean','persistence','ridge_matched_fit','candidate_no_memory']
        if config.head=='xgboost':names.append('xgboost_matched_fit')
        for name in names:
            error=truth-np.array([r[name] for r in records]);mse=float(np.mean(error**2))
            metrics[name]={'mse':mse,'mae':float(np.abs(error).mean()),'oos_r2_vs_historical_mean':None if mean_mse==0 else 1-mse/mean_mse}
        metadata.update(status='completed',metrics=metrics,elapsed_seconds=time.monotonic()-started,
            checkpoint_sha256={p.name:digest(p) for p in sorted(out.glob('*.pkl'))},predictions_sha256=digest(out/'predictions.json'))
    except Exception as error:
        metadata.update(status='failed',error=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-started)
        raise
    finally:(out/'receipt.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    return metadata


def infer(checkpoint,expected_sha256,manifest,as_of,*,max_context_age_seconds):
    if not np.isfinite(as_of) or not np.isfinite(max_context_age_seconds) or max_context_age_seconds<=0:
        raise ValueError('finite as-of and explicit positive maximum context age required')
    if digest(checkpoint)!=expected_sha256:raise ValueError('checkpoint checksum mismatch')
    model=FinalModel.load(checkpoint)
    if getattr(model,'source_revision',None)!=source_revision():
        raise ValueError('checkpoint source revision differs from running implementation')
    panel=load_panel(manifest)
    idx=np.flatnonzero(panel.observed<as_of)
    if len(idx)<model.config.context:raise ValueError('insufficient observed inference history')
    if as_of-panel.observed[idx[-1]]>max_context_age_seconds:
        raise ValueError('latest available context exceeds declared maximum age')
    x=panel.returns[idx[-model.config.context:]][None,:,:]
    if not hasattr(model,'asset_names') or tuple(panel.assets)!=tuple(model.asset_names):
        raise ValueError('checkpoint asset identities differ from inference data')
    return model.predict(x,as_of)[0],model,panel
