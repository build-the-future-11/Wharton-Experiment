"""One admitted run; evidence wrappers never change model inputs or predictions."""
from pathlib import Path
import datetime, hashlib, importlib.metadata, json, os, platform, resource, sys, time, traceback

HERE=Path(__file__).resolve().parent
PROTOCOL=HERE.parent/'prospective_protocol/WHARTON_PROSPECTIVE_SYNTHETIC_COMPONENT_PROTOCOL.json'
EXPECTED='543530312a36b3c001f272eb8db9e43f895455ae878dd827b9adefc6310197f5'
OUT=HERE/'attempt-001'
START=time.monotonic(); CPU=time.process_time()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def stamp():return {'wall_seconds':time.monotonic()-START,'cpu_seconds':time.process_time()-CPU,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
def inclusive(bounds):return list(range(bounds[0],bounds[1]+1))

OUT.mkdir(exist_ok=False)
receipt={'status':'checking_gates','protocol_sha256':EXPECTED,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'one synthetic component-development path; no market evidence or model adoption',
 'execution_identity':{'worker_sha256':sha(__file__),'supervisor_sha256':sha(HERE/'supervisor.py')},
 'instrumentation':'Wrappers record exact original FinalModel.fit, XGBRegressor.fit, Ridge.fit calls without modifying arguments or return values. No refits for evidence.'}
write('worker_receipt.json',receipt)
try:
    assert sha(PROTOCOL)==EXPECTED,'protocol bytes changed'
    p=json.loads(PROTOCOL.read_text());root=Path(p['source_and_environment']['source_root'])
    assert Path.cwd().resolve()==root.resolve(),'incorrect candidate working directory'
    guard=json.loads((HERE/'GUARD_TEST_RESULTS.json').read_text())
    assert guard['status']=='PASS' and guard['supervisor_sha256']==sha(HERE/'supervisor.py'),'guard evidence does not bind supervisor'
    assert not any(t.get('signal_errors') for t in guard['tests'])
    receipt['guard_test_sha256']=sha(HERE/'GUARD_TEST_RESULTS.json')
    expected_source=p['source_and_environment']['source_revision']
    actual_files={name:sha(root/name) for name in expected_source['files']}
    assert actual_files==expected_source['files'],'source hash mismatch'
    actual_source=hashlib.sha256(json.dumps(actual_files,sort_keys=True).encode()).hexdigest()
    assert actual_source==expected_source['sha256']
    packages={name:importlib.metadata.version(name) for name in p['source_and_environment']['packages']}
    assert packages==p['source_and_environment']['packages'],'dependency mismatch'
    assert sys.version_info[:2]==(3,14) and platform.system()=='Darwin' and platform.machine()=='arm64'
    for key,value in p['resource_and_cost_budget']['environment_before_any_import'].items():assert os.environ.get(key)==value,(key,'thread/import environment mismatch')
    config_data=p['fixed_hyperparameters']['model_config']
    receipt.update(protocol_id=p['protocol_id'],source_revision=expected_source,packages=packages,python=sys.version,
      config=config_data,config_sha256=hashlib.sha256(json.dumps(config_data,sort_keys=True).encode()).hexdigest())
    generator=p['data']['generator_python']
    assert hashlib.sha256(generator.encode()).hexdigest()==p['data']['identity_before_generation']['generator_sha256']
    (OUT/'generator.py').write_text(generator)
    from wharton_final.pipeline import source_revision, evaluate
    from wharton_final.model import Config, FinalModel
    from wharton_final.data import load_panel, samples, matured_indices
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor
    import numpy as np
    assert source_revision()==expected_source
    receipt.update(status='gates_passed',pre_generation=stamp());write('worker_receipt.json',receipt)
    namespace={};exec(compile(generator,str(OUT/'generator.py'),'exec'),namespace)
    manifest=namespace['write_generated_panel'](OUT/'data') # ONLY generator invocation
    panel=load_panel(manifest);config=Config(**config_data)
    assert panel.returns.shape==(660,2) and panel.assets==('SYNTH_A','SYNTH_B')
    assert np.array_equal(np.diff(panel.decisions),np.full(659,86400))
    assert np.array_equal(panel.observed-panel.decisions,np.full(660,3600))
    s=samples(panel,128,20,0);table={e['origin']:e for e in p['chronology']['exact_index_table']}
    for row,e in table.items():
        i=int(np.flatnonzero(s.rows==row)[0]);asof=s.decisions[i];idx=matured_indices(s,asof)
        split=max(8,int(.8*len(idx)));fit=idx[:split];cal=idx[split:];purged=fit[s.maturity[fit]>=s.decisions[cal[0]]];fit=fit[s.maturity[fit]<s.decisions[cal[0]]]
        for actual,key in ((s.rows[idx],'eligible'),(s.rows[fit],'fit'),(s.rows[cal],'calibration'),(s.rows[purged],'purged')):assert actual.tolist()==inclusive(e[key]),(row,key)
        assert np.all(s.maturity[idx]<asof) and np.all(s.maturity[fit]<s.decisions[cal[0]])
        assert np.array_equal(s.context[i],panel.returns[row-128:row]) and panel.observed[row-1]<asof
        assert s.maturity[i]==panel.observed[row+19]
    receipt.update(data_revision=panel.revision,status='data_admitted',data_admitted=stamp());write('worker_receipt.json',receipt)
    print('All source, dependency, data and chronology gates passed; beginning one fixed evaluation.',flush=True)
    origins=[];boosters=[];ridges=[]
    original_fit=FinalModel.fit;original_tree_fit=XGBRegressor.fit;original_ridge_fit=Ridge.fit
    def fitted_model(self,samples_arg,asof):
        row=int(panel.decisions.searchsorted(asof));assert row==[600,620,640][len(origins)]
        now=stamp()
        if origins:origins[-1]['origin_interval_end']=now
        item={'origin':row,'start':now};origins.append(item);write('origin_timings.json',origins)
        print('Origin',row,'starting',now,flush=True)
        result=original_fit(self,samples_arg,asof)
        assert self.train_rows==inclusive(table[row]['fit']) and self.calibration_rows==inclusive(table[row]['calibration'])
        assert len(self.losses)==20 and np.isfinite(self.losses).all()
        assert np.all(self.memory_maturity<asof)
        item['model_fit_end']=stamp();write('origin_timings.json',origins)
        return result
    def fitted_tree(self,*args,**kwargs):
        result=original_tree_fit(self,*args,**kwargs)
        booster=self.get_booster();resolved=json.loads(booster.save_config());params=self.get_params()
        index=len(boosters);assert index<12
        record={'fit_number':index+1,'origin':origins[-1]['origin'],'method':'candidate' if index%4<2 else 'xgboost_matched_fit',
            'channel':index%2,'num_boosted_rounds':booster.num_boosted_rounds(),'resolved_config':resolved}
        boosters.append(record);write('booster_configs.json',boosters)
        assert booster.num_boosted_rounds()==64
        for key,val in {'n_estimators':64,'max_depth':3,'learning_rate':.05,'subsample':1,'colsample_bytree':1,'random_state':11,'n_jobs':1,'objective':'reg:squarederror'}.items():assert params[key]==val,(key,params[key])
        learner=resolved['learner'];generic=learner['generic_param'];tree=learner['gradient_booster']['tree_train_param']
        assert generic['device']=='cpu' and int(generic['nthread'])==1 and int(generic['seed'])==11
        assert learner['objective']['name']=='reg:squarederror'
        for key,val in {'max_depth':3,'max_bin':256,'max_leaves':0,'min_child_weight':1,'gamma':0,'reg_alpha':0,'reg_lambda':1,'max_delta_step':0,'subsample':1,'colsample_bytree':1,'colsample_bylevel':1,'colsample_bynode':1}.items():assert float(tree[key])==val,(key,tree[key])
        assert np.isclose(float(tree['learning_rate']),.05,rtol=1e-6,atol=0)
        assert tree['grow_policy']=='depthwise' and tree['sampling_method']=='uniform'
        assert learner['gradient_booster']['gbtree_train_param']['updater_seq']=='grow_quantile_histmaker'
        return result
    def fitted_ridge(self,*args,**kwargs):
        result=original_ridge_fit(self,*args,**kwargs);ridges.append({'origin':origins[-1]['origin'],'n_samples':len(args[0]),'n_features':args[0].shape[1]});write('ridge_fits.json',ridges);return result
    FinalModel.fit=fitted_model;XGBRegressor.fit=fitted_tree;Ridge.fit=fitted_ridge
    metadata=evaluate(manifest,config,first_test_row=600,test_count=3,stride=20,output=OUT/'evaluation')
    origins[-1]['origin_interval_end']=stamp()
    for item in origins:
        item['origin_interval_wall_seconds']=item['origin_interval_end']['wall_seconds']-item['start']['wall_seconds']
        item['origin_interval_cpu_seconds']=item['origin_interval_end']['cpu_seconds']-item['start']['cpu_seconds']
    write('origin_timings.json',origins)
    records=json.loads((OUT/'evaluation/predictions.json').read_text());names=[m['name'] for m in p['methods']]
    assert [r['row'] for r in records]==[600,620,640] and len(boosters)==12 and len(ridges)==6
    assert metadata['source_revision']==expected_source and metadata['data_revision']==panel.revision
    assert metadata['evidence_class']=='correctness_only' and metadata['status']=='completed'
    assert sorted(metadata['metrics'])==sorted(names)
    for r in records:
        e=table[r['row']]
        for key,target in [('fit_rows','fit'),('matched_baseline_rows','fit'),('calibration_rows','calibration'),('baseline_rows','eligible')]:assert r[key]==inclusive(e[target])
        assert len(r['losses'])==20 and np.isfinite(r['losses']).all()
        for name in ['truth',*names]:assert np.asarray(r[name]).shape==(2,) and np.isfinite(r[name]).all()
    truth=np.asarray([r['truth'] for r in records]);errors={name:truth-np.asarray([r[name] for r in records]) for name in names}
    metrics=metadata['metrics'];contrasts={name:metrics['candidate']['mse']-metrics[name]['mse'] for name in names if name!='candidate'}
    derived={'scope':receipt['scope'],'independent_units':1,'forecast_values_per_method':6,'methods':7,'metrics':metrics,
      'primary_delta_memory':contrasts['candidate_no_memory'],'candidate_minus_comparator_mse':contrasts,
      'max_abs_memory_prediction_change':float(np.max(np.abs(errors['candidate']-errors['candidate_no_memory']))),
      'errors':{name:{'squared':(error**2).tolist(),'absolute':np.abs(error).tolist()} for name,error in errors.items()},
      'interpretation':'Descriptive results on exactly one generated path. No significance, market-performance, adoption or competition-finalization claim.'}
    assert source_revision()==expected_source
    write('descriptive_results.json',derived)
    receipt.update(status='completed',completion=stamp(),origin_count=3,method_count=7,forecast_count=42,neural_epochs=60,scalar_tree_fits=12,trees_per_fit=64,multivariate_ridge_fits=6,
       timing_definition='Per-origin intervals start immediately before original FinalModel.fit and end at the next origin fit or evaluation return; include baselines and checkpoint/prediction serialization. Initial imports/data and final validation are in whole-worker time.')
except BaseException as error:
    receipt.update(status='failed',error=repr(error),failure=stamp(),traceback=traceback.format_exc())
    traceback.print_exc();raise
finally:
    receipt['artifact_sha256']={str(path.relative_to(OUT)):sha(path) for path in sorted(OUT.rglob('*')) if path.is_file() and path.name!='worker_receipt.json'}
    receipt['retained_bytes_excluding_this_receipt']=sum(path.stat().st_size for path in OUT.rglob('*') if path.is_file() and path.name!='worker_receipt.json')
    write('worker_receipt.json',receipt)
    print('Worker disposition:',receipt['status'],stamp(),flush=True)
