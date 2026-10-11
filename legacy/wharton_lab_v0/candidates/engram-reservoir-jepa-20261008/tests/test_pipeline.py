import json,tempfile,unittest
from pathlib import Path
import numpy as np
from wharton_final.data import Panel,samples,matured_indices,load_panel,digest
from wharton_final.model import Config,FinalModel
from wharton_final.pipeline import evaluate
from wharton_final.planning import scenarios,rollout,cvar,plan
from wharton_final.accounting import State,Policy,Obligation

def panel(n=90):
    # Test fixture only, never observed market performance.
    rng=np.random.default_rng(42);r=rng.normal(0,.01,(n,2));d=np.arange(n)*86400.+1700000000
    return Panel(r,d,d+3600,('A','B'),{'role':'synthetic_correctness_fixture'})
def fixture_files(root,n=90):
    from datetime import datetime,timezone
    p=panel(n);csv=root/'fixture.csv';lines=['decision_time,observed_at,A,B']
    for d,o,r in zip(p.decisions,p.observed,p.returns):lines.append(','.join([datetime.fromtimestamp(d,timezone.utc).isoformat(),datetime.fromtimestamp(o,timezone.utc).isoformat(),*map(str,r)]))
    csv.write_text('\n'.join(lines)+'\n')
    m={'csv':'fixture.csv','sha256':digest(csv),'assets':['A','B'],'source':'synthetic unit fixture, not real performance','license':'generated test fixture','lawful_use_attested':True,'asset_order_verified':True,'role':'development','return_kind':'total','protected_rows':[],'data_kind':'synthetic_fixture'}
    path=root/'manifest.json';path.write_text(json.dumps(m));return path,p

class ChronologyTests(unittest.TestCase):
    def test_features_do_not_change_with_future(self):
        p=panel();s=samples(p,6,3,0);p.returns[40:]=.2;t=samples(p,6,3,0);i=np.where(s.rows==40)[0][0]
        np.testing.assert_array_equal(s.context[i],t.context[i]);self.assertFalse(np.array_equal(s.labels[i],t.labels[i]))
    def test_true_horizon_and_label_maturation(self):
        p=panel();s=samples(p,6,3,100);i=5;t=s.rows[i]
        np.testing.assert_allclose(s.labels[i],np.prod(1+p.returns[t:t+3],axis=0)-1)
        self.assertEqual(s.maturity[i],p.observed[t+2]+100);self.assertNotIn(i,matured_indices(s,s.maturity[i]));self.assertIn(i,matured_indices(s,s.maturity[i]+1))
    def test_delayed_observation_rejected(self):
        p=panel();p.observed=p.decisions+90000
        with self.assertRaises(ValueError):samples(p,6,3,0)
    def test_data_admission_hash_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m,p=fixture_files(root);loaded=load_panel(m);np.testing.assert_array_equal(p.returns,loaded.returns)
            obj=json.loads(m.read_text());obj['assets']=['B','A'];m.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):load_panel(m)
            obj['assets']=['A','B'];obj['sha256']='0'*64;m.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):load_panel(m)
    def test_spent_lockbox_rejected(self):
        from wharton_final.data import LEGACY_SHA
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m,_=fixture_files(root);obj=json.loads(m.read_text());obj['upstream_sha256']=LEGACY_SHA;obj['upstream_rows']=list(range(800,890));m.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):load_panel(m)
        with self.assertRaises(ValueError):Config(seed=997)

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=panel();cls.s=samples(cls.p,6,3,0);cls.c=Config(context=6,horizon=3,latent=4,reservoir=5,epochs=2,head='ridge');cls.asof=cls.p.decisions[70];cls.model=FinalModel(cls.c).fit(cls.s,cls.asof)
    def test_predict_save_reload_and_batch_invariance(self):
        x=self.s.context[-2:];pred=self.model.predict(x,self.asof)
        np.testing.assert_allclose(pred,np.vstack([self.model.predict(v[None],self.asof) for v in x]),rtol=1e-6,atol=1e-8)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.pkl';self.model.save(p);restored=FinalModel.load(p);np.testing.assert_array_equal(pred,restored.predict(x,self.asof))
    def test_future_label_poison_does_not_change_fit(self):
        s=samples(panel(),6,3,0);s.labels[s.maturity>=self.asof]=999;s.future[s.maturity>=self.asof]=999;m=FinalModel(self.c).fit(s,self.asof)
        np.testing.assert_array_equal(self.model.predict(self.s.context[-1:],self.asof),m.predict(self.s.context[-1:],self.asof))
    def test_memory_matured_and_encoder_learns(self):
        self.assertTrue(np.all(self.model.memory_maturity<self.asof));self.assertNotEqual(self.model.losses[0],self.model.losses[-1]);self.assertLess(np.linalg.norm(self.model.wres,2),1)
    def test_full_evaluation_function(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_=fixture_files(root);receipt=evaluate(manifest,self.c,first_test_row=65,test_count=2,stride=3,output=root/'eval')
            self.assertEqual(receipt['status'],'completed');self.assertEqual(set(receipt['metrics']),{'candidate','ridge','historical_mean','persistence','ridge_matched_fit','candidate_no_memory'});self.assertTrue((root/'eval'/'model-65.pkl').exists());self.assertTrue(receipt['source_revision']['sha256'])
    def test_no_implicit_xgboost_substitution(self):self.assertEqual(Config().head,'xgboost')
    def test_inference_rejects_old_clock(self):
        with self.assertRaises(ValueError):self.model.predict(self.s.context[-1:],self.asof-1)

class PlannerTests(unittest.TestCase):
    def test_forecast_reaches_scenarios(self):
        p=panel();a=scenarios(p.returns,[0,0],horizon=3,count=4,seed=11);b=scenarios(p.returns,[.1,.2],horizon=3,count=4,seed=11)
        self.assertTrue(np.all(b>a));np.testing.assert_array_equal(a,scenarios(p.returns,[0,0],horizon=3,count=4,seed=11))
    def test_shortfall_counted_once(self):
        state=State(0,(0,),(1,),5);policy=Policy('cash_only','carry','after_market','mark',0);out=rollout(state,np.zeros((2,2,1)),[0],[[Obligation('a',1,10)],[]],policy,terminal_cash_target=0,cash_rates=[0,0],future_liability_policy='reserve_nominal_cash')
        np.testing.assert_array_equal(out[:,2],[5,5]);np.testing.assert_array_equal(out[:,-1],[5,5])
    def test_writeoff_does_not_hide_failure(self):
        state=State(0,(0,),(1,),5);policy=Policy('cash_only','write_off','after_market','mark',0);out=rollout(state,np.zeros((2,1,1)),[0],[[Obligation('a',1,10)]],policy,terminal_cash_target=0,cash_rates=[0],future_liability_policy='reserve_nominal_cash')
        np.testing.assert_array_equal(out[:,-1],[5,5])
    def test_terminal_cash_policy_used_by_planner(self):
        state=State(0,(0,),(1,),10);policy=Policy('cash_only','carry','after_market','mark',0);result=plan(state,np.zeros((2,1,1)),[[]],policy,max_weight=1,alpha=.5,terminal_cash_target=10,cash_rates=[0],future_liability_policy='reserve_nominal_cash',maxiter=5)
        np.testing.assert_allclose(result.weights,[0]);self.assertEqual(result.objective,0)
    def test_fractional_cvar(self):
        self.assertAlmostEqual(cvar([0,1,2,3],.625),(3+.5*2)/1.5);self.assertEqual(cvar([1,1,1],.9),1)

class FullHeadAndCliTests(unittest.TestCase):
    def test_xgboost_pipeline_and_forecast_cli(self):
        import subprocess,sys
        from datetime import datetime,timezone
        from wharton_final.pipeline import infer
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,p=fixture_files(root)
            cfg=Config(context=6,horizon=3,latent=4,reservoir=5,epochs=2,head='xgboost')
            config=root/'config.json'
            from dataclasses import asdict
            config.write_text(json.dumps(asdict(cfg)))
            cmd=[sys.executable,'-m','wharton_final','evaluate','--manifest',str(manifest),'--config',str(config),'--first-test-row','65','--test-count','1','--stride','3','--output',str(root/'run')]
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
            self.assertEqual(result.returncode,0,result.stderr)
            cp=root/'run'/'model-65.pkl';asof=datetime.fromtimestamp(p.decisions[-1]+86400,timezone.utc).isoformat()
            cmd=[sys.executable,'-m','wharton_final','forecast','--checkpoint',str(cp),'--checkpoint-sha256',digest(cp),'--manifest',str(manifest),'--as-of',asof,'--max-context-age-seconds','86400','--output',str(root/'forecast.json')]
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
            self.assertEqual(result.returncode,0,result.stderr)
            pred=json.loads((root/'forecast.json').read_text())
            self.assertEqual(set(pred['forecast']),{'A','B'})
            contract={'assets':['A','B'],'state':{'at':0,'units':[0,0],'prices':[1,1],'cash':100},'policy':{'payment':'cash_only','unpaid':'carry','timing':'after_market','terminal':'mark','fee_rate':0},'scenario_count':2,'seed':11,'obligations':[[{'id':'a','due_at':1,'amount':1}],[],[]],'max_weight':.8,'alpha':.5,'terminal_cash_target':90,'cash_rates':[0,0,0],'future_liability_policy':'reserve_nominal_cash'}
            pc=root/'plan.json';pc.write_text(json.dumps(contract))
            with_plan=cmd[:-2]+['--output',str(root/'planned.json'),'--plan-contract',str(pc)]
            result=subprocess.run(with_plan,capture_output=True,text=True,timeout=45)
            self.assertEqual(result.returncode,0,result.stderr)
            planned=json.loads((root/'planned.json').read_text())
            self.assertEqual(pred['forecast'],planned['forecast']);self.assertEqual(planned['plan']['scenario_paid'],[1,1])
            # Stale contexts must fail against an explicit age allowance.
            with self.assertRaises(ValueError):infer(cp,digest(cp),manifest,p.decisions[-1]+10*86400,max_context_age_seconds=86400)
            # The same-shape panel with renamed assets must fail before inference.
            obj=json.loads(manifest.read_text());obj['assets']=['C','D'];csv=root/'fixture.csv';csv.write_text(csv.read_text().replace('observed_at,A,B','observed_at,C,D'));obj['sha256']=digest(csv);manifest.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):infer(cp,digest(cp),manifest,p.decisions[-1]+86400,max_context_age_seconds=86400)
    def test_ledger_cli(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'ledger.json'
            r=subprocess.run([sys.executable,'-m','wharton_final','ledger','--contract','configs/ledger-correctness.json','--output',str(output)],capture_output=True,text=True,timeout=30)
            self.assertEqual(r.returncode,0,r.stderr);receipt=json.loads(output.read_text());self.assertEqual([x['state']['cash'] for x in receipt['receipts']],['0.9','0.8'])
    def test_multiasset_optimizer_roundoff(self):
        state=State(0,(0,0),(1,1),10);policy=Policy('sell_pro_rata','carry','after_market','liquidate','.001')
        paths=np.array([[[.01,.02]],[[.02,-.01]]])
        result=plan(state,paths,[[]],policy,max_weight=.8,alpha=.5,terminal_cash_target=11,cash_rates=[0],future_liability_policy='reserve_nominal_cash',maxiter=10)
        self.assertLessEqual(result.weights.sum(),1+1e-10);self.assertTrue(np.isfinite(result.objective))
