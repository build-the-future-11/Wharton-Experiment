import copy,unittest
import numpy as np
from wharton_final.accounting import State,Policy,Obligation,transition
from wharton_final.planning import rollout
from wharton_final.model import Config,FinalModel
from wharton_final.data import samples
from test_pipeline import panel

class ReviewRegressions(unittest.TestCase):
    def test_reserved_future_liability_is_not_missed_payment(self):
        state=State(0,(0,),(1,),100,(Obligation('later',5,80),))
        p=Policy('cash_only','carry','after_market','mark',0)
        result=rollout(state,np.zeros((2,1,1)),[0],[[]],p,terminal_cash_target=0,cash_rates=[0],future_liability_policy='reserve_nominal_cash')
        self.assertEqual(result[0,2],0)
        self.assertEqual(result[0,-1],0)
    def test_cash_yield_must_be_explicit(self):
        with self.assertRaises(TypeError):
            transition(State(0,(0,),(1,),100),at=1,returns=[0],policy=Policy('cash_only','carry','after_market','mark',0))
    def test_nonfinite_inference_clock_rejected(self):
        p=panel();s=samples(p,6,3,0);m=FinalModel(Config(context=6,horizon=3,latent=4,reservoir=5,epochs=1,head='ridge')).fit(s,p.decisions[70])
        with self.assertRaises(ValueError):m.predict(s.context[-1:],float('nan'))
    def test_failed_refit_cannot_leave_partial_model_predictable(self):
        p=panel();s=samples(p,6,3,0);m=FinalModel(Config(context=6,horizon=3,latent=4,reservoir=5,epochs=1,head='ridge')).fit(s,p.decisions[70])
        s.labels[0]=np.nan
        with self.assertRaises((ValueError,FloatingPointError)):m.fit(s,p.decisions[70])
        self.assertFalse(m.fitted)

class CausalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=panel();cls.s=samples(cls.p,6,3,0);cls.asof=cls.p.decisions[70]
        cls.c=Config(context=6,horizon=3,latent=4,reservoir=5,epochs=1,head='ridge')
        cls.m=FinalModel(cls.c).fit(cls.s,cls.asof)
    def test_input_and_latent_normalizers_fit_training_only(self):
        x=self.s.context[self.m.train_indices]
        np.testing.assert_array_equal(self.m.mean,x.reshape(-1,2).mean(0))
        np.testing.assert_array_equal(self.m.scale,x.reshape(-1,2).std(0).clip(1e-8))
        z=self.m._latent(x)
        np.testing.assert_array_equal(self.m.zmean,z.mean(0));np.testing.assert_array_equal(self.m.zscale,z.std(0).clip(1e-8))
        self.assertLess(self.s.maturity[self.m.train_indices].max(),self.s.decisions[self.m.calibration_indices[0]])
    def test_calibration_future_targets_cannot_train_encoder(self):
        s=copy.deepcopy(self.s);s.future[self.m.calibration_indices]=1000
        m=FinalModel(self.c).fit(s,self.asof)
        for k,v in self.m.encoder.state_dict().items():np.testing.assert_array_equal(v.numpy(),m.encoder.state_dict()[k].numpy())
        np.testing.assert_array_equal(self.m.mean,m.mean)
    def test_poisoned_unavailable_context_cannot_affect_fit(self):
        s=copy.deepcopy(self.s);s.context[s.decisions>=self.asof]=1000;s.labels[s.maturity>=self.asof]=1000
        m=FinalModel(self.c).fit(s,self.asof)
        for k,v in self.m.encoder.state_dict().items():np.testing.assert_array_equal(v.numpy(),m.encoder.state_dict()[k].numpy())
        np.testing.assert_array_equal(self.m.keys,m.keys);np.testing.assert_array_equal(self.m.residuals,m.residuals)
    def test_future_memory_values_never_retrieved(self):
        m=copy.deepcopy(self.m);x=self.s.context[-1:];expected=m.predict(x,self.asof)
        m.keys=np.vstack([m.keys,m.keys[-1]]);m.residuals=np.vstack([m.residuals,[1e20,1e20]]);m.memory_maturity=np.r_[m.memory_maturity,self.asof]
        np.testing.assert_array_equal(expected,m.predict(x,self.asof))
    def test_memory_off_is_exact_head_prediction(self):
        x=self.s.context[-2:];z=(self.m._latent(x)-self.m.zmean)/self.m.zscale
        expected=np.column_stack([h.predict(z) for h in self.m.heads])
        np.testing.assert_array_equal(expected,self.m.predict(x,self.asof,use_memory=False))
    def test_repeated_and_reordered_inference_has_no_state_mutation(self):
        x=self.s.context[-3:];keys=self.m.keys.copy();pred=self.m.predict(x,self.asof)
        np.testing.assert_allclose(pred,self.m.predict(x[::-1],self.asof)[::-1],rtol=1e-6,atol=1e-8)
        np.testing.assert_array_equal(self.m.keys,keys)
    def test_timezone_and_feature_availability_are_strict(self):
        from wharton_final.data import timestamp
        with self.assertRaises(ValueError):timestamp('2026-10-08T10:00:00')
        p=panel();p.observed[5]=p.decisions[6]
        s=samples(p,6,3,0);self.assertNotIn(6,s.rows)
    def test_missing_policy_fields_rejected(self):
        fields=dict(payment='cash_only',unpaid='carry',timing='before_trade',terminal='mark',fee_rate=0)
        for key in fields:
            other=dict(fields);other.pop(key)
            with self.assertRaises(TypeError):Policy(**other)
    def test_baseline_rows_and_inputs_are_bound(self):
        import json,tempfile
        from pathlib import Path
        from test_pipeline import fixture_files
        from wharton_final.pipeline import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_=fixture_files(root)
            evaluate(manifest,self.c,first_test_row=65,test_count=1,stride=3,output=root/'run')
            record=json.loads((root/'run/predictions.json').read_text())[0]
            self.assertEqual(record['matched_baseline_rows'],record['fit_rows'])
            self.assertTrue(set(record['fit_rows']).issubset(record['baseline_rows']))
            self.assertTrue(set(record['calibration_rows']).issubset(record['baseline_rows']))

class ClientAndFundingAuditTests(unittest.TestCase):
    def test_contribution_precedes_trade_and_return(self):
        from wharton_final.accounting import Contribution
        p=Policy('cash_only','carry','before_trade','mark',0)
        r=transition(State(2026,(0,),(1,),0),at=2027,returns=['.1'],policy=p,cash_rate=0,return_kind='price',target_weights=[1],contributions=[Contribution('first',300000)])
        self.assertEqual(r.contributed,300000);self.assertEqual(r.market_pnl,30000);self.assertEqual(r.state.wealth,330000);r.verify()
        with self.assertRaises(ValueError):transition(r.state,at=2028,returns=[0],policy=p,cash_rate=0,return_kind='price',contributions=[Contribution('first',150000)])
    def test_client_source_contributions_and_2033_order(self):
        from wharton_final.client_contract import contributions_for_year,operating_schedule,required_operating_reserve,facility_capacity
        p=Policy('cash_only','carry','before_trade','mark',0);s=State(2026,(0,),(1,),0)
        for year in range(2027,2033):s=transition(s,at=year,returns=[0],policy=p,cash_rate=0,return_kind='price',contributions=contributions_for_year(year)).state
        self.assertEqual(s.cash,450000);schedule=operating_schedule();self.assertEqual([x.due_at for x in schedule],list(range(2033,2043)))
        reserve=required_operating_reserve([0]*9);self.assertEqual(reserve,500000)
        capacity=facility_capacity(s.cash,reserve,10000);self.assertEqual(capacity['mechanical_ceiling'],0);self.assertEqual(capacity['reserve_and_buffer_gap'],60000)
        r=transition(s,at=2033,returns=[0],policy=p,cash_rate=0,return_kind='price',obligations=schedule)
        self.assertEqual(r.paid,50000);self.assertEqual(r.state.cash,400000);self.assertEqual(len(r.state.liabilities),9)
    def test_reserve_first_payment_is_not_discounted(self):
        from wharton_final.client_contract import required_operating_reserve
        from decimal import Decimal as D
        reserve=required_operating_reserve(['.02']*9)
        expected=sum(D(50000)/D('1.02')**k for k in range(10))
        self.assertAlmostEqual(float(reserve),float(expected),places=8)
        with self.assertRaises(ValueError):required_operating_reserve([0]*10)
        with self.assertRaises(TypeError):required_operating_reserve()
    def test_due_arrears_and_future_reserve_not_double_counted(self):
        p=Policy('cash_only','carry','after_market','mark',0)
        for cash,expected in [(100,0),(50,30)]:
            s=State(0,(0,),(1,),cash,(Obligation('future',5,80),))
            out=rollout(s,np.zeros((2,1,1)),[0],[[]],p,terminal_cash_target=0,cash_rates=[0],future_liability_policy='reserve_nominal_cash')
            self.assertEqual(out[0,2],0);self.assertEqual(out[0,-1],expected)
    def test_omitted_return_convention_rejected(self):
        with self.assertRaises(TypeError):transition(State(0,(0,),(1,),0),at=1,returns=[0],cash_rate=0,policy=Policy('cash_only','carry','after_market','mark',0))
    def test_future_reserve_policy_is_explicit_and_changes_objective(self):
        p=Policy('cash_only','carry','after_market','mark',0);s=State(0,(0,),(1,),0,(Obligation('later',5,80),));paths=np.zeros((2,1,1))
        with self.assertRaises(TypeError):rollout(s,paths,[0],[[]],p,terminal_cash_target=0,cash_rates=[0])
        nominal=rollout(s,paths,[0],[[]],p,terminal_cash_target=0,cash_rates=[0],future_liability_policy='reserve_nominal_cash')
        report=rollout(s,paths,[0],[[]],p,terminal_cash_target=0,cash_rates=[0],future_liability_policy='report_only')
        self.assertEqual(nominal[0,-1],80);self.assertEqual(report[0,-1],0);self.assertEqual(report[0,5],80)
