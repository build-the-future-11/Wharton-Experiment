import itertools,unittest
from decimal import Decimal as D
from wharton_final.accounting import State,Policy,Obligation,transition

def policy(**kw):
    options=dict(payment='cash_only',unpaid='carry',timing='after_market',terminal='mark',fee_rate='0');options.update(kw);return Policy(**options)
def step(s,**kw):
    args=dict(at=s.at+1,returns=[0]*len(s.units),policy=policy(),cash_rate=0,return_kind='price');args.update(kw);return transition(s,**args)

class LedgerTests(unittest.TestCase):
    def test_zero_partial_full_payments(self):
        for cash,paid,left in [('0','0','10'),('4','4','6'),('10','10','0'),('20','10','0')]:
            receipt=step(State(0,(0,),(1,),cash),obligations=[Obligation('a',1,'10')])
            self.assertEqual(receipt.paid,D(paid));self.assertEqual(receipt.state.debt,D(left));receipt.verify()
    def test_original_m11_regression(self):
        s=State(0,(0,),(1,),'1');a=step(s,obligations=[Obligation('a',1,'.1')]);b=step(a.state,obligations=[Obligation('b',2,'.1')])
        self.assertEqual(a.state.cash,D('.9'));self.assertEqual(b.state.cash,D('.8'))
    def test_payment_carries_wealth_forward(self):
        a=step(State(0,(10,),(10,),0),policy=policy(payment='sell_pro_rata'),obligations=[Obligation('a',1,20)]);b=step(a.state,returns=['.1'])
        self.assertEqual(a.state.wealth,80);self.assertEqual(b.state.wealth,88);self.assertEqual(b.state.cash,0)
    def test_capital_gain_is_not_cash(self):
        a=step(State(0,(10,),(10,),0),returns=['.1'],obligations=[Obligation('a',1,5)])
        self.assertEqual(a.market_pnl,10);self.assertEqual(a.state.cash,0);self.assertEqual(a.state.debt,5)
    def test_fees_on_executed_trades(self):
        s=State(0,(5,5),(10,10),0);a=step(s,target_weights=['.5','.5'],policy=policy(fee_rate='.01'))
        self.assertEqual(a.fees,0)
        b=step(s,target_weights=[0,0],policy=policy(fee_rate='.01'))
        self.assertEqual(b.fees,1);self.assertEqual(b.turnover,100);self.assertEqual(b.state.cash,99)
    def test_purchase_is_self_financing(self):
        a=step(State(0,(0,),(10,),100),target_weights=[1],policy=policy(fee_rate='.1'))
        self.assertAlmostEqual(float(a.state.wealth),100/1.1);self.assertGreaterEqual(a.state.cash,0);a.verify()
    def test_sale_funds_payment_net_fees(self):
        a=step(State(0,(10,),(10,),0),policy=policy(payment='sell_pro_rata',fee_rate='.1'),obligations=[Obligation('a',1,45)])
        self.assertEqual(a.paid,45);self.assertEqual(a.fees,5);self.assertEqual(a.state.wealth,50)
    def test_insufficient_sale_records_arrears(self):
        a=step(State(0,(10,),(10,),0),policy=policy(payment='sell_pro_rata',fee_rate='.1'),obligations=[Obligation('a',1,110)])
        self.assertEqual(a.paid,90);self.assertEqual(a.state.debt,20);self.assertEqual(a.state.wealth,0)
    def test_arrears_not_new_liabilities(self):
        a=step(State(0,(1,),(10,),0),obligations=[Obligation('a',1,5)]);b=step(a.state,dividends=[3]);c=step(b.state,dividends=[3])
        self.assertEqual(b.new_liabilities,0);self.assertEqual(b.state.debt,2);self.assertEqual(c.state.debt,0);self.assertEqual(c.state.cash,1)
    def test_writeoff_explicit(self):
        a=step(State(0,(0,),(1,),2),policy=policy(unpaid='write_off'),obligations=[Obligation('a',1,5)])
        self.assertEqual(a.write_off,3);self.assertEqual(a.state.debt,0);self.assertEqual(a.payments[0].unpaid,3)
    def test_future_liability_not_paid_early(self):
        a=step(State(0,(0,),(1,),20),obligations=[Obligation('a',3,5)]);b=step(a.state);c=step(b.state)
        self.assertEqual(a.paid,0);self.assertEqual(a.state.equity,15);self.assertEqual(b.paid,0);self.assertEqual(c.paid,5)
    def test_fifo_deterministic(self):
        a=step(State(0,(0,),(1,),7),obligations=[Obligation('z',1,5),Obligation('a',1,5)])
        self.assertEqual([p.id for p in a.payments],['a','z']);self.assertEqual(a.payments[1].paid,2)
    def test_corporate_actions_and_price_order(self):
        a=step(State(0,(10,),(100,),0),splits=[2],returns=['-.02'],dividends=[1])
        self.assertEqual(a.state.units,(D(20),));self.assertEqual(a.state.prices,(D(49),));self.assertEqual(a.state.cash,20);self.assertEqual(a.state.wealth,1000)
    def test_total_return_double_count_rejected(self):
        with self.assertRaises(ValueError):step(State(0,(1,),(1,),1),return_kind='total',dividends=[1])
    def test_time_order_changes_funding_but_conserves(self):
        s=State(0,(1,),(10,),0);obs=[Obligation('a',1,1)]
        a=step(s,policy=policy(timing='before_trade'),dividends=[1],obligations=obs);b=step(s,policy=policy(timing='after_market'),dividends=[1],obligations=obs)
        self.assertEqual(a.paid,0);self.assertEqual(b.paid,1);a.verify();b.verify()
    def test_terminal_cash_is_not_marked_wealth(self):
        s=State(0,(1,),(10,),2);a=step(s,terminal=True);b=step(s,terminal=True,policy=policy(terminal='liquidate',fee_rate='.1'))
        self.assertEqual(a.state.cash,2);self.assertEqual(a.state.wealth,12);self.assertEqual(b.state.cash,11);self.assertEqual(b.state.units,(D(0),))
        with self.assertRaises(ValueError):step(b.state)
    def test_duplicate_obligation_and_bad_clock(self):
        a=step(State(0,(0,),(1,),10),obligations=[Obligation('a',1,1)])
        with self.assertRaises(ValueError):step(a.state,obligations=[Obligation('a',2,1)])
        with self.assertRaises(ValueError):step(a.state,at=1)
    def test_reject_invalid_contracts(self):
        for weights in ([-1],[1.1],[0,1]):
            with self.assertRaises(ValueError):step(State(0,(0,),(1,),10),target_weights=weights)
        with self.assertRaises(ValueError):policy(fee_rate=1)
        with self.assertRaises(ValueError):step(State(0,(0,),(1,),10),returns=['NaN'])
    def test_all_policy_combinations_conserve(self):
        for payment,unpaid,timing,terminal in itertools.product(('cash_only','sell_pro_rata'),('carry','write_off'),('before_trade','after_market'),('mark','liquidate')):
            p=Policy(payment,unpaid,timing,terminal,'.01');s=State(0,(1,2),(10,20),3)
            for t in range(1,4):
                receipt=step(s,policy=p,target_weights=['.3','.4'],returns=['.1','-.2'],dividends=['.1','.2'],obligations=[Obligation(str(t),t,15)],terminal=t==3)
                receipt.verify();self.assertGreaterEqual(receipt.state.cash,0);s=receipt.state
    def test_total_loss_and_zero_mark(self):
        a=step(State(0,(1,),(10,),0),returns=[-1],obligations=[Obligation('a',1,1)],policy=policy(payment='sell_pro_rata'))
        self.assertEqual(a.state.wealth,0);self.assertEqual(a.state.debt,1)
        with self.assertRaises(ValueError):step(State(0,(0,),(0,),10),target_weights=[1])
    def test_cash_interest_separate(self):
        a=step(State(0,(1,),(10,),5),returns=['.1'],cash_rate='.1')
        self.assertEqual(a.market_pnl,1);self.assertEqual(a.cash_interest,D('.5'));self.assertEqual(a.state.cash,D('5.5'))
