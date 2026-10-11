import argparse,json
from dataclasses import asdict
from pathlib import Path
from .model import Config
from .pipeline import evaluate,infer
from .data import timestamp,digest
from .accounting import State,Policy,Obligation,Contribution,transition
from .planning import scenarios,plan

def obligation(o):return Obligation(o['id'],o['due_at'],o['amount'])
def portfolio(x):return State(x['at'],tuple(x['units']),tuple(x['prices']),x['cash'],tuple(obligation(o) for o in x.get('liabilities',[])),tuple(obligation(o) for o in x.get('arrears',[])))
def main():
    p=argparse.ArgumentParser(description='Wharton amended development candidate; offline inputs only')
    sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('evaluate');e.add_argument('--manifest',required=True);e.add_argument('--config',required=True);e.add_argument('--first-test-row',type=int,required=True);e.add_argument('--test-count',type=int,required=True);e.add_argument('--stride',type=int,required=True);e.add_argument('--output',required=True)
    f=sub.add_parser('forecast');f.add_argument('--checkpoint',required=True);f.add_argument('--checkpoint-sha256',required=True);f.add_argument('--manifest',required=True);f.add_argument('--as-of',required=True);f.add_argument('--max-context-age-seconds',type=float,required=True);f.add_argument('--plan-contract');f.add_argument('--output',required=True)
    l=sub.add_parser('ledger');l.add_argument('--contract',required=True);l.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='evaluate':
        result=evaluate(a.manifest,Config(**json.loads(Path(a.config).read_text())),first_test_row=a.first_test_row,test_count=a.test_count,stride=a.stride,output=a.output)
        print(json.dumps(result['metrics'],indent=2));return
    if a.command=='forecast':
        pred,model,panel=infer(a.checkpoint,a.checkpoint_sha256,a.manifest,timestamp(a.as_of),max_context_age_seconds=a.max_context_age_seconds)
        result={'role':'development_forecast','forecast':dict(zip(panel.assets,pred.tolist())),'as_of':a.as_of,'last_observation_timestamp':float(panel.observed[panel.observed<timestamp(a.as_of)][-1]),'max_context_age_seconds':a.max_context_age_seconds,'checkpoint_sha256':a.checkpoint_sha256,'data_revision':panel.revision}
        if a.plan_contract:
            c=json.loads(Path(a.plan_contract).read_text());state=portfolio(c['state']);policy=Policy(**c['policy'])
            if tuple(c['assets'])!=panel.assets:raise ValueError('plan asset identities/order mismatch')
            history=panel.returns[panel.observed<timestamp(a.as_of)]
            paths=scenarios(history,pred,horizon=model.config.horizon,count=c['scenario_count'],seed=c['seed'])
            schedule=[[obligation(o) for o in step] for step in c['obligations']]
            result['plan']=asdict(plan(state,paths,schedule,policy,max_weight=c['max_weight'],alpha=c['alpha'],terminal_cash_target=c['terminal_cash_target'],cash_rates=c['cash_rates'],future_liability_policy=c['future_liability_policy']))
            result['plan_contract_sha256']=digest(a.plan_contract)
    else:
        c=json.loads(Path(a.contract).read_text());state=portfolio(c['state']);policy=Policy(**c['policy']);records=[]
        for step in c['steps']:
            step=dict(step);step['obligations']=tuple(obligation(o) for o in step.get('obligations',[]))
            step['contributions']=tuple(Contribution(o['id'],o['amount']) for o in step.get('contributions',[]))
            receipt=transition(state,policy=policy,**step);records.append(asdict(receipt));state=receipt.state
        result={'role':'deterministic_ledger','contract_sha256':digest(a.contract),'receipts':records}
    def encode(x):return x.tolist() if hasattr(x,'tolist') else str(x)
    # Avoid clobbering an existing artifact.
    with Path(a.output).open('x') as out:json.dump(result,out,indent=2,default=encode,allow_nan=False)

if __name__=='__main__':main()
