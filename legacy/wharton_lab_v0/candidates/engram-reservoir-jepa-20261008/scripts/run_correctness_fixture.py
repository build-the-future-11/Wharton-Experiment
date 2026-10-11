"""Retained end-to-end synthetic correctness run; never market evidence."""
import json,sys,subprocess
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_pipeline import fixture_files
from wharton_final.model import Config
from wharton_final.data import digest

output=ROOT/'evidence'/'review-correctness-fixture'
output.mkdir(exist_ok=False)
manifest,panel=fixture_files(output)
config=output/'config.json';config.write_text(json.dumps(asdict(Config(context=6,horizon=3,latent=4,reservoir=5,epochs=2,head='xgboost')),indent=2)+'\n')
commands=[]
def run(args):
    command=[sys.executable,'-m','wharton_final',*args]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=45)
    commands.append({'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    result.check_returncode()
run(['evaluate','--manifest',str(manifest),'--config',str(config),'--first-test-row','65','--test-count','1','--stride','3','--output',str(output/'run')])
checkpoint=output/'run'/'model-65.pkl'
plan={'purpose':'synthetic correctness fixture, not client policy','assets':['A','B'],'state':{'at':0,'units':[0,0],'prices':[1,1],'cash':100},'policy':{'payment':'cash_only','unpaid':'carry','timing':'after_market','terminal':'mark','fee_rate':0},'scenario_count':2,'seed':11,'obligations':[[{'id':'a','due_at':1,'amount':1}],[],[]],'max_weight':.8,'alpha':.5,'terminal_cash_target':90,'cash_rates':[0,0,0],'future_liability_policy':'reserve_nominal_cash'}
contract=output/'plan.json';contract.write_text(json.dumps(plan,indent=2)+'\n')
run(['forecast','--checkpoint',str(checkpoint),'--checkpoint-sha256',digest(checkpoint),'--manifest',str(manifest),'--as-of',datetime.fromtimestamp(panel.decisions[-1]+86400,timezone.utc).isoformat(),'--max-context-age-seconds','86400','--plan-contract',str(contract),'--output',str(output/'forecast-plan.json')])
run(['ledger','--contract',str(ROOT/'configs'/'ledger-correctness.json'),'--output',str(output/'ledger.json')])
print('Synthetic correctness artifacts retained:',output)
