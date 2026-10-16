"""Record exact observed engineering evidence without inferring scientific support."""
from pathlib import Path
from datetime import datetime, UTC
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE/'integrated_verification_attempt2'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

commands = json.loads((OUT/'commands.json').read_text())
assert len(commands)==11 and all(row['exit_code']==0 for row in commands)
assert json.loads((OUT/'archive_inspection.json').read_text())['status']=='PASS'
match = re.search(r'(\d+) passed in ([\d.]+)s', (OUT/'tests.log').read_text())
assert match
count, seconds = int(match[1]), float(match[2])
prior = json.loads((HERE/'predecessor_check_final.json').read_text())
assert prior['checked']==48 and not prior['mismatches'] and prior['journal_before_prefix_preserved']
assert json.loads((HERE/'import_boundary_final.json').read_text())['status']=='PASS'
paths = sorted((ROOT/'successors/frem_s10/src/frem_s10').rglob('*.py'))
paths += sorted((ROOT/'successors/frem_s10/tests').glob('test_*.py'))
paths += [ROOT/'successors/frem_s10'/s for s in ['README.md','pyproject.toml','uv.lock','src/frem_s10/science.json','src/frem_s10/py.typed']]
source = {str(p.relative_to(ROOT)):sha(p) for p in paths}
snapshot = HERE/'integrated_verified_snapshot'
snapshot.mkdir(exist_ok=False)
for path in paths:
    dest = snapshot/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
write(HERE/'integrated_verified_snapshot.json', {'scope':'exact source and tests for integrated_verification_attempt2','files':source})
logs = list(HERE.glob('*.log')) + list(OUT.glob('*.log'))
logs += [HERE/q for q in ['host_admission.json','host_admission_final.json','predecessor_check_final.json','import_boundary_final.json','VERIFICATION_FINAL.json','environment.json','isolated_environment.json','experiment_engineering_verification.json','late_integrity_review.json','CHECKPOINTS.json','REQUIREMENT_STATUS.json','REPORT.md','FLAW_CHECKLIST.md','VERIFICATION_SUPPLEMENT.json','DESIGN_ADDENDUM_CL15.md','DESIGN_ADDENDUM_CL16.md']]
logs += [OUT/'commands.json',OUT/'archive_inspection.json', OUT/'plan.json',OUT/'wheel_plan.json',OUT/'requirements.txt',OUT/'runner_source.py']
logs += list((OUT/'dist').iterdir())
receipt = {
 'schema_version':1,'id':'FREM-S10-INTEGRATED-REPAIR-20261015','timestamp':datetime.now(UTC).isoformat(),
 'status':'GENERATED_ENGINEERING_VERIFIED_SCIENTIFIC_EXECUTION_BLOCKED',
 'design_id':'FREM-S10-DESIGN-4-CL16','design_revision':4,
 'scientific_status':'NOT_EVALUATED','positive_market_result_established':False,
 'research_complete':False,'full_pipeline_complete':False,'run_ready':False,
 'market_fits':0,'market_policy_replays':0,'protected_access':False,
 'new_scientific_freeze':False,'completed_predecessors_reopened':False,
 'original_thresholds_changed':False,'s9_freeze_consumed':False,
 'tests':{'passed':count,'pytest_seconds':seconds,'command':commands[0],
          'warnings':'error','retained_fixture_directory':str((OUT/'pytest_artifacts').relative_to(ROOT))},
 'generated_evidence':{'matrix_cells':192,'matrix_blocks_per_fold':2,'matrix_law_atoms':3,
    'matrix_policy_and_accounting':'actual production functions on generated inputs',
    'forecast_kernel':'all six variants in separate bounded cases, including kernel-to-ledger bridge',
    'precision':'actual primary kernel, fixed pre2010 scoped generated inputs; no real precision claim',
    'analysis':'separate24/25-date constructed retained-record fixtures; not an actual market replay',
    'readback':'actual saved192-cell small generated replay roundtrip; fitting/selection calls forbidden',
    'supervisor':'real PID/RSS metadata with generated host admission counters, not scientific resource admission'},
 'verification_catalog':json.loads((HERE/'VERIFICATION_FINAL.json').read_text())['counts'],
 'known_flaws':{'retained':31,'additional':24,'total':55},
 'quality':{'strict_mypy_source_and_tests':True,'ruff':True,'format':True,'static_import_boundary':True},
 'package':{'fresh_clean_environment':True,'offline_no_index_require_hashes':True,'archive_source_match':True,
   'validation_platform':'CPython3.12.14 macOSarm64; not cross-platform certification'},
 'historical_identities_checked':48,'historical_mismatches':0,'old_journal_prefix_preserved':True,
 'source_sha256':source,'evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in logs},
 'host_admission':json.loads((HERE/'host_admission_final.json').read_text()),
 'independent_empirical_reproduction':False,
 'remaining_internal_gates':[
    'Operation-specific scientific receipt producers and fixed launcher/supervisor/freeze integration',
    'Weekly mixture certificates/maturity/state archival and complete scientific saved-output audit',
    'Full verification catalog closure and maximum-support/length time/RSS profiling',
    'Evidence-bound full empirical paper, figures and local research release'],
 'remaining_scientific_external_gates':[
    'Real source/calendar/close-clock admission and rights',
    'Fresh passing512MiB/pageout resource admission',
    'Fixed distinct real pre2010 precision preparation',
    'Exact prospective development freeze and one bounded campaign only if precision passes',
    'Unused-data confirmation and independent empirical reproduction'],
 'report':'research/frem_s10_repair_20261015/REPORT.md',
 'checklist':'research/frem_s10_repair_20261015/FLAW_CHECKLIST.md',
 'failed_attempts':'Earlier logs/receipts and both source snapshots retained; no repair replaces old results'
}
write(HERE/'RECEIPT.json',receipt)
decision={
 'timestamp':receipt['timestamp'],'iteration':'FREM-S10','design_revision':4,
 'observation':'Partial interrupted implementation and valid predecessor failures did not establish a viable S10 experiment or positive market signal.',
 'hypothesized_cause':'Concrete numerical, identity, event-ownership, lifecycle, integration and evidence-scope defects; empirical signal remains unknown.',
 'change':'Isolated generated-data implementation and repair of causal forecasts, policy/benchmarks,192-cell replay,persistence/readback,paired inference,pilot isolation,supervisor and structural declarations; prospective CL15/CL16 retained.',
 'predicted_effect':'Fail closed on tested malformed evidence, preserve causal/accounting identities and enable traceable bounded experiments; no predicted profit or guaranteed signal.',
 'discriminating_experiment':'Independent constructed references, chronology/corruption/failure cases, actual small192-cell generated replay/readback, separate real-kernel bridge and scoped pilot, strict quality and clean offline wheel verification.',
 'result':receipt['status'],'tests_passed':count,'receipt':'research/frem_s10_repair_20261015/RECEIPT.json',
 'receipt_sha256':sha(HERE/'RECEIPT.json'),'scientific_status':'NOT_EVALUATED','market_fits':0,'market_policy_replays':0,
 'protected_access':False,'new_scientific_freeze':False,'primary_thresholds_changed':False,
 'history_preserved':True,'scientific_terminal_result_required_before_research_complete':True,
 'design_addenda':['research/frem_s10_repair_20261015/DESIGN_ADDENDUM_CL15.md','research/frem_s10_repair_20261015/DESIGN_ADDENDUM_CL16.md']}
write(ROOT/'research/decisions/frem_s10_repair_20261015.json',decision)
print(json.dumps({'tests_passed':count,'pytest_seconds':seconds,'source_files':len(source),'receipt':str(HERE/'RECEIPT.json')}))
