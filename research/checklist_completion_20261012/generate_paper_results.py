from pathlib import Path
import json, hashlib
root=Path.cwd(); out=root/'research/checklist_completion_20261012'
def read(p):return json.loads((root/p).read_text())
s3=read('outputs/frem_s3_20261011_attempt1/public_summary.json'); s2=read('outputs/frem_s2_development_20261011_attempt3/summary.json'); slow=read('outputs/public_equity_slow_20261011_attempt1/summary.json')
def table(headers,rows):
 return '\\begin{center}\n\\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'+'\n'.join(' & '.join(str(v).replace('_',r'\_') for v in row)+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{center}\n'
text=r'''\section{Completed public exploratory studies}
These studies are distinct from the proposed canonical confirmation and from the synthetic fixture above. They use public current-survivor histories, current-vintage adjustments and research-coordinate timestamps. They do not establish historical eligibility, executable global holdings, or investment profitability. No protected 2025 onward outcomes were accessed. Valid completed protocols are closed; no favorable arm replaces a predeclared primary.

\subsection{S2: improved fitting without a memory advantage}
The S2 campaign completed 22 learned fits and 24 prediction cells on 23 survivor stocks, 19 crisis months and two optimization seeds. Lower Student-t NLL is better. These are date-balanced descriptive scores; no significance test was performed. Both seeds selected zero residual-memory correction.
'''
text+=table(['Model','NLL'],[(k,f'{v:.6f}') for k,v in s2['date_balanced_nll'].items()])
text+=r'''Scaling improved optimization relative to legacy FREM, but the zero-return control remained best. The frozen residual and memory-off successor are identical in score. This does not support a memory-benefit or state-of-the-art claim.

\subsection{Six-window public slow study}
All 24 declared fits completed: six windows, four methods, seed 11 and five neural epochs. An explicitly recorded whole-month missing-data amendment was used; the strict original zero-quote rejection and earlier failures remain preserved. These single-seed comparisons have exploratory unadjusted uncertainty. Positive gain below is no-memory NLL minus FREM NLL.
'''
rows=[]
for fold in dict.fromkeys(c['fold'] for c in slow['cells']):
 cells={c['model']:c['metrics']['nll'] for c in slow['cells'] if c['fold']==fold}
 rows.append((fold,f"{cells['frem']:.6f}",f"{cells['no_memory']:.6f}",f"{cells['no_memory']-cells['frem']:+.6f}"))
text+=table(['Window','FREM','No memory','Gain'],rows)
text+=r'''FREM improved point NLL in four windows and worsened in 2020 and 2022. There is no pooled confirmatory or profitability conclusion.

\subsection{S3: primary memory result and uncertainty}
S3 freezes a calibrated base and applies causal delayed, versioned residual replay. It completed 24 learned public fits, 12 constant-baseline cells and 15 generated mechanism cells. The public sample contains twelve US current-survivor stocks, with 144 asset-month outcomes in January--December 2023 and 120 in January--October 2024. Seeds 11, 29 and 47 are optimization repeats, averaged before date-level inference. The primary remains combined memory versus the identical calibrated memory-off base. The practical threshold is 0.01 nats.
'''
rows=[]
for fold,c in s3['comparisons'].items():
 p=c['paired']['frem/off']['3'];rows.append((fold.replace('fresh_',''),f"{p['estimate']:+.6f}",f"[{p['ci'][0]:+.6f}, {p['ci'][1]:+.6f}]",str(p['months'])))
text+=table(['Window','Gain',r'Exploratory 95\% interval','Months'],rows)
text+=r'''Both three-month block intervals include zero and fail the practical-support criterion. The declared family retains Holm adjustment. The 2024 six-month sensitivity is unavailable because ten months cannot supply two six-month blocks; no block was shortened to force a result.

\subsection{Complete S3 memory arms}
The following gains use the same calibrated checkpoint with memory off as control. No tested arm improved both windows. Secondary point gains do not replace the primary comparison.
'''
a,b=[s3['comparisons'][f]['nll'] for f in ['fresh_2023','fresh_2024']]
text+=table(['Arm','2023 gain','2024 gain'],[(k.split('/')[1],f"{a['frem/off']-v:+.6f}",f"{b['frem/off']-b[k]:+.6f}") for k,v in a.items() if k.startswith('frem/')])
text+=r'''\subsection{Base controls and calibration}
'''
text+=table(['Model','2023 NLL','2024 NLL'],[(k,f'{a[k]:.6f}',f'{b[k]:.6f}') for k in a if k.endswith('/off')])
raw=[]
for fold in ['fresh_2023','fresh_2024']:
 vals=[c['raw_nll'] for c in s3['cells'] if c['name']=='frem' and c['fold']==fold];raw.append(sum(vals)/len(vals))
text+=f'Raw FREM before extra calibration scored {raw[0]:.6f} and {raw[1]:.6f}. Extra calibration worsened both windows. The no-JEPA control also had better point scores than calibrated FREM in both windows. These observations do not establish causal attribution or authorize retrospective primary selection.\n'
text+=r'''\subsection{Mechanism evidence and failure diagnosis}
Generated recurrence demonstrates behavior under planted information, not financial predictability. Its retrieval keys directly include the planted regime state. The null showed small harm within the declared tolerance. In real S3 predictions, retrieval and nonzero gates execute; selected corrections fail to transfer consistently. A later descriptive audit identifies tiny validation advantages, old base/calibration blocks and nearly flat within-neighbor similarity weighting. For unit-normalized 103-dimensional keys, mean squared distance is at most $4/103$, so $\exp(-d)$ is at least 0.9619. This is a scale-design concern, not proof that the completed experiment is invalid. A successor must test altered distance scaling and recent chronological fitting separately, retaining zero correction and matched-random controls.

\subsection{Provenance and evidence boundary}
Tables are generated directly from retained S2, slow-study and S3 JSON summaries by \path{research/checklist_completion_20261012/generate_paper_results.py}. Their input identities are saved in \path{research/checklist_completion_20261012/table_sources.json}. Raw predictions, freezes, calibration choices, failed attempts and source archives remain in their original run directories. The S3 receipt records 217 regression passes, not a fresh test run for this paper edit. Independent score reconstruction is distinct from retraining. Canonical confirmation, independent training reproduction, broad robustness and executable portfolio evidence remain incomplete.
'''
(root/'paper/public_results.tex').write_text(text)
files=['outputs/frem_s3_20261011_attempt1/public_summary.json','outputs/frem_s2_development_20261011_attempt3/summary.json','outputs/public_equity_slow_20261011_attempt1/summary.json']
(out/'table_sources.json').write_text(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print('Regenerated public results tables from retained JSON summaries')
