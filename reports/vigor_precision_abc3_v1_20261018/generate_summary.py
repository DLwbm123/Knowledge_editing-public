from pathlib import Path
import json,csv
p=Path.cwd()/'reports/vigor_precision_abc3_v1_20261018/completion'
r=json.loads((p/'RESULT.json').read_text()); rows=[]
for point in r['points']:
 a=point['attribution']
 for branch,b in a['branches'].items():
  rows.append(dict(sample=point['sample'],step=point['step'],branch=branch,visual_nll=b['nll'][0],null_nll=b['nll'][1],initial_visual_nll=b['initial_nll'][0],initial_null_nll=b['initial_nll'][1],step_start_visual_nll=b['step_start_nll'][0],step_start_null_nll=b['step_start_nll'][1],cumulative_null_change=b['cumulative_null_change'],null_margin=b['null_margin'],visual_gain_from_step_start=b['visual_gain_from_step_start'],combined_native_fisher_cost=b['combined_native_fisher_cost'],null_pass=b['gates']['null'],visual_pass=b['gates']['visual'],trust_pass=b['gates']['trust'],all_gates=b['all_gates'],original_method=b['original_method']))
with (p/'ABC_VALUES.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
summary=[]
for pt in r['points']:
 a=pt['attribution'];bs=a['branches']
 summary.append(dict(sample=pt['sample'],step=pt['step'],**{k:v for k,v in a.items() if k!='branches'},reconstruction_checks=len(pt['checks']),max_nll_difference=max([x.get('absolute_difference',0) for x in pt['checks']]),max_energy_relative_difference=max([x.get('relative_difference',0) for x in pt['checks']]),base_restored=pt['base_restored']))
(p/'SUMMARY.json').write_text(json.dumps(dict(points=summary,elapsed_seconds=r['elapsed_seconds'],peak_allocated_bytes=r['peak_allocated_bytes']),indent=2)+'\n')
print(json.dumps(dict(points=summary,branches=rows,elapsed_seconds=r['elapsed_seconds'],peak_allocated_bytes=r['peak_allocated_bytes']),indent=2))
