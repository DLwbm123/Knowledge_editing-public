from pathlib import Path
import json,csv
p=Path('reports/vigor_amplitude3_v1_20261018/completion')
r=json.loads((p/'RESULT.json').read_text());rows=[]
for pt in r['points']:
 c=pt['candidate']['correction'];m=c.get('compensation',{})
 row=dict(sample=pt['sample'],step=pt['step'],status=pt['status'],passed=pt['passed'],candidate_count=pt['candidate_count'],rho=m.get('rho'),multiplier=m.get('multiplier'),realized_effect_ratio_after=m.get('realized_effect_ratio_after'))
 for branch in ['control','candidate']:
  b=pt[branch];v=b['verdict'];corr=b['correction']
  row.update({branch+'_'+k:value for k,value in dict(visual_nll=v['native_nll'][0],null_nll=v['native_nll'][1],null_drift=v['cumulative_null_change'],null_margin=b['null_margin'],visual_gain=b['visual_gain_from_step_start'],fisher_cost=v['combined_native_fisher_cost'],null_pass=v['gates']['null'],visual_pass=v['gates']['visual'],trust_pass=v['gates']['trust'],joint_pass=v['accepted'],fp32_null_linear_change=corr.get('fp32_update',{}).get('null_linear_change'),native_null_linear_change=corr.get('native_update',{}).get('null_linear_change'),rounding_l2=corr.get('quantization',{}).get('absolute_rounding_l2'),rounding_relative=corr.get('quantization',{}).get('relative_rounding_l2')).items()})
 row['max_nll_reconstruction_difference']=max(c.get('absolute_difference',0) for c in pt['checks'])
 row['max_E_reconstruction_difference']=max(c.get('relative_difference',0) for c in pt['checks'])
 row['checks']=len(pt['checks']);row['base_restored']=pt['base_restored']
 rows.append(row)
with (p/'POINT_SUMMARY.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
s=dict(status=r['status'],denominator=3,passed=sum(pt['passed'] for pt in r['points']),points=rows,elapsed_seconds=r['elapsed_seconds'],peak_allocated_bytes=r['peak_allocated_bytes'])
(p/'SUMMARY.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s,indent=2))
