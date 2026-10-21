"""Offline strict-action contradiction audit with explicit scalar-NLL scope boundary."""
import json,math
from pathlib import Path

def main():
 root=Path(__file__).resolve().parents[2];out=root/'reports/vigor_jointAction3_v1_20261020/completion';v=json.loads((out/'RESULT.json').read_text());end=json.loads((out/'ENDED.json').read_text());cert=json.loads((out/'CERTIFICATE_AUDIT.json').read_text());checks=[]
 def check(k,ok):checks.append(dict(field=k,passed=bool(ok)))
 check('terminal',v['status']==end['status']=='COMPLETE')
 for x in v['checks']:check('runtime.'+x['field'],x['passed'])
 for k,x in v['restoration'].items():check('restoration.'+k,x is True)
 rows=[]
 for i,t in enumerate(v['teachers']):
  sample=t['sample'];check('fixed_source_order',sample==['E01','E02','E03'][i]);check(sample+'.fit_steps',t['fit_forwards']==[15,17,10][i] and t['backward_calls']==[14,16,9][i]);check(sample+'.exact_source',t['source_output_exact_binding']);check(sample+'.base_restore',t['base_restored']);check(sample+'.no_candidate_or_judge',t['new_candidate_edits']==t['new_judge_calls']==0)
  d=t['diagnosis'];a=d['source_space'];coverage=t['key_coverage'];check(sample+'.source_space',a['positions']==coverage['all_positions']==coverage['prefill_positions']+coverage['all_decode_positions'] and a['orthogonality_residual']<=1e-8 and a['rank']+a['remaining_input_freedom']==14336)
  check(sample+'.orthogonal_partition',abs(d['projected_fraction']**2+d['perpendicular_fraction']**2-1)<=1e-8)
  sep=d['separated_consistency'];joint=d['joint_consistency'];same=sep['consistent']==joint['consistent'];status=('CONSISTENT_STRONG_SUBPROBLEM' if joint['consistent'] else 'INCONSISTENT_STRONG_SUBPROBLEM') if same else 'UNRESOLVED_NUMERICAL_CLASSIFICATION'
  check(sample+'.two_consistency_classifications',d['numerical_status']==status and sep['consistent']==(sep['relative_residual']<=1e-8) and joint['consistent']==(joint['relative_residual']<=1e-8))
  w=d['exact_witness'];c=next(x for x in cert if x['sample']==sample);check(sample+'.exact_certificate',w['status']=='EXACT_STRICT_ACTION_CONTRADICTION' and w['shared_key_found'] and w['exact_nonzero'] and math.isfinite(w['absolute_action']) and w['absolute_action']>0 and c['exact_integer_nonzero'] and c['public_action_matches_private_integer'])
  rows.append(dict(sample=sample,source_rank=a['rank'],null_rank=d['null_space']['rank'],null_positions=d['null_positions'],remaining_input_dimensions=a['remaining_input_freedom'],projected_fraction=d['projected_fraction'],perpendicular_fraction=d['perpendicular_fraction'],per_column_perpendicular_fraction=d['per_column_perpendicular_fraction'],fixed_null_action_norm=d['fixed_null_action_norm'],perpendicular_rank=sep['rank'],perpendicular_condition_retained=sep['condition_retained'],joint_rank=joint['rank'],separated_relative_residual=sep['relative_residual'],joint_relative_residual=joint['relative_residual'],numerical_status=status,exact_shared_key_contradiction=True,exact_nonzero_action_absolute=w['absolute_action']))
 fit=sum(t['fit_forwards'] for t in v['teachers']);gen=sum(t['generation_forwards'] for t in v['teachers']);controls=end['model_forward_calls']-fit-gen
 check('all_three',len(rows)==3);check('global_budget',end['model_forward_calls']<=600 and end['resident_seconds']<=900 and controls>=0 and fit<=78)
 analysis=dict(rows=rows,summary=dict(exact_strict_action_contradictions=sum(r['exact_shared_key_contradiction'] for r in rows),denominator=3,source_output_binding=3,new_candidates=0,new_judge_calls=0,locality_new_support=0,scalar_null_NLL_feasibility='UNTESTED_NOT_INFERRED',new_semantic_performance_claim=False),costs=dict(GPU_seconds=end['resident_seconds'],total_forwards=end['model_forward_calls'],teacher_fit=fit,teacher_source_generation=gen,control_null_capture_binding_restore=controls,backwards=sum(t['backward_calls'] for t in v['teachers']),peak_allocated_bytes=end['peak_allocated_bytes'],checkpoint=0,judge=0),scope='Only exact successful source layer action plus unchanged null layer action is inconsistent; original scalar null NLL gate and semantic-equivalent Pure-W feasibility remain open')
 audit=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,check_count=len(checks),failed=[c for c in checks if not c['passed']])
 for n,x in [('ANALYSIS.json',analysis),('AUDIT.json',audit)]:(out/n).write_text(json.dumps(x,indent=2)+'\n')
 print(json.dumps(dict(audit=audit['status'],checks=len(checks),failed=audit['failed'],**analysis),indent=2))
 if audit['status']!='PASS':raise SystemExit('Audit failed')
if __name__=='__main__':main()
