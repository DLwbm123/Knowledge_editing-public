"""Offline endpoint eligibility and actual locality audit; old TQ locality stays NA."""
import json,math,importlib.util
from pathlib import Path

def main():
 root=Path(__file__).resolve().parents[2];out=root/'reports/vigor_teacherLocality3_v1_20261020/completion'
 v=json.loads((out/'RESULT.json').read_text());end=json.loads((out/'ENDED.json').read_text());s=json.loads((out/'SCORES.json').read_text());old=json.loads((root/'reports/vigor_ftmq3_v1_20261020/completion/RESULT.json').read_text());qual=json.loads((root/'reports/vigor_ftmq3_v1_20261020/completion/QUALIFICATION.json').read_text());checks=[]
 spec=importlib.util.spec_from_file_location('bindings',Path(__file__).with_name('teacherL3_ops.py'));ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
 def check(k,ok):checks.append(dict(field=k,passed=bool(ok)))
 check('terminal',v['status']==end['status']=='COMPLETE');check('scoring_terminal',s['status']=='COMPLETE')
 for c in v['checks']:check('runtime.'+c['field'],c['passed'])
 for k,b in v['restoration'].items():check('restoration.'+k,b is True)
 rows=[]
 for i,t in enumerate(v['teachers']):
  sample=t['sample'];check('case_order',sample==ops.AUTH['samples'][i]);o=next(x for x in old['teachers'] if x['sample']==sample);check(sample+'.fit_binding',ops.fit_matches(t,o));check(sample+'.source_output_binding',t['source_output_exact_binding'] is True);check(sample+'.restored',t['base_restored'] is True)
  g=t['endpoint'];native=g['native_nll'];drift=native[1]-t['base_native_nll'][1];cost=g['combined_native_fisher_cost'];verdict=dict(null=math.isfinite(drift) and abs(drift)<=.01,visual=math.isfinite(native[0]) and native[0]<t['base_native_nll'][0],trust=math.isfinite(cost) and 0<=cost<=.001)
  check(sample+'.gate_recomputed',g['gates']==verdict and abs(drift-g['cumulative_null_change'])<=1e-12 and g['accepted']==all(verdict.values())==t['allowed_endpoint_submission']);check(sample+'.native_repeat',all(abs(x-y)<=1e-5 for x,y in zip(native,t['teacher_native_nll'])))
  ev=[x for x in v['evaluation'] if x['sample']==sample];check(sample+'.six_complete_panel',len(ev)==6 and all(x['status']=='COMPLETE' for x in ev));damaged=[x['panel_id'] for x in ev if s['scores'].get(x['alias']) is False];missing=sum(s['scores'].get(x['alias']) is None for x in ev)
  # No new source Judge: old source agreement is reused only after exact full output binding.
  agreement=qual['criteria']['agreement']['scores'][sample+'_TQ'];faithful=qual['criteria']['faithfulness']['scores'][sample+'_TQ']
  check(sample+'.original_source_qualified',agreement is True and faithful is True)
  rows.append(dict(sample=sample,source_agreement_reused=agreement,source_faithfulness_reused=faithful,source_output_exact_binding=t['source_output_exact_binding'],fit_forwards=t['fit_forwards'],backwards=t['backward_calls'],native_source_NLL=native[0],native_null_NLL=native[1],base_null_NLL=t['base_native_nll'][1],null_drift=drift,endpoint_F0_cost=cost,endpoint_gates=verdict,allowed_endpoint_submission=g['accepted'],locality_damage=len(damaged),locality_denominator=6,locality_missing=missing,damaged_panel=damaged))
 check('18_complete_scores',len(s['scores'])==18 and not s['missing']);check('three_teachers',len(rows)==3)
 for e in s['judge_evidence']:
  check('judge_once',e['attempts_per_payload']==1 and e['retry_count']==0 and e['payloads']<=18);check('judge_isolation',not e['tool_event_types'] and all(e['isolation_checks'].values()))
 fit=sum(t['fit_forwards'] for t in v['teachers']);gen=sum(t['generation_forwards'] for t in v['teachers'])+sum(t['generation_forwards'] for t in v['evaluation']);controls=end['model_forward_calls']-fit-gen
 check('global_budget',end['model_forward_calls']<=3200 and end['resident_seconds']<=1800 and controls>=0 and fit<=78 and all(t['fit_forwards']<=26 for t in v['teachers']))
 damage={(r['sample'],p) for r in rows for p in r['damaged_panel']};missing=sum(r['locality_missing'] for r in rows);candidate_better=not missing and damage<={('E01','L006')};eligible=sum(r['allowed_endpoint_submission'] for r in rows)
 analysis=dict(rows=rows,summary=dict(source_qualified=3,source_denominator=3,source_new_judge_calls=0,locality_damage=len(damage),locality_denominator=18,locality_missing=missing,endpoint_null_pass=sum(r['endpoint_gates']['null'] for r in rows),endpoint_trust_pass=sum(r['endpoint_gates']['trust'] for r in rows),endpoint_eligible=eligible,candidate_better_on_development_panel=candidate_better,protected_success=candidate_better and eligible==3),costs=dict(GPU_seconds=end['resident_seconds'],total_forwards=end['model_forward_calls'],teacher_fit=fit,backwards=sum(t['backward_calls'] for t in v['teachers']),final_generation=gen,control_binding_restore=controls,peak_allocated_bytes=end['peak_allocated_bytes'],new_judge_payloads=s['new_payloads'],cached_locality_endpoints=sum(x!='new_blind_score' for x in s['provenance'].values()),checkpoint=0,student_edits=0),historical_TQ3_locality_remains_NA=True,near_locality_support=0,distinct_unrelated_queries=6,scope='Independent fixed reconstructed teacher endpoint diagnosis; no old controller variant or teacher tuning')
 audit=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',check_count=len(checks),failed=[x for x in checks if not x['passed']],checks=checks)
 for name,x in [('ANALYSIS.json',analysis),('AUDIT.json',audit)]:(out/name).write_text(json.dumps(x,indent=2)+'\n')
 print(json.dumps(dict(audit=audit['status'],checks=len(checks),failed=audit['failed'],**analysis),indent=2))
 if audit['status']!='PASS':raise SystemExit('Audit failed')
if __name__=='__main__':main()
