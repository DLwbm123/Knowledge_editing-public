"""Validate complete assistant Base review and stage a deterministic conditional admission."""
import argparse,json,time
from pathlib import Path

def stage(private,remote):
    def read(p):return json.loads(p.read_text())
    def write(n,x):(private/n).write_text(json.dumps(x,indent=2)+'\n')
    m=read(private/'CANDIDATES_FROZEN.json');v=read(private/'BASE_VERDICTS.json')['rows'];labels={x['query_id']:x['label'] for x in v}
    assert len(labels)==len(v)==len(m['rows']) and set(labels)=={x['query_id'] for x in m['rows']}
    assert set(labels.values())<={'CORRECT','INCORRECT','UNCERTAIN','CONTRADICTORY'}
    candidates=m['candidates'];wrong=[c for c in candidates if all(labels[r['query_id']]=='INCORRECT' for r in c['rows'])];correct=[c for c in candidates if all(labels[r['query_id']]=='CORRECT' for r in c['rows'])]
    missing=[]
    if not wrong:missing.append('No clearly incorrect native/cross pair')
    if len(correct)<2:missing.append('Fewer than two Base-correct native/cross CT groups')
    for part in (0,1):
        receipt=read(private/'remote'/str(part)/'COMPLETE.json');assert receipt['status']=='BASE_QUALIFICATION_COMPLETE'
    if missing:
        a=dict(status='DATA_GATE_NOT_MET',missing=missing,wrong_pair_count=len(wrong),correct_pair_count=len(correct),six_arm_started=False,time=time.time());write('DATA_ADMISSION.json',a);return a
    retention=correct[0];tasks=[]
    for i,c in enumerate(wrong[:2]):
        key='E'+str(i+1).zfill(2);source=c['rows'][0];rephrase=dict(source,query_id=key+'_rephrase',question=c['original']['rephrase'],audit_role='TEXT_GENERALIZATION')
        tasks.append(dict(key=key,source=source,rephrase=rephrase,retention=retention['rows'][0],retention_correct=retention['original']['pred'],retention_wrong=retention['original']['alt'],source_component=c['component']))
    reuse={r['query_id']:remote+'/'+str((c['index']-1)%2)+'/outputs/BASE/'+r['query_id']+'.json' for c in candidates for r in c['rows']}
    write('BASE_REUSE.json',reuse);write('EVAL_ROWS.json',m['rows'])
    a=dict(status='PASS',tasks=tasks,retention_construct_group=retention['index'],prespecified_independent_check_group=correct[1]['index'],wrong_pair_count=len(wrong),correct_pair_count=len(correct),candidate_count=len(candidates),all_candidates_in_evaluation=True,time=time.time(),frozen_before_method_results=True,independent_patient_confirmation=False)
    write('DATA_ADMISSION.json',a);return a

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True,type=Path);p.add_argument('--remote',required=True);a=p.parse_args();v=stage(a.private,a.remote);print(json.dumps(dict(status=v['status'],tasks=len(v.get('tasks',[])),wrong_pairs=v['wrong_pair_count'],correct_pairs=v['correct_pair_count'])))
