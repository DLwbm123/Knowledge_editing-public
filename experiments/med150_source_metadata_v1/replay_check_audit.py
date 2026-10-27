"""Audit fixed source candidates against already-consumed replay CHECK metadata."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]

def main():
    private=ROOT/'.private/med150_source_metadata_v1_20261028'
    selected=json.loads((private/'CANDIDATES_FROZEN.json').read_text())['selections']
    snapshot=json.loads((ROOT/'.private/prospective_confirmation_v111_20261025/EXPOSURE_SNAPSHOT.json').read_text())
    assert len(selected)==8 and len(snapshot['history_groups'])==177
    names=[x['source_group'] for x in selected];history=snapshot['history_groups']
    script='names='+repr(names)+'\nhistory='+repr(history)+'\n'+'''from pathlib import Path
import json
cfg=json.loads(Path('/data/bmw/.q.20261027.150/ENV.json').read_text());run=Path(cfg['REPLAY_PARENT'])
check=json.loads((run/'private/REPLAY_CHECK.json').read_text());fit=json.loads((run/'private/REPLAY_FIT.json').read_text())
def key(row):return str(row['source_group']).lower().split(':')[-1]
check_groups={key(v) for v in check};fit_groups={key(v) for v in fit}
assert len(check)==96 and len(check_groups)==32 and len(fit_groups)==64
print(json.dumps(dict(indices=[i for i,n in enumerate(names,1) if n in check_groups],check_count=32,fit_count=64,fit_overlap=len(set(names)&fit_groups),in_snapshot=len(check_groups&set(history)),outside_snapshot=len(check_groups-set(history)))))
'''
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000','/data/bmw/envs/v0/bin/python','-'],input=script,text=True,capture_output=True,timeout=30)
    if p.returncode:raise RuntimeError(p.stderr[-1500:])
    d=json.loads(p.stdout);assert d['check_count']==32 and d['fit_count']==64
    feasibility=json.loads(Path('/private/tmp/y.20261028.4/SOURCE_BASE_FEASIBILITY_PRIVATE.json').read_text())['groups']
    potential_overlap=sum(feasibility[i-1]['status']=='ANNOTATION_ASSUMED_BASE_POTENTIAL' for i in d['indices'])
    result=dict(time=time.time(),candidate_groups=8,already_consumed_check_groups=32,overlap_groups=len(d['indices']),potential_overlap=potential_overlap,
        check_groups_absent_prior_history=d['outside_snapshot'],check_groups_in_prior_history=d['in_snapshot'],fit_groups_overlap=d['fit_overlap'],
        source_QA_not_returned=True,full_old_development_role_json_parsed_in_remote_memory=True,unexposed_status='NOT_ESTABLISHED',candidate_replacement=False)
    target=private/'REPLAY_CHECK_EXPOSURE_AUDIT.private.json'
    # Preserve original audit and only add this bounded cross-check receipt.
    assert target.exists();(private/'REPLAY_CHECK_AUDIT_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
