"""Freeze a role-safe metadata-only TRAIN_DEV candidate census."""
import collections
import json
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
PRIVATE=ROOT/'.private/med150_source_metadata_v1_20261028'


def read(p):return json.loads(Path(p).read_text())
def normalize(v):
    v=str(v).lower();assert re.fullmatch(r'xmlab\d+',v), 'Unrecognized source identifier'
    return v


def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU census wall limit')));signal.alarm(1800)
    assert not PRIVATE.exists();PRIVATE.mkdir(parents=True)
    snap=read(ROOT/'.private/prospective_confirmation_v111_20261025/EXPOSURE_SNAPSHOT.json')
    v111=read(ROOT/'.private/prospective_confirmation_v111_20261025/CONDITIONAL_CANDIDATES_FROZEN.json')
    assert len(snap['history_groups'])==177 and len(snap['V95_reserved'])==24 and len(snap['V106_audited'])==8 and len(snap['replay64'])==64
    assert len(v111['all_remaining_metadata'])==15 and v111['source_contents_used'] is False
    exclusions={k:{normalize(x) for x in snap[k]} for k in ('history_groups','V95_reserved','V106_audited','replay64')}
    exclusions['V111_all15']={normalize(x['source_group']) for x in v111['all_remaining_metadata']}
    meta=read(ROOT/'.private/med146_single_sequential_v133_20261027/scoring_recovery1/PACKET_METADATA_B.json')
    extra=read(ROOT/'.private/med150_feasibility_20261027/ROLE_SAFE_CANDIDATES.private.json')['items']
    # Other datasets use different source identifiers and are not SLAKE groups.
    allnames=[t['native']['source_group'] for t in meta['tasks']]+[t['task']['native']['source_group'] for t in extra]
    exclusions['MED150_slake']={normalize(x) for x in allnames if re.fullmatch(r'xmlab\d+',str(x).lower())}
    union=set().union(*exclusions.values())
    script='''from pathlib import Path
import collections,json,re
root=Path('/data/bmw/DataP/knowledge_editing/data/m3bench/SLAKE')
meta={}
for split in ('train','validation','test'):
 rows=json.loads((root/(split+'.json')).read_text())
 groups=set();openq=collections.defaultdict(set)
 for row in rows:
  g=Path(row['img_name']).parts[0].lower()
  assert re.fullmatch(r'xmlab\\d+',g)
  groups.add(g)
  if row.get('q_lang')=='en' and row.get('answer_type')=='OPEN':openq[g].add(int(row['qid']))
 meta[split]={'groups':sorted(groups),'open':{g:sorted(v) for g,v in openq.items()},'rows':len(rows)}
print(json.dumps(meta))
'''
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000','/data/bmw/envs/v0/bin/python','-'],input=script,text=True,capture_output=True,timeout=300)
    assert p.returncode==0,p.stderr[-1500:];splits=json.loads(p.stdout)
    cross=(set(splits['train']['groups'])&set(splits['validation']['groups']))|(set(splits['train']['groups'])&set(splits['test']['groups']))
    eligible={normalize(g):ids for g,ids in splits['train']['open'].items() if len(ids)>=2 and g not in union and g not in cross}
    ordered=sorted(eligible,key=lambda x:int(x[5:]));chosen=ordered[:8]
    output=dict(status='METADATA_FROZEN_NOT_SOURCE_ADMITTED',time=time.time(),role='TRAIN_DEV_METADATA_CANDIDATE',
        selections=[dict(source_group=g,qid_pool=eligible[g]) for g in chosen],counts=dict(train_rows=splits['train']['rows'],train_groups=len(splits['train']['groups']),train_open_at_least_two_groups=sum(len(v)>=2 for v in splits['train']['open'].values()),cross_split_train_groups=len(cross),excluded_union_groups=len(union),eligible_groups=len(eligible),selected_groups=len(chosen),exclusion_counts={k:len(v) for k,v in exclusions.items()}),
        model_calls=0,gpu_calls=0,answers_used=False,questions_used=False,pixel_reads=0,patient_independence='UNKNOWN',unseen='NO_MATCH_ENUMERATED_METADATA_ONLY',backfill=False)
    (PRIVATE/'CANDIDATES_FROZEN.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='selections'}))

if __name__=='__main__':main()
