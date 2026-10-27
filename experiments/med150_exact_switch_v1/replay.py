"""Freeze exact-switch choices without answers, verify old inputs, reuse labels."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.private/med150_exact_switch_v1_20261028'


def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);assert not p.exists(),str(p);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')


def remote(code,timeout=300):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000','/data/bmw/envs/v0/bin/python','-'],input=code,text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr[-1800:])
    assert len(p.stdout.encode())<20_000_000
    return json.loads(p.stdout)


def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU hard limit')));signal.alarm(1800)
    assert not OUT.exists();OUT.mkdir(parents=True,mode=0o700);clock=dict(original_start=time.time(),hard_end=time.time()+1800)
    write(OUT/'CLOCK.json',clock);stat=os.statvfs(OUT);assert stat.f_bavail*stat.f_frsize>8*1024**3
    select=(ROOT/'experiments/med150_exact_switch_v1/exact_select.py').read_text()
    # Phase 1 reads source questions and route-only records; no answer/score files.
    code='select_code='+repr(select)+'\n'+'''from pathlib import Path
import json,sys,os,signal,time,torch
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU remote bound')));signal.alarm(1200)
q=Path('/data/bmw/.q.20261027.150');os.environ.update(json.loads((q/'ENV.json').read_text()));r=Path(os.environ['EXPERIMENT_ROOT']);p=Path(os.environ['PARENT_RUN']);sys.path[:0]=[str(r/'tools'),str(p/'private/tools'),str(p/'private/source'),str(p/'private/official_llava')]
import module_run
scope={'__name__':'exact_select'};exec(compile(select_code,'<exact_select>','exec'),scope);assert scope['selfcheck']()['status']=='PASS'
med=Path('/data/bmw/Knowledge_editing/outputs/med150-v1-20261027/run');modal=Path('/data/bmw/Knowledge_editing/outputs/med150-modal-v1-20261028/run')
assert json.loads((med/'EVAL_GUARD_EXIT.json').read_text())['code']==json.loads((modal/'EVAL_GUARD_EXIT.json').read_text())['code']==0
pkg=torch.load(med/'retained/MED150.pt',map_location='cpu',weights_only=True);tasks=pkg['tasks'];assert len(tasks)==150
sources={t['edit_id']:[q['question'] for q in pkg['source_rows'][str(t['order'])]] for t in tasks};assert len(sources)==150 and all(len(v)==5 for v in sources.values())
plan=json.loads((med/'EVAL_PLAN_FROZEN.json').read_text());bank=[x for x in plan['plan'] if x['mode']=='bank' and x['prefix']==150];assert len(bank)==2148
choices=[]
for row in bank:
 query=row['query'];qid=query['query_id'];v=json.loads((modal/'routes'/(module_run.c.digest(qid)+'.json')).read_text());original=v['original_route']['logical_edit_id'];candidate=v['selected']
 chosen=scope['select'](original,candidate,query['question'],sources[candidate] if candidate is not None else [])
 assert chosen==original or chosen==candidate
 choices.append(dict(query_id=qid,original=original,modal=candidate,selected=chosen,use_modal=chosen!=original,original_route=v['original_route']))
assert len(choices)==len({x['query_id'] for x in choices})==2148
print(json.dumps(dict(choices=choices,source_question_count=750,answer_files_read=0,score_files_read=0)))
'''
    route=remote(code);assert len(route['choices'])==2148 and route['answer_files_read']==route['score_files_read']==0
    write(OUT/'ROUTES_FROZEN.private.json',dict(time=time.time(),**route))
    # Only after freeze, audit each prior actual answer's complete input and state.
    code='choices='+repr(route['choices'])+'\n'+'''from pathlib import Path
import json,sys,os,signal,torch
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU remote bound')));signal.alarm(1200)
q=Path('/data/bmw/.q.20261027.150');os.environ.update(json.loads((q/'ENV.json').read_text()));r=Path(os.environ['EXPERIMENT_ROOT']);p=Path(os.environ['PARENT_RUN']);sys.path[:0]=[str(r/'tools'),str(p/'private/tools'),str(p/'private/source'),str(p/'private/official_llava')]
import module_run
med=Path('/data/bmw/Knowledge_editing/outputs/med150-v1-20261027/run');modal=Path('/data/bmw/Knowledge_editing/outputs/med150-modal-v1-20261028/run');pkg=torch.load(med/'retained/MED150.pt',map_location='cpu',weights_only=True);hashes=pkg['state_hashes']
for x in choices:
 qid=x['query_id'];c1=json.dumps(['MED','bank',150,0,qid],separators=(',',':'));c2=json.dumps(['MED_MODAL','bank',150,0,qid],separators=(',',':'))
 a=json.loads((med/'eval_outputs'/(module_run.c.digest(c1)+'.json')).read_text());b=json.loads((modal/'eval_outputs'/(module_run.c.digest(c2)+'.json')).read_text())
 assert a['binding']['input']==b['binding']['input'] and a['binding']['judge_input']==b['binding']['judge_input']
 assert a['effective_expert']==x['original'] and b['effective_expert']==x['modal']
 for v in (a,b):
  key=v['effective_expert'];assert v['effective_state_hash']==(hashes[key] if key is not None else 'UNEDITED_BASE')
  assert v['R0']['raw_token_ids']
print(json.dumps(dict(queries=len(choices),input_and_state_pairs_verified=len(choices),same_generation_rng_in_prior_protocol=True,actual_scores_recomputed=0)))
'''
    audit=remote(code);assert audit['input_and_state_pairs_verified']==2148
    write(OUT/'IDENTITY_AUDIT.json',audit)
    before=read('/private/tmp/u.20261027.150/PACKET_PRIVATE.json')['consumers'];modal=read('/private/tmp/v.20261028.2/PACKET_PRIVATE.json')['consumers']
    a={v['query_id']:v for v in before if v['mode']=='bank' and v['prefix']==150};b={v['query_id']:v for v in modal}
    assert len(a)==len(b)==2148 and set(a)==set(b)=={x['query_id'] for x in route['choices']}
    deca=read('/private/tmp/u.20261027.150/DECISIONS.json');decb=read('/private/tmp/v.20261028.2/DECISIONS.json');assert not deca['unknown'] and not decb['unknown']
    decisions=deca['decisions'].copy()
    for k,v in decb['decisions'].items():assert k not in decisions or decisions[k]==v;decisions[k]=v
    consumers=[]
    for x in route['choices']:
        v=dict(b[x['query_id']] if x['use_modal'] else a[x['query_id']]);assert v['effective_expert']==x['selected'] and v['key'] in decisions
        v['route']=dict(original_route=x['original_route'],selected=x['selected'],exact_switch=True,used_modal=x['use_modal'])
        consumers.append(v)
    assert len(consumers)==2148 and len({v['consumer_id'] for v in consumers})==2148
    write(OUT/'PACKET_PRIVATE.json',dict(consumers=consumers))
    write(OUT/'DECISIONS.json',dict(decisions={v['key']:decisions[v['key']] for v in consumers},unknown=[]))
    unique=len({v['key'] for v in consumers});write(OUT/'COMPLETE.json',dict(time=time.time(),consumers=2148,unique_payloads=unique,reused=unique,new_attempt_items=0,unknown=0,generation_and_scoring_complete=True,CPU_only=True))
    write(OUT/'GUARD_EXIT.json',dict(time=time.time(),code=0,clock=clock));print(json.dumps(dict(status='COMPLETE',queries=2148,model_calls=0,new_scoring=0,identity_pairs_verified=2148)))


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():
            p=OUT/'FAILURE.json'
            if not p.exists():write(p,dict(time=time.time(),error=traceback.format_exc(),no_retry=True))
        raise
