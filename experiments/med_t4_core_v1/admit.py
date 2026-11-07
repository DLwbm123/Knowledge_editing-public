"""Materialize only CORE-qualified sources, bind tokens, launch guarded consumers."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    cfg=json.loads(Path(os.environ['T4_CONFIG']).read_text());w=Path(cfg['worktree']);local=Path(cfg['root'])
    assert json.loads((local/'GUARD_EXIT.json').read_text())['code']==0
    masks=json.loads((local/'BASE_MASK_FROZEN.json').read_text())
    selected=[e for e in masks['events'] if e['status']=='BASE_ELIGIBLE']
    native_ids=sorted({e['edit_query_id'] for e in selected})
    assert len(selected)==len(native_ids) and len(native_ids)<=137
    sys.path.insert(0,str(w/'experiments/med_t4_core_v1'))
    from build_sources import build
    sources=build(w)
    for name in ['projection.py','pipeline_guard.py','bank.py']:
        sources[name]=(w/'experiments/med_t4_core_v1'/name).read_text()
    sources['stream_extract.py']=(w/'experiments/med150_source_review_v1/stream_extract.py').read_text()
    # The remote projector may only deserialize the locked native/FIT and probe IDs.
    all_bindings=json.loads((local/'BASE_BINDINGS.private.json').read_text())
    selected_bindings={b['query_id']:b for b in all_bindings['bindings'].values() if b['query_id'] in native_ids}
    payload=dict(events=selected,native_ids=native_ids,expected=selected_bindings,
        configuration=all_bindings['configuration'],clock=json.loads(Path(cfg['clock']).read_text()),
        sources=sources,paths=cfg['remote_paths'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    primitives=sources['stream_extract.py'].split('def selfcheck():')[0]
    projector=sources['projection.py'].replace('import stream_extract as s','')
    command='import types\ns=types.ModuleType("stream_extract")\nexec('+repr(primitives)+',s.__dict__)\n'+projector+'\nPAYLOAD='+repr(payload)+'\n'+r'''
from pathlib import Path
import os,subprocess,time,re,collections
root=Path(PAYLOAD['paths']['root'])
neutral=Path(PAYLOAD['paths']['neutral'])
assert not root.exists() and not neutral.exists() and time.time()<PAYLOAD['clock']['hard_end']-660
root.parent.mkdir(exist_ok=True);fs=os.statvfs(root.parent);assert fs.f_bavail*fs.f_frsize>8*1024**3
root.mkdir();neutral.mkdir();probe=root/'.probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink()
(root/'tools').mkdir()
def write(name,value):
 p=root/name;p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x') as f:json.dump(value,f,ensure_ascii=False)
for name,source in PAYLOAD['sources'].items():(root/'tools'/name).write_text(source)
legacy=Path(PAYLOAD['paths']['legacy'])/'COHORT_AND_SUPPORT_LEDGER.json'
qfields={k:True for k in ('query_id','question','reference','dataset','image_path','original_image_path','image_sha256','source_group')}
fields={'order':True,'edit_id':True,'seed':True,'native':qfields,'fit_questions':True,'fit_status':True}
probes={q for e in PAYLOAD['events'] for q in e['eligible_probe_query_ids']}
original=read(legacy,{'tasks':[{'__select__':('edit_id',PAYLOAD['native_ids']),'fields':fields}],
 'queries':{q:qfields for q in probes}})
assert {t['edit_id'] for t in original['tasks']}==set(PAYLOAD['native_ids'])
def norm(x):return re.sub(r'\s+',' ',''.join(c if c.isalnum() else ' ' for c in x.casefold())).strip()
evaluation_text={norm(q['question']) for q in original['queries'].values()}
entries=[];excluded=[]
for t in original['tasks']:
 b=PAYLOAD['expected'][t['edit_id']]
 assert t['native']['query_id']==t['edit_id'] and all(t['native'][k]==b[k] for k in ('question','reference','image_sha256'))
 assert t['fit_status']=='SUPPORTED' and len(t['fit_questions'])==4
 if any(norm(x) in evaluation_text for x in t['fit_questions']):
  excluded.append({'edit_id':t['edit_id'],'reason':'EXACT_FIT_EVALUATION_TEXT_COLLISION'});continue
 rows=[dict(t['native'],query_id=f"edit_{t['order']}_{i}",question=question,audit_role='NATIVE' if i==0 else 'FIT') for i,question in enumerate([t['native']['question']]+t['fit_questions'])]
 assert len(rows)==5
 entries.append({'task':t,'source_rows':rows})
assert entries
write('CLOCK.json',PAYLOAD['clock']);write('SOURCE_EXCLUSIONS.private.json',excluded)
write('TRAIN_SOURCE_BOUND_PRIVATE.json',entries)
write('ROLE_ADMISSION.json',{'status':'TRAINING_SOURCE_ROLE_ADMITTED','owners':[e['task']['order'] for e in entries]})
write('SPAN_LOCK.private.json',{'records':[]})
write('BASE_CONFIGURATION.private.json',PAYLOAD['configuration'])
base_env=json.loads(Path(PAYLOAD['paths']['environment']).read_text())
for name in ['tmp','cuda','torch','rpc']:(neutral/name).mkdir()
base_env.update(TMPDIR=str(neutral/'tmp'),CUDA_CACHE_PATH=str(neutral/'cuda'),TORCH_EXTENSIONS_DIR=str(neutral/'torch'),VLLM_RPC_BASE_PATH=str(neutral/'rpc'))
cpu_env=dict(os.environ,**base_env,SOURCE_ROOT=str(root),OUTPUT_ROOT=str(root),OFFICIAL_SOURCE=base_env['M3BENCH_LLAVA_SOURCE'],TOKENIZER=base_env['M3BENCH_MODEL_PATH'],DEADLINE=str(PAYLOAD['clock']['hard_end']),CURRENT_SOURCE=str(root/'tools/coverage.py'))
for name in ['n.py','w.py','e.py','g.py']:(neutral/name).write_text("import os,runpy\nrunpy.run_path(os.environ['CURRENT_SOURCE'],run_name='__main__')\n")
p=subprocess.run(['/data/bmw/envs/v0/bin/python',str(neutral/'n.py')],env=cpu_env,text=True,capture_output=True,timeout=600)
write('TOKEN_WORKER_EXIT.json',{'code':p.returncode,'stderr':p.stderr[-2000:]})
assert p.returncode==0,p.stderr[-2000:]
tokens=json.loads((root/'TOKEN_LOCK.private.json').read_text());bound={x['order'] for x in tokens['records'] if x['status']=='TOKEN_BOUND'}
unavailable=[e['task']['edit_id'] for e in entries if e['task']['order'] not in bound]
entries=[e for e in entries if e['task']['order'] in bound];assert entries
tasks=[e['task'] for e in entries];admitted={t['edit_id'] for t in tasks}
events=[e for e in PAYLOAD['events'] if e['edit_query_id'] in admitted]
plan=[]
for entry in entries:
 t=entry['task']
 plan.append({'edit_id':t['edit_id'],'role':'T0','query':dict(t['native'],audit_role='NATIVE')})
 plan += [{'edit_id':t['edit_id'],'role':'FIT','query':q} for q in entry['source_rows'][1:]]
 for event in events:
  if event['edit_query_id']==t['edit_id']:
   plan += [{'edit_id':t['edit_id'],'role':event['task'],'query':dict(original['queries'][qid],audit_role=event['task'])} for qid in event['eligible_probe_query_ids']]
assert 2*len(plan)<=3000
gpus=[]
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
for line in raw.splitlines():
 index,uuid,free=(x.strip() for x in line.split(','))
 if int(free)>=45056:gpus.append({'index':index,'uuid':uuid,'free_MiB':int(free)})
gpus=sorted(gpus,key=lambda x:(-x['free_MiB'],int(x['index'])))[:min(2,len(entries))];assert gpus
workers=[];shardmap={};candidate_binding=PAYLOAD['source_commit']+':T4_CORE_SOURCE_V1'
parent=Path(base_env['PARENT_RUN'])
for index,gpu in enumerate(gpus):
 shard=root/f'shard{index}';shard.mkdir();(shard/'tools').symlink_to(root/'tools',target_is_directory=True)
 chosen=entries[index::len(gpus)];owners=[e['task']['order'] for e in chosen]
 for owner in owners:shardmap[str(owner)]=shard.name
 def put(name,value):
  with (shard/name).open('x') as f:json.dump(value,f,ensure_ascii=False)
 put('CLOCK.json',PAYLOAD['clock'])
 put('ADDITIONAL_INPUTS.json',{'status':'T4_SOURCE_ADMITTED','candidate_binding':candidate_binding,'tasks':chosen})
 put('EXPECTED_SOURCE_BINDINGS.json',[dict(PAYLOAD['expected'][e['task']['edit_id']],query_id=e['task']['edit_id']) for e in chosen])
 put('TOKEN_LOCK.private.json',{'records':[x for x in tokens['records'] if x['order'] in owners]})
 put('TRAIN_LOCK.json',{'physical_GPU':int(gpu['index']),'owners':owners,'candidate_binding':candidate_binding,'source_commit':PAYLOAD['source_commit'],'budget_binding':candidate_binding,'budgets':{'forwards':20000000,'non_generation_forwards':20000000,'backwards':len(chosen)*1200,'operator_backwards':len(chosen)*160,'updates':len(chosen)*480,'generation_attempts':0}})
 env=dict(base_env,EXPERIMENT_ROOT=str(shard),MED150_TOOLS=PAYLOAD['paths']['med150_tools'],CUDA_VISIBLE_DEVICES=gpu['index'],M3BENCH_FORMAL_EXPECTED_GPU_UUID=gpu['uuid'],M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=gpu['index'],M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=gpu['index'],CURRENT_SOURCE=str(root/'tools/initialize.py'))
 workers.append({'entry':str(neutral/'w.py'),'env':env})
write('TASK_INPUTS.private.json',{'tasks':tasks,'plan':plan,'owner_shards':shardmap,'events':events})
gpu=gpus[0];eval_env=dict(workers[0]['env'],T4_REMOTE_ROOT=str(root),CURRENT_SOURCE=str(root/'tools/bank.py'))
write('PIPELINE_CONFIG.private.json',{'gpus':gpus,'train_workers':workers,'eval_worker':{'entry':str(neutral/'e.py'),'env':eval_env}})
admission={'status':'PASS','Base_eligible_edits':len(PAYLOAD['events']),'source_exclusions':len(excluded),'token_unavailable':len(unavailable),'admitted_edits':len(entries),'tasks':dict(collections.Counter(e['task'] for e in events)),'single_consumers':len(plan),'total_consumers':2*len(plan),'updates_max':480*len(entries),'new_bytes_limit':4000000000,'snapshot_bytes_before_launch':sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink()),'no_QA_from_unselected_sources':True,'source_commit':PAYLOAD['source_commit']}
write('CPU_ADMISSION.json',admission)
env=dict(os.environ,**base_env,T4_REMOTE_ROOT=str(root),T4_NEUTRAL_ROOT=str(neutral),CURRENT_SOURCE=str(root/'tools/pipeline_guard.py'))
with (root/'guard.log').open('x') as log:
 child=subprocess.Popen(['/data/bmw/envs/v0/bin/python',str(neutral/'g.py')],env=env,stdout=log,stderr=subprocess.STDOUT,cwd='/data/bmw',start_new_session=True)
write('START.private.json',{'guard_pid':child.pid,'time':time.time(),'clock':PAYLOAD['clock'],'gpus':gpus})
print(json.dumps({'admission':admission,'guard_pid':child.pid,'gpus':gpus,'root':str(root),'neutral':str(neutral)}))
'''
    result=subprocess.run(['ssh','-o','BatchMode=yes',cfg['ssh_host'],cfg['remote_python'],'-'],input=command,
        text=True,capture_output=True,timeout=720)
    path=Path(cfg['private_root'])/'REMOTE_ADMISSION_RETURN.json'
    path.write_text(json.dumps(dict(code=result.returncode,stdout=result.stdout,stderr=result.stderr[-4000:]),indent=2))
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    status=json.loads(result.stdout)
    (Path(cfg['report'])/'TRAINING_ADMISSION.json').write_text(json.dumps(status['admission'],indent=2)+'\n')
    print(json.dumps(status))


if __name__=='__main__':
    main()
