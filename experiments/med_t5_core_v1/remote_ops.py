"""Data- and count-bound launch admission for T5 Base and selected MDU consumers."""
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False)


def devices(cfg,maximum):
    # Preserve GPUs registered to the still-running T4 chain, even during a transient release.
    done=Path(cfg['t4_root'])/'GUARD_EXIT.json'
    reserved=set() if done.exists() and read(done)['code']==0 else {'2','3'}
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    available=[]
    for line in rows.splitlines():
        index,uuid,free=(x.strip() for x in line.split(','))
        if index not in reserved and int(free)>=45056:available.append(dict(index=index,uuid=uuid,free_MiB=int(free)))
    available=sorted(available,key=lambda x:(-x['free_MiB'],int(x['index'])))[:maximum]
    assert available,'No admitted GPU with required memory reserve'
    return available


def gpu_env(base,gpu):
    return dict(base,CUDA_VISIBLE_DEVICES=gpu['index'],M3BENCH_FORMAL_EXPECTED_GPU_UUID=gpu['uuid'],
        M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=gpu['index'],M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=gpu['index'])


def launch(cfg,env,entry,log):
    with Path(log).open('x') as stream:
        child=subprocess.Popen([cfg['python'],str(entry)],env=dict(os.environ,**env),cwd=cfg['workdir'],
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    return child.pid


def base(root):
    cfg=read(root/'CONFIG.private.json');base=read(cfg['environment']);neutral=Path(cfg['neutral'])
    base.update(TMPDIR=str(neutral/'tmp'),CUDA_CACHE_PATH=str(neutral/'cuda'),TORCH_EXTENSIONS_DIR=str(neutral/'torch'),
        VLLM_RPC_BASE_PATH=str(neutral/'rpc'),T5_REMOTE_ROOT=str(root),T5_NEUTRAL_ROOT=str(neutral),
        T5_TOOLS=str(root/'tools'),MED150_TOOLS=cfg['med150_tools'])
    env=dict(base,CURRENT_SOURCE=str(root/'tools/preprocess.py'))
    result=subprocess.run([cfg['python'],str(neutral/'c.py')],env=dict(os.environ,**env),capture_output=True,text=True,timeout=1200)
    write(root/'PREPROCESS_EXIT.json',dict(code=result.returncode,stderr=result.stderr[-3000:]));assert result.returncode==0,result.stderr[-3000:]
    import sys
    sys.path.insert(0,str(root/'tools'));import projection,stream_extract as s
    with (Path(base['PARENT_RUN'])/'private/EVAL_BINDINGS.json').open('rb') as f:
        assert s.nonspace(f)==b'{';assert s.nonspace(f)==b'"';s.string(f,True);assert s.nonspace(f)==b':'
        configuration=projection.project(f,s.nonspace(f),{'runtime':True,'generation':True})
    write(root/'BASE_CONFIGURATION.private.json',configuration)
    gpus=devices(cfg,2);workers=[]
    for shard,gpu in enumerate(gpus):
        env=gpu_env(base,gpu);env.update(SHARD=str(shard),BASE_SHARDS=str(len(gpus)),CURRENT_SOURCE=str(root/'tools/base.py'))
        workers.append(dict(entry=str(neutral/'b.py'),env=env))
    write(root/'BASE_PIPELINE_CONFIG.private.json',dict(gpus=gpus,workers=workers))
    pid=launch(cfg,dict(base,CURRENT_SOURCE=str(root/'tools/base_guard.py')),neutral/'g.py',root/'base_guard.log')
    write(root/'BASE_START.private.json',dict(pid=pid,gpus=gpus,time=time.time()))
    return dict(pid=pid,gpus=gpus,source_admission=read(root/'SOURCE_ADMISSION.json'))


def train(root):
    cfg=read(root/'CONFIG.private.json');neutral=Path(cfg['neutral']);clock=read(root/'CLOCK.json')
    mask=read(root/'BASE_MASK_FROZEN.json');events=[e for e in mask['events'] if e['status']=='BASE_ELIGIBLE']
    wanted={e['owner'] for e in events};source=read(root/'SOURCE_INPUTS.private.json')
    tasks=[t for t in source['tasks'] if t['order'] in wanted]
    assert tasks and len(tasks)==len(wanted)<=322
    entries=[]
    for t in tasks:
        rows=[dict(t['native'],query_id=f"edit_{t['order']}_{i}",question=q,audit_role='NATIVE' if i==0 else 'FIT') for i,q in enumerate([t['native']['question']]+t['fit_questions'])]
        entries.append(dict(task=t,source_rows=rows))
    write(root/'TRAIN_SOURCE_BOUND_PRIVATE.json',entries)
    write(root/'ROLE_ADMISSION.json',dict(status='TRAINING_SOURCE_ROLE_ADMITTED',owners=[t['order'] for t in tasks]))
    write(root/'SPAN_LOCK.private.json',dict(records=[]))
    base=read(cfg['environment']);base.update(TMPDIR=str(neutral/'tmp'),CUDA_CACHE_PATH=str(neutral/'cuda'),
        TORCH_EXTENSIONS_DIR=str(neutral/'torch'),VLLM_RPC_BASE_PATH=str(neutral/'rpc'),T5_REMOTE_ROOT=str(root),
        T5_NEUTRAL_ROOT=str(neutral),T5_TOOLS=str(root/'tools'),MED150_TOOLS=cfg['med150_tools'])
    cpu=dict(base,SOURCE_ROOT=str(root),OUTPUT_ROOT=str(root),OFFICIAL_SOURCE=base['M3BENCH_LLAVA_SOURCE'],
        TOKENIZER=base['M3BENCH_MODEL_PATH'],DEADLINE=str(clock['hard_end']),CURRENT_SOURCE=str(root/'tools/coverage.py'))
    result=subprocess.run([cfg['python'],str(neutral/'c.py')],env=dict(os.environ,**cpu),text=True,capture_output=True,timeout=600)
    write(root/'TOKEN_EXIT.json',dict(code=result.returncode,stderr=result.stderr[-2000:]));assert result.returncode==0,result.stderr[-2000:]
    tokens=read(root/'TOKEN_LOCK.private.json');admitted={x['order'] for x in tokens['records'] if x['status']=='TOKEN_BOUND'}
    entries=[e for e in entries if e['task']['order'] in admitted];assert entries
    tasks=[e['task'] for e in entries];by_owner={t['order']:t for t in tasks}
    base_outputs=[read(p) for p in (root/'base_outputs').glob('*.json')]
    expected={v['owner']:v for v in base_outputs if v['role']=='T0'}
    expectations={v['query_id']:{k:v['judge_input'][k] for k in ('question','reference','image_sha256','prompt_ids','attention_mask')} for v in base_outputs}
    write(root/'BASE_EXPECTATIONS.private.json',expectations)
    plan=[]
    for entry in entries:
        t=entry['task'];owner=t['order']
        plan.append(dict(edit_id=t['edit_id'],role='T0',query=t['native']))
        plan.extend(dict(edit_id=t['edit_id'],role='FIT',query=q) for q in entry['source_rows'][1:])
        follow=next(x['query'] for x in source['base_plan'] if x['owner']==owner and x['role']=='T5')
        plan.append(dict(edit_id=t['edit_id'],role='T5',query=follow))
    bank_edits={t['edit_id'] for t in tasks if t['sequential_role_safe']}
    total=len(plan)+sum(p['edit_id'] in bank_edits for p in plan)
    assert total<=4000
    gpus=devices(cfg,min(4,len(tasks)));workers=[];shardmap={}
    binding=cfg['source_commit']+':T5_FIXED_C_MED_MDU'
    for index,gpu in enumerate(gpus):
        shard=root/f'shard{index}';shard.mkdir();(shard/'tools').symlink_to(root/'tools',target_is_directory=True)
        chosen=entries[index::len(gpus)];owners=[e['task']['order'] for e in chosen]
        for owner in owners:shardmap[str(owner)]=shard.name
        write(shard/'CLOCK.json',clock)
        write(shard/'ADDITIONAL_INPUTS.json',dict(status='T5_SOURCE_ADMITTED',candidate_binding=binding,tasks=chosen))
        write(shard/'EXPECTED_SOURCE_BINDINGS.json',[dict(expected[e['task']['order']]['judge_input'],query_id=e['task']['native']['query_id']) for e in chosen])
        write(shard/'TOKEN_LOCK.private.json',dict(records=[x for x in tokens['records'] if x['order'] in owners]))
        write(shard/'TRAIN_LOCK.json',dict(physical_GPU=int(gpu['index']),owners=owners,candidate_binding=binding,
            source_commit=cfg['source_commit'],budget_binding=binding,budgets=dict(forwards=20000000,
            non_generation_forwards=20000000,backwards=1200*len(chosen),operator_backwards=160*len(chosen),
            updates=480*len(chosen),generation_attempts=0)))
        env=gpu_env(base,gpu);env.update(EXPERIMENT_ROOT=str(shard),CURRENT_SOURCE=str(root/'tools/initialize.py'))
        workers.append(dict(entry=str(neutral/'w.py'),env=env))
    write(root/'TASK_INPUTS.private.json',dict(tasks=tasks,plan=plan,owner_shards=shardmap,events=events,expected_consumers=total))
    evaluate=gpu_env(base,gpus[0]);evaluate.update(CURRENT_SOURCE=str(root/'tools/bank.py'))
    write(root/'PIPELINE_CONFIG.private.json',dict(gpus=gpus,train_workers=workers,eval_worker=dict(entry=str(neutral/'e.py'),env=evaluate)))
    admission=dict(status='PASS',source_Base_eligible_cases=len(wanted),admitted_cases=len(tasks),
        token_unavailable=len(wanted)-len(tasks),sequential_cases=sum(t['sequential_role_safe'] for t in tasks),
        total_consumers=total,updates_max=480*len(tasks),source_commit=cfg['source_commit'])
    write(root/'CPU_ADMISSION.json',admission)
    pid=launch(cfg,dict(base,T5_NEUTRAL_ROOT=str(neutral),CURRENT_SOURCE=str(root/'tools/pipeline_guard.py')),
        neutral/'g.py',root/'guard.log')
    write(root/'START.private.json',dict(pid=pid,gpus=gpus,time=time.time()))
    return dict(pid=pid,gpus=gpus,admission=admission)
