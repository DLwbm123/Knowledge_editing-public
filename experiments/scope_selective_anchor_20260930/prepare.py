"""Prepare isolated execution using audited read-only predecessor assets."""
import os,json,shutil,hashlib,time,subprocess
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);OLD=Path(os.environ['PREDECESSOR_ROOT'])
def read(p):return json.loads(p.read_text())
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2))
def main():
    assert read(ROOT/'public/INIT_AND_REUSE_AUDIT.json')['common_initializers']==48
    for name in ['source','source_patch','models']:
        if not (ROOT/name).exists():(ROOT/name).symlink_to((OLD/name).resolve(),target_is_directory=True)
    for name in ['worker_v3.py','router_r3.py','structures.py','evaluation.py','judge_io.py','diagnostics.py','protection.py','mechanisms.py','hard_pool.py','CONFIG_LOCK.json','SUPPORT_REPAIR.json']:
        shutil.copy2(OLD/name,ROOT/name)
    for name in ['TASKS_R2_LOCKED.json','G_SUPPORTS.json','BASE_MASKS.json','CHECK_POS.json','CHECK_NEG.json','U_bg.json','PILOT_SELECTION.json','LOCALITY_STRESS_HOLDOUT.json']:
        shutil.copy2(OLD/'private'/name,ROOT/'private'/name)
    repair=read(ROOT/'SUPPORT_REPAIR.json')['repaired_path'];(ROOT/repair).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/repair,ROOT/repair)
    for name in ['private/base','private/keys','private/scales','private/hard','private/judge/scores']:
        if not (ROOT/name).exists():shutil.copytree(OLD/name,ROOT/name)
    for name in ['private/judge/pending','private/black','jobs','logs']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    shutil.copy2(OLD/'private/judge/JUDGE_MISSING_LOCK.json',ROOT/'private/judge/JUDGE_MISSING_LOCK.json')
    prior=read(OLD/'RESOURCE_LEDGER.json')
    assert all(s.get('ended_epoch') for s in prior['gpu_sessions'])
    write(ROOT/'RESOURCE_LEDGER.json',dict(historical_judge_attempts=prior['judge_submission_attempt_items'],current_judge_attempts=0,judge_submission_attempt_items=prior['judge_submission_attempt_items'],judge_submission_attempt_items_limit=prior['judge_submission_attempt_items'],judge_attempts=prior['judge_attempts'],physical_requests=prior['physical_requests'],physical_requests_limit=None,historical_gpu_seconds=0,current_gpu_seconds=0,lifetime_gpu_seconds=0,gpu_sessions=[]))
    devices=[]
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    for row in text.strip().splitlines():
        i,u,f=[x.strip() for x in row.split(',')]
        if int(i) in [5,6,7]:devices.append(dict(index=int(i),uuid=u,free_MiB=int(f)))
    assert len(devices)==3 and all(x['free_MiB']>=30000 for x in devices)
    write(ROOT/'GPU_BINDINGS.json',dict(devices=devices,authorization='user current attachment, physical GPU 5/6/7 only'))
    write(ROOT/'STORAGE_POLICY.json',dict(quotas_enabled=True,soft_bytes=10*1024**3,hard_bytes=20*1024**3,min_free_bytes=8*1024**3,checkpoint_bytes=2*1024**3,teacher_bytes=2*1024**3,final_total_bytes=16*1024**3,final_model_bytes=1024**3,control_reserved_bytes=2*1024**3,environment_reserved_bytes=0,emergency_bytes=256*1024**2,local_scorer_reserved_bytes=512*1024**2,external_owned_roots=[]))
    write(ROOT/'PREDECESSOR.json',dict(root=str(OLD),old_readonly=True))
    science=read(ROOT/'public/RUN_PROTOCOL.json');science['parent_science']=read(OLD/'SCIENCE_LOCK.json')['id'];science['id']=hashlib.sha256(json.dumps(science,sort_keys=True).encode()).hexdigest();write(ROOT/'SCIENCE_LOCK.json',science)
    # Target window controls new-block admission, never kills a useful paired block at hour 16.
    write(ROOT/'RUN_MANIFEST.json',dict(started_epoch=time.time(),target_hours=[12,16],no_new_large_block_after_seconds=12*3600))
    write(ROOT/'RUN_STATUS.json',dict(status='READY_FOR_MECHANICAL_SMOKE',Judge='WAITING_FOR_JUDGE_RESERVATION'))
    write(ROOT/'QUEUE.json',[])
    for name in ['pending','scores']:(ROOT/'private/judge_sol'/name).mkdir(parents=True,exist_ok=True)
    if not (ROOT/'private/judge_sol/JUDGE_MISSING_LOCK.json').exists():write(ROOT/'private/judge_sol/JUDGE_MISSING_LOCK.json',dict(keys=[]))
    print(json.dumps(dict(status='READY_FOR_MECHANICAL_SMOKE',GPU=devices,smoke_judge_calls=0)))
if __name__=='__main__':main()
