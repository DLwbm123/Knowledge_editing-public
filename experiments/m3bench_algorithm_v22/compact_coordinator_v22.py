"""Evidence-gated handoff after the existing OFF stage; never repeat a Base call."""
import json,os,sys,time,subprocess,traceback,shutil
from pathlib import Path
ROOT=Path(os.environ['V22_ROOT'])
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def main():
    with (ROOT/'COMPACT_COORDINATOR_ONCE').open('x'):pass
    start=read(ROOT/'START.json');jobs=start['jobs']
    # Every old worker writes BASE_COMPLETE before consulting its training admission.
    while True:
        complete=all((ROOT/str(i)/'BASE_COMPLETE.json').exists() for i in range(5))
        exited=all((ROOT/str(i)/'FORMAL_EXIT.json').exists() for i in range(5))
        live=False
        for j in jobs:
            stat=Path('/proc')/str(j['pid'])/'stat'
            live=live or (stat.exists() and stat.read_text().split()[21]==str(j['start_ticks']))
        exited=exited and not live
        write(ROOT/'COMPACT_QUEUE_STATUS.json',dict(stage='WAITING_OFF_COMPLETION_AND_CONTROLLED_EXIT',all_Base_complete=complete,all_original_exited=exited,time=time.time()))
        if exited:
            assert complete,'Unexpected old failure before OFF completed; do not silently rerun'
            break
        time.sleep(10)
    old_cost=[]
    for i,j in enumerate(jobs):
        out=ROOT/str(i);exit=read(out/'FORMAL_EXIT.json');stat=Path('/proc')/str(j['pid'])/'stat'
        assert not (stat.exists() and stat.read_text().split()[21]==str(j['start_ticks'])),'Old process identity still alive'
        assert exit['counts']['backwards']==exit['counts']['updates']==0,'Training already executed; scope handoff invalid'
        assert exit['counts']['generations']==read(out/'BASE_COMPLETE.json')['queries']
        old_cost.append(exit)
    queries=read(ROOT/'QUERY_ROWS.json');assert len(list((ROOT/'base').glob('*/*.json')))==len(queries)==2078
    # Inspect the saved feature shards in a CPU-only process; no model import/load.
    check='import torch,json;from pathlib import Path;r=Path('+repr(str(ROOT))+');sets=[set(torch.load(r/str(i)/"QUERY_KEYS.pt",map_location="cpu",weights_only=True)) for i in range(5)];q=set(json.loads((r/"QUERY_ROWS.json").read_text()));assert sum(map(len,sets))==len(q)==len(set.union(*sets));print("KEY_COVERAGE_PASS")'
    key_check=Path('/data/bmw/tmp/q.20261020.22.k.py');key_check.write_text(check);subprocess.run(['/data/bmw/envs/v0/bin/python',str(key_check)],check=True)
    archive=ROOT/'controlled_scope_handoff1';archive.mkdir(exist_ok=False)
    for i in range(5):
        dest=archive/str(i);dest.mkdir()
        for name in ('FORMAL_FAILURE.json','FORMAL_EXIT.json','FORMAL_MODEL_READY.json','LAUNCH.json'):
            src=ROOT/str(i)/name
            if src.exists():shutil.copyfile(src,dest/name)
    write(ROOT/'CONTROLLED_SCOPE_HANDOFF.json',dict(status='PASS',original_exit_codes=[x['code'] for x in old_cost],reason='User authorized compact10 scope before any training',old_cost=old_cost,Base_answers_reused=2078,new_Base_calls=0,all_old_exact_identities_gone=True,time=time.time()))
    admission=read(ROOT/'COMPACT_CPU_ADMISSION.json');assert admission['status']=='PASS'
    compact=read(ROOT/'COMPACT_DATA_ADMISSION.json');assert len(compact['tasks'])==10 and compact['method_answers']==1512
    write(ROOT/'FULL146_DATA_ADMISSION.json',read(ROOT/'DATA_ADMISSION.json'));write(ROOT/'DATA_ADMISSION.json',compact)
    write(ROOT/'FORMAL_CPU_ADMISSION.json',admission)
    env=read(ROOT/'PARENT_ENV.json');gpus=[0,1,3,4,5];started=time.time();launched=[]
    for part,gpu in enumerate(gpus):
        data=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',');uuid=data[0].strip();assert int(data[1])>=37000
        out=ROOT/str(part);entry=Path(f'/data/bmw/tmp/q.20261020.22.b{part}.py');assert not entry.exists()
        e=dict(env);e.pop('DEADLINE_EPOCH',None);e.update(V22_ROOT=str(ROOT),OUTPUT_RUN=str(out),PARTITION=str(part),GPU=str(part),FORMAL_START=str(started),CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES='0,1,3,4,5',M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuid)
        write(out/'COMPACT_ENV.json',e)
        entry.write_text('import os,json,runpy\nfrom pathlib import Path\np=Path('+repr(str(out/'COMPACT_ENV.json'))+')\nos.environ.update({k:str(v) for k,v in json.loads(p.read_text()).items()})\nrunpy.run_path('+repr(str(ROOT/'tools/worker_compact_v22.py'))+',run_name="__main__")\n')
        log=(out/'compact.log').open('ab');p=subprocess.Popen(['/data/bmw/envs/v0/bin/python','-u',str(entry)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd='/data/bmw/tmp',env=dict(os.environ,**{k:str(v) for k,v in e.items()}));log.close()
        ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21];j=dict(pid=p.pid,start_ticks=ticks,physical_gpu=gpu,partition=part,uuid=uuid,time=time.time());write(out/'COMPACT_LAUNCH.json',j);launched.append(j)
    write(ROOT/'COMPACT_START.json',dict(start=started,deadline=None,jobs=launched,original_counts_preserved=True,scope='FIXED_FIRST10_SEVEN_ARMS',new_Base_calls=0))
    print('COMPACT_LAUNCHED',json.dumps(launched),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        write(ROOT/'COMPACT_COORDINATOR_FAILURE.json',dict(error=traceback.format_exc(),time=time.time(),automatic_retry=False));raise
