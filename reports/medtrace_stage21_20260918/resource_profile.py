"""Read-only CPU profiling of retained artifacts; no model/GPU initialization."""
import json,os,time,statistics
from pathlib import Path
import torch


def read(p):return json.loads(p.read_text())

def summarize(values):
    v=[x for x in values if x is not None]
    return dict(n=len(v),mean=statistics.mean(v) if v else None,min=min(v) if v else None,max=max(v) if v else None)

def tensors(x):
    if isinstance(x,torch.Tensor):return x.numel()*x.element_size()
    if isinstance(x,dict):return sum(tensors(v) for v in x.values())
    if isinstance(x,(tuple,list)):return sum(tensors(v) for v in x)
    return 0

def run(old,current):
    profiles={}
    for arm,root,pattern in [('Stage20_FACT_H',old,'private/edits/e*/C_FACT/latest.pt'),('Stage20_BE',old,'private/BE/e*.pt'),('NO_H_HSIC_NEW',current,'private/edits/e*/C_NO_H/latest.pt'),('NO_H_HSIC_REUSED',old/'private/ablation','private/edits/e*/C_NO_H/latest.pt')]:
        rows=[];be_bytes=None
        for p in sorted(root.glob(pattern)):
            if arm=='Stage20_BE':
                # Full BE files are ~224 MiB. Sample one tensor layout; record each file's actual storage size.
                if be_bytes is None:
                    start=time.perf_counter();d=torch.load(p,map_location='cpu',weights_only=False);cpu_seconds=time.perf_counter()-start;be_bytes=tensors(d['wrapper']);del d
                rows.append(dict(position=int(p.stem[1:]),file_bytes=p.stat().st_size,writer_tensor_bytes=be_bytes,tensor_measurement='one compatible BE artifact layout; all artifact file sizes measured',training_seconds=None,CPU_sample_load_seconds=cpu_seconds if not rows else None))
                continue
            receipt=p.parent/'TRAINING.json'
            if not receipt.exists():continue
            t=read(receipt);start=time.perf_counter();d=torch.load(p,map_location='cpu',weights_only=True);cpu_seconds=time.perf_counter()-start
            if d['step']!=320:continue
            rows.append(dict(position=int(p.parent.parent.name[1:]),file_bytes=p.stat().st_size,writer_tensor_bytes=tensors(d['expert']),training_seconds=t['session_seconds'],CPU_sample_load_seconds=cpu_seconds,optimizer_and_resume_in_file=True));del d
        profiles[arm]=dict(rows=rows,complete_edits=len(rows),mean_file_bytes=statistics.mean(r['file_bytes'] for r in rows) if rows else None,mean_writer_tensor_bytes=statistics.mean(r['writer_tensor_bytes'] for r in rows) if rows else None,recorded_training_seconds=sum(r['training_seconds'] or 0 for r in rows) if any(r['training_seconds'] is not None for r in rows) else None,bytes_at_prefix={str(n):dict(file_bytes=sum(r['file_bytes'] for r in rows if r['position']<=n),writer_tensor_bytes=sum(r['writer_tensor_bytes'] for r in rows if r['position']<=n)) if {r['position'] for r in rows if r['position']<=n}==set(range(1,n+1)) else None for n in (11,19,32,45)})
    combined=[dict(r,reused=True,new_stage21_training_seconds=0) for r in profiles['NO_H_HSIC_REUSED']['rows']]+[dict(r,reused=False,new_stage21_training_seconds=r['training_seconds']) for r in profiles['NO_H_HSIC_NEW']['rows']]
    assert len({r['position'] for r in combined})==len(combined)
    profiles['NO_H_HSIC_LOGICAL_BANK']=dict(rows=sorted(combined,key=lambda r:r['position']),complete_edits=len(combined),new_stage21_training_seconds=sum(r['new_stage21_training_seconds'] for r in combined),mean_file_bytes=statistics.mean(r['file_bytes'] for r in combined),mean_writer_tensor_bytes=statistics.mean(r['writer_tensor_bytes'] for r in combined),bytes_at_prefix={str(n):dict(file_bytes=sum(r['file_bytes'] for r in combined if r['position']<=n),writer_tensor_bytes=sum(r['writer_tensor_bytes'] for r in combined if r['position']<=n)) if {r['position'] for r in combined if r['position']<=n}==set(range(1,n+1)) else None for n in (11,19,32,45)})
    timing={}
    for label,root in [('Stage20',old),('Stage21_A',current)]:
        text=(root/'private/OUTPUTS.jsonl').read_text();lines=text.splitlines()
        if not text.endswith('\n'):lines=lines[:-1]
        records=[json.loads(x) for x in lines];base={r['query_id']:r['Base'] for r in records}
        timing[label]=dict(Base_generation_seconds=summarize([r.get('seconds') for r in base.values()]),active_expert_generation_seconds=summarize([r['output'].get('seconds') for r in records if r['route']['activated']]),routing_with_cold_key_cache_seconds=summarize([r.get('routing_seconds') for r in records]),checkpoint_CPU_load_and_GPU_transfer_seconds=summarize([r.get('checkpoint_load_seconds') for r in records]),historical_observations_not_matched_latency_benchmark=True)
        generated=root/'public/GENERATED.json';timing[label]['peak_gpu_allocated_bytes']=read(generated).get('peak_gpu_memory_bytes') if generated.exists() else None
    a=profiles['Stage20_FACT_H'];b=profiles['Stage20_BE']
    return dict(status='PARTIAL_WHILE_A_RUNNING',profiles=profiles,timing=timing,BE_to_FACT_file_storage_ratio=b['mean_file_bytes']/a['mean_file_bytes'],BE_to_FACT_writer_tensor_ratio=b['mean_writer_tensor_bytes']/a['mean_writer_tensor_bytes'],file_size_caveat='FACT latest.pt includes optimizer, RNG and training curve; writer-only tensor size is a different quantity. CPU sampling time is not GPU checkpoint activation time.',storage_advantage_is_not_compute_advantage=True,Stage20_read_only=True)


if __name__=='__main__':
    cfg=read(Path(os.environ['JOB_CONFIG']));value=run(Path(cfg['stage20_run']),Path(cfg['run']));path=Path(cfg['run'])/'public/RESOURCE_PROFILE.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
