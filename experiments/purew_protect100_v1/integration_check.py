"""CPU fixture for actual update/state code and real historical queue bindings; no model/Judge request."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import faulthandler
faulthandler.dump_traceback_later(45,repeat=True)

payload=json.loads(sys.stdin.readline())
P=Path(payload['protocol']['reference_run'])
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PROTECT_CPU_TRACE='1')
with tempfile.TemporaryDirectory(prefix='e.',dir='/data/bmw/tmp') as folder:
    R=Path(folder);(R/'private/tools').mkdir(parents=True);(R/'public').mkdir()
    for path in (P/'private/tools').glob('*.py'):shutil.copy2(path,R/'private/tools'/path.name)
    for name,source in payload['files'].items():
        compile(source,name,'exec');(R/'private/tools'/name).write_text(source)
    (R/'private/source').symlink_to((P/'private/source').resolve())
    for name in ('BENCHMARK146_QUEUE.json','EVAL_LEDGER.json'):(R/'private'/name).symlink_to(P/'private'/name)
    os.environ.update(RUN_ROOT=str(R),ACTION='cpu_fixture');sys.path.insert(0,str(R/'private/tools'))
    import common as c
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print('CPU_STAGE math',flush=True)
    from protection_math import selfcheck as math_check
    from logic import selfcheck as logic_check,resume_check
    result=math_check();result.update(logic_check())
    c.write(R/'private/FROZEN_PROTOCOL.json',payload['protocol'])
    c.write(R/'private/GPU_SOURCE_VERSION.json',dict(commit='CPU_FIXTURE_ONLY'))
    c.write(R/'RUN_MANIFEST.json',dict(deadline_epoch=time.time()+3600,GPU_seconds_limit=86400,Judge_limit=2181,
           min_free_bytes=1,owned_weight_limit_bytes=2147483648))
    c.write(R/'STAGE_CAP.json',dict(GPU_seconds_limit=86400))
    c.write(R/'RESOURCE_LEDGER.json',dict(Judge_attempts=0,Astra_purew146_batches=[],gpu_sessions=[],gpu_seconds_used=0,weights_observed_peak_bytes=0))
    import numpy as np
    import random
    from types import SimpleNamespace
    import run as executor
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.randn(7,3).half())
        def forward(self,inputs_embeds,**kwargs):
            return SimpleNamespace(logits=torch.nn.functional.linear(inputs_embeds,self.weight))
    torch.manual_seed(31);model=Toy();runtime=SimpleNamespace(model=model,device=torch.device('cpu'))
    batch=(dict(inputs_embeds=torch.randn(1,3,3).half()),torch.tensor([[-100,2,4]]))
    with torch.no_grad():teacher=(batch,executor.b.logits(runtime,batch)[0].clone())
    initial=model.weight.detach().float().clone();edit_initial=initial.clone()
    torch.manual_seed(22);random.seed(22);np.random.seed(22)
    initial_rng=c.rng();initial_numpy=np.random.get_state()
    def advance(master,start,end):
        event=None
        for step in range(start,end+1):
            random.random();np.random.random();torch.rand(2)
            master,event=executor.optimize(runtime,model.weight,master,[(batch,.5),(batch,.5)],teacher,.002,True)
            event.update(owner=1,step=step,mode='bank_R0')
        return master,event
    uninterrupted,_=advance(initial.clone(),1,8);full_native=model.weight.detach().clone()
    c.restore_rng(initial_rng);np.random.set_state(initial_numpy)
    with torch.no_grad():model.weight.copy_(initial.half())
    partial,event=advance(initial.clone(),1,4)
    saved=executor.save_state(R/'private/active/bank_R0.pt',partial,model.weight,1,4,4,event,edit_initial)
    restored=torch.load(R/'private/active/bank_R0.pt',map_location='cpu',weights_only=True);resume_check(restored,{})
    assert torch.equal(restored['native'],restored['master'].half())
    c.restore_rng(restored['rng']);nr=restored['numpy_rng'];np.random.set_state((nr[0],nr[1].numpy().astype('uint32'),nr[2],nr[3],nr[4]))
    with torch.no_grad():model.weight.copy_(restored['native'])
    resumed,_=advance(restored['master'],5,8)
    assert torch.equal(uninterrupted,resumed) and torch.equal(full_native,model.weight)
    result.update(actual_executor_toy_update=True,exact_executor_checkpoint_resume=True,source_only_batches=True)
    print('CPU_STAGE historical_binding',flush=True)
    import queue_protect as queue
    q=queue.q;q.ROOT.mkdir(parents=True,exist_ok=True)
    q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch='CPU_FIXTURE_ONLY',model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT))
    db=q.connect();queue.ingest(db)
    before=db.execute('SELECT count(*) FROM payload').fetchone()[0]
    old=queue.ro(P/'private/judge_purew146/queue.sqlite')
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(R/'private/EVAL_LEDGER.json')
    lookup={(x['mode'],x['prefix'],x['edit_order'],x['query_id']):x['path'] for x in old.execute("SELECT * FROM consumer WHERE method='PUREW_H146'")}
    for mode,owner,prefix in [('bank_R0',0,n) for n in (1,50,100)]+[('insertion_R0',n,n) for n in range(1,101)]:
        phase=dict(arm='K',mode=mode,owner=owner,prefix=prefix,protocol=c.digest(payload['protocol']),code='CPU_FIXTURE_ONLY')
        ids=queue.expected_ids(tasks,ledger,phase);dest=R/'private/outputs'/mode/str(owner)/str(prefix)
        for qid in ids:
            key=(mode,prefix,owner,qid)
            source=lookup.get(key) or lookup['bank_R0',100,0,qid]
            d=c.read(source);d['binding']['phase']=phase;c.write(dest/(qid+'.json'),d)
        c.write(dest/'COMPLETE.json',dict(phase=phase,outputs=len(ids)))
    queue.ingest(db)
    assert db.execute("SELECT count(*) FROM consumer WHERE method='K'").fetchone()[0]==0
    assert not queue.terminal_allowed(db) and queue.status(db)['status']=='GENERATING'
    c.write(R/'private/FINAL_CONSUMED.json',dict(test_only=True));c.write(R/'private/GENERATION_COMPLETE.json',dict(test_only=True))
    queue.ingest(db);queue.ingest(db)
    assert db.execute("SELECT count(*) FROM consumer WHERE method='K'").fetchone()[0]==2181
    assert db.execute('SELECT count(*) FROM payload').fetchone()[0]==before and queue.terminal_allowed(db)
    assert db.execute("SELECT count(*) FROM payload WHERE status='MISSING'").fetchone()[0]==11
    print('CPU_STAGE report_fixture',flush=True)
    # The report fixture uses the fixed norm metadata, not model-generated scientific events.
    prep=c.read(P.parent.parent/'purew-protect100-v1-20261013/preparation/QUALIFICATION_PRIVATE.json')['targets']
    for owner in range(1,101):
        for step in range(1,161):
            k=(owner-1)*160+step-1
            c.write(R/'private/steps/bank_R0'/str(owner)/(str(step)+'.json'),dict(owner=owner,step=step,
                proposal_norm=prep['proposal_norms'][k],target_proposal_norm=prep['proposal_norms'][k],
                actual_native_norm=prep['native_step_norms'][k],C_native_norm=prep['native_step_norms'][k],
                changed_elements=1,BASIS_KL=0.,native_master_discrepancy=0.,edit_native_net_displacement=.01,
                diagnostics=dict(source_gradient_norm=1.,history_gradient_norm=0.,KL_gradient_norm=0.,
                    source_history_cosine=None,source_KL_cosine=None,history_KL_cosine=None) if step in (1,40,80,160) else None))
    c.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(test_only=True));c.write(R/'public/NATIVE_RELOAD.json',dict(test_only=True))
    import protect_report
    protect_report.main()
    final=c.read(R/'public/RESULTS.json')
    assert final['status']=='COMPLETE_WITH_MISSING' and not final['scientific_success']
    assert not final['gates']['protection_zero_Base_damage'] and not final['gates']['full_required_scoring']
    contrast=next(x for x in final['contrasts'] if x['comparison']=='K-minus-C-prefix100' and x['task']=='T2G')
    assert contrast['delta_macro_bounds_pp']==[0.,0.]
    primary=next(x for x in final['panels'] if x['arm']=='K' and x['mode']=='bank_R0' and x['prefix']==100 and x['task']=='T2G')
    assert primary['observations']==386
    result.update(historical_binding_reuse=True,old_missing_preserved=11,new_consumers_checked=2181,
        no_evaluation_scoring_before_generation=True,idempotent_panels=True,terminal_barrier=True,
        shared_missing_exact_cancellation=True,full_denominator_report=True,unknown_protection_not_passed=True,
        GPU_updates=0,Judge_requests=0,real_VLM=False,full_execution_recovery_checked=True)
    print(json.dumps(result))
