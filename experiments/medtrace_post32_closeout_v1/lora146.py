"""Complete the frozen normalized-LoRA reference on the original 146 edits."""
import os
from pathlib import Path
import subprocess
import time
import traceback

import benchmark146 as retro
import lora

common, worker, RUN = retro.common, retro.worker, retro.RUN
ARM = 'LORA_R1_W0_RETRO146'


def tasks():
    return retro.queue()['tasks']


def point(t):
    return RUN/'private/lora146'/t['anonymous_edit']/'W0.pt'


def outputs(ts):
    for i, t in enumerate(ts, 1):
        groups = [('single', 1, retro.query_ids([t])),
                  ('insertion', i, retro.query_ids([dict(t, events=[e for e in t['events'] if e['task']=='T0'])]))]
        if i in retro.PREFIXES:
            groups.append(('bank', i, retro.query_ids(ts[:i])))
        for mode, prefix, ids in groups:
            folder = RUN/'private/outputs'/mode/t['anonymous_edit']/'s0'/ARM/'n0'/('p'+str(prefix))/'R0'
            for qid in sorted(ids):
                yield folder/(common.digest(qid)+'.json'), mode, prefix, t, qid


def configure():
    worker.clone = lora.clone
    original = worker.write
    def write(path, data):
        if 'TT_parameters_per_expert' in data:
            data.pop('TT_parameters_per_expert')
            data.update(expert_parameters_per_expert=18432, representation='NORMALIZED_LORA_R1')
        original(path, data)
    worker.write = write


def prepare():
    import torch
    common.budget()
    manifest = common.read(RUN/'RUN_MANIFEST.json')
    assert manifest.get('Judge_limit_enabled') is False
    assert common.read(RUN/'private/judge_astra_medium_recovery1/REPORT_COMPLETE.json')['status']=='COMPLETE_NO_MISSING_AND_PUBLISHED'
    ts = tasks()
    assert len(ts)==146 and len({t['edit_id'] for t in ts})==146
    for t in ts:
        assert len(t['fit_questions'])==4 and Path(common.local_path(t['native']['image_path'])).is_file()
        assert retro.point(t).is_file()
        router = torch.load(RUN/'private/edits'/t['anonymous_edit']/'ROUTER.pt', map_location='cpu', weights_only=True)
        assert len(router['entries'])==1 and router['entries'][0]['logical_edit_id']==t['edit_id']
    lock = dict(arm=ARM, tasks=common.digest(ts), parameters=18432, rank=1, normalization='input RMS',
                warmup=[140,80,320], U_supervision=False, prefixes=list(retro.PREFIXES),
                queue='BENCHMARK146_QUEUE.json', runtime=common.digest(common.read(RUN/'private/EVAL_BINDINGS.json')),
                source=common.read(RUN/'private/LORA146_SOURCE.json'), deadline_epoch=manifest['deadline_epoch'])
    p = RUN/'private/LORA146_LOCK.json'
    if p.exists():
        assert common.read(p)==lock
    else:
        common.write(p, lock)
    assert len(list(outputs(ts)))==retro.consumer_bound(ts)==5722
    common.write(RUN/'private/LORA146_CONSUMERS.json', [dict(path=str(p),mode=mode+'_R0',prefix=n,
        edit_order=t['order'],query_id=qid,owner=t['anonymous_edit']) for p,mode,n,t,qid in outputs(ts)])
    # Initial small-scale weights were already consumed and deleted; reconstruct once.
    common.write(RUN/'public/LORA146_ADMISSION.json', dict(status='PASS', edits=146, new_experts=146,
        old_LoRA8_weights_reused=0, TT_outputs_and_scores_reused=True, output_consumers=5722,
        single_consumers=sum(len(retro.query_ids([t])) for t in ts), final_bank_queries=len(retro.query_ids(ts)),
        prefixes=list(retro.PREFIXES), parameters=18432, TT_parameters=7168,
        Judge_limit_enabled=False, deadline_epoch=manifest['deadline_epoch'], independent_confirmation=False,
        primary_endpoint='TT minus LoRA T2G edit macro at prefix146', epoch=time.time()))


def train_single():
    import torch
    configure()
    part, gpu = int(os.environ['PARTITION']), int(os.environ['GPU'])
    assert part in (0,1) and gpu in (5,6)
    with common.lease(gpu):
        runtime, bindings = common.load(gpu)
        for t in tasks()[part::2]:
            done = point(t).parent/'SINGLE_COMPLETE.json'
            if done.exists():
                assert common.read(done)['lock']==common.digest(common.read(RUN/'private/LORA146_LOCK.json'))
                continue
            p = lora.warm(runtime,t,point(t).parent)
            state = torch.load(p,map_location='cpu',weights_only=True)
            assert state['binding']['task']==t and state['binding']['parameters']==18432
            assert set(state['expert'])=={'A','B'} and sum(v.numel() for v in state['expert'].values())==18432
            worker.evaluate(runtime,bindings,t,0,ARM,0,{t['edit_id']:p},worker.router(runtime,t),[t])
            common.write(done,dict(status='GENERATED_NOT_SCORED',lock=common.digest(common.read(RUN/'private/LORA146_LOCK.json')),epoch=time.time()))
            print('SINGLE_COMPLETE',t['order'],flush=True)
    common.write(RUN/'private'/('LORA146_SINGLE_'+str(part)+'.json'),dict(status='COMPLETE',epoch=time.time()))


def banks():
    configure()
    part, gpu = int(os.environ['PARTITION']), int(os.environ['GPU'])
    assert gpu==4+part and all((RUN/'private'/('LORA146_SINGLE_'+str(p)+'.json')).exists() for p in (0,1))
    def available(g):
        common.budget()
        uuid, free = subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
        assert uuid==common.read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(g)]
        return int(free)>=24000
    common.available = available
    ts = tasks()
    with common.lease(gpu):
        runtime, bindings = common.load(gpu)
        first = ts[0]
        native = dict(first,events=[e for e in first['events'] if e['task']=='T0'])
        probe = RUN/'private'/('LORA146_PROBE_'+str(part))
        (probe/'private/bank_bindings').mkdir(parents=True,exist_ok=True)
        for name in ('EVAL_LEDGER.json','GPU_SOURCE_VERSION.json'):
            p = probe/'private'/name
            if not p.exists(): p.symlink_to(RUN/'private'/name)
        oldroot = worker.RUN
        worker.RUN = probe
        try:
            worker.evaluate(runtime,bindings,first,0,ARM,0,{first['edit_id']:point(first)},worker.router(runtime,first),[native],mode='bank',prefix=1)
        finally:
            worker.RUN = oldroot
        qid = next(iter(retro.query_ids([native])))
        suffix = Path(first['anonymous_edit'])/'s0'/ARM/'n0/p1/R0'/(common.digest(qid)+'.json')
        ref = common.read(RUN/'private/outputs/single'/suffix)
        actual = common.read(probe/'private/outputs/bank'/suffix)
        assert all(actual[k]==ref[k] for k in ('R0','route','effective_expert','weight'))
        common.write(RUN/'private'/('LORA146_NATIVE_'+str(part)+'.json'),dict(status='PASS',exact_tokens_routes=True,epoch=time.time()))
        bank, points = [], {}
        for i,t in enumerate(ts,1):
            bank += worker.router(runtime,t)
            points[t['edit_id']] = point(t)
            native = dict(t,events=[e for e in t['events'] if e['task']=='T0'])
            panel = retro.shard_tasks([native],part)
            if retro.query_ids(panel):
                worker.evaluate(runtime,bindings,t,0,ARM,0,points,bank,panel,mode='insertion',prefix=i)
            if i in retro.PREFIXES:
                worker.evaluate(runtime,bindings,t,0,ARM,0,points,bank,retro.shard_tasks(ts[:i],part),mode='bank',prefix=i)
                common.write(RUN/'private'/('LORA146_PREFIX_'+str(i)+'_'+str(part)+'.json'),dict(status='GENERATED_NOT_SCORED',epoch=time.time()))
    common.write(RUN/'private'/('LORA146_BANK_'+str(part)+'.json'),dict(status='COMPLETE',epoch=time.time()))


def controller():
    import pipeline
    assert common.read(RUN/'public/LORA146_ADMISSION.json')['status']=='PASS'
    common.write(RUN/'public/LORA146_PROGRESS.json',dict(status='TRAINING_AND_SINGLE',whole_comparison_complete=False))
    pipeline.wait([pipeline.launch('lora146.py','lora146_single',5,0),pipeline.launch('lora146.py','lora146_single',6,1)])
    common.write(RUN/'public/LORA146_PROGRESS.json',dict(status='FOUR_GPU_BANK_GENERATION',whole_comparison_complete=False))
    pipeline.wait([pipeline.launch('lora146.py','lora146_bank',4+p,p) for p in range(4)])
    entries = list(outputs(tasks()))
    assert len(entries)==5722 and all(p.is_file() for p,*_ in entries)
    common.write(RUN/'private/LORA146_GENERATION_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',consumers=len(entries),epoch=time.time()))
    common.write(RUN/'public/LORA146_PROGRESS.json',dict(status='GENERATION_COMPLETE_ASTRA_PENDING',whole_comparison_complete=False))
    pipeline.wait([pipeline.launch('lora146_queue.py','lora146_ingest')])
    # All GPU consumers and complete score-input bindings are now durable.
    removed = []
    for t in tasks():
        p = point(t)
        assert not p.is_symlink() and p.parent.resolve().parent== (RUN/'private/lora146').resolve()
        removed.append(dict(path=str(p),bytes=p.stat().st_size))
        p.unlink()
    common.write(RUN/'private/LORA146_DELETION.json',dict(files=removed,backup=False,recovery='Retraining required',epoch=time.time()))
    common.write(RUN/'public/LORA146_LIFECYCLE.json',dict(weights_deleted=len(removed),bytes_deleted=sum(x['bytes'] for x in removed),
        all_GPU_consumers_complete=True,score_inputs_and_bindings_preserved=True,backup=False,recovery_requires_retraining=True))
    finish()


def finish():
    """Resume only the score/report tail after reviewed worker recovery."""
    import pipeline
    assert (RUN/'private/LORA146_GENERATION_COMPLETE.json').exists()
    assert (RUN/'private/LORA146_DELETION.json').exists()
    root = RUN/'private/judge_lora146_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        common.budget()
        failures = list((root/'workers').glob('*/SCORER_FAILURE.json'))
        assert not failures, 'Scorer needs engineering review; never retry attempted payloads automatically'
        time.sleep(30)
    pipeline.wait([pipeline.launch('lora146_report.py','lora146_report')])


if __name__=='__main__':
    try:
        {'lora146_prepare':prepare,'lora146_single':train_single,'lora146_bank':banks,'lora146_controller':controller,'lora146_finish':finish}[os.environ['ACTION']]()
    except BaseException as error:
        common.write(RUN/'private'/('FAILURE_'+os.environ['ACTION']+'_'+os.environ.get('PARTITION','all')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),retry=False))
        raise
