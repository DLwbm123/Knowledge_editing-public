"""Finish the authorized retrospective TT88 benchmark using the original ledger.

The eight-edit LoRA comparison is separate; no LoRA146 claim is made here.
All stages share the existing resource/Judge ledger and original wall deadline.
"""
import os
import time
import traceback
from pathlib import Path

import core

common, worker = core.common, core.worker
RUN, BASE = core.RUN, core.BASE
PREFIXES = (1, 50, 100, 146)
ARM = 'TT88_W0_RETRO146'


def query_ids(tasks):
    return {q for t in tasks for e in t['events'] for q in e['all_probe_query_ids']}


def consumer_bound(tasks):
    return (sum(len(query_ids([t])) for t in tasks)
            + sum(len(query_ids(tasks[:n])) for n in PREFIXES)
            + sum(len(query_ids([dict(t, events=[e for e in t['events'] if e['task'] == 'T0'])])) for t in tasks))


def prepare():
    import torch
    common.budget()
    assert (RUN/'private/CONTROLLER_COMPLETE.json').exists()
    assert not any(not s.get('ended_epoch') for s in common.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions'])
    parent = Path(common.read(BASE/'PLAN_CONFIG.json')['parent_run'])
    ledger = common.read(parent/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json')
    bindings = common.read(parent/'private/legacy_stage17/BINDINGS.json')
    byid = {t['edit_id']: t for t in ledger['tasks']}
    original = [byid[e] for e in ledger['main_T0']]
    assert len(original) == 146 and len({t['edit_id'] for t in original}) == 146
    old = {t['edit_id']: t for t in common.read(BASE/'private/QUEUES.json')['tasks'] if t['cohort'] == 'P2'}
    tasks, reused = [], {}
    for i, t in enumerate(original, 1):
        x = {k: t[k] for k in ('edit_id', 'order', 'native', 'fit_questions', 'seed')}
        x.update(cohort='RETRO146', anonymous_edit='R146_E'+str(i).zfill(3),
                 events=[e for e in t['events'] if e['task'] in ('T0', 'T1G', 'T2G', 'T1L', 'T2L')])
        assert len(x['fit_questions']) == 4 and Path(common.local_path(x['native']['image_path'])).is_file()
        if t['edit_id'] in old:
            prior = old[t['edit_id']]
            assert all(prior[k] == x[k] for k in ('native', 'fit_questions', 'seed', 'events', 'order'))
            p = BASE/'private/edits'/prior['anonymous_edit']/'warmup/W0.pt'
            s = torch.load(p, map_location='cpu', weights_only=True)
            assert s['step'] == 320 and s['binding']['phase'] == 'W0'
            assert set(s['expert']) == {'G1', 'G2', 'G3', 'G4'} and sum(v.numel() for v in s['expert'].values()) == 7168
            assert s['binding']['task']['native'] == x['native'] and s['binding']['task']['fit_questions'] == x['fit_questions']
            assert s['binding']['task']['seed'] == x['seed'] and not s['binding']['task']['U_fit']
            for phase, steps in [('native', 140), ('A2', 80), ('W0', 320)]:
                tr = common.read(p.parent/(phase+'_TRAINING.json'))
                assert tr['status'] == 'COMPLETE' and tr['steps'] == steps
            reused[t['edit_id']] = str(p)
        tasks.append(x)
    assert len(reused) == 24
    queries = {q: ledger['queries'][q] for q in query_ids(tasks)}
    bs = {q['opaque_Base_id']: bindings[q['opaque_Base_id']] for q in queries.values()}
    reference = next(iter(common.read(RUN/'private/EVAL_BINDINGS.json').values()))
    for q in queries.values():
        b = bs[q['opaque_Base_id']]
        assert Path(common.local_path(q['image_path'])).is_file()
        assert b['runtime'] == reference['runtime'] and b['generation'] == reference['generation']
        assert (q['question'], q['reference'], q['image_sha256']) == (b['question'], b['reference'], b['image_sha256'])
        assert type(ledger['Base_correctness'][q['query_id']]) is bool
    bound = consumer_bound(tasks)
    attempts = common.read(RUN/'RESOURCE_LEDGER.json')['Judge_attempts']
    assert bound + attempts <= common.read(RUN/'RUN_MANIFEST.json')['Judge_limit']
    for name, incoming in [('EVAL_LEDGER.json', dict(queries=queries, Base_correctness={q: ledger['Base_correctness'][q] for q in queries})), ('EVAL_BINDINGS.json', bs)]:
        existing = common.read(RUN/'private'/name)
        if name == 'EVAL_LEDGER.json':
            for field, values in incoming.items():
                assert all(existing[field][k] == v for k, v in values.items() if k in existing[field])
                existing[field].update(values)
        else:
            assert all(existing[k] == v for k, v in incoming.items() if k in existing)
            existing.update(incoming)
        common.write(RUN/'private'/name, existing)
    common.write(RUN/'private/BENCHMARK146_QUEUE.json', dict(tasks=tasks, reused=reused, prefixes=PREFIXES))
    common.write(RUN/'public/BENCHMARK146_ADMISSION.json', dict(status='PASS', retrospective=True,
        original_order=True, edits=146, TT88_W0_reused=24, TT88_W0_to_train=122,
        original_queries=len(queries), consumer_upper_bound=bound, Judge_attempts_before=attempts,
        Judge_worst_case_total=attempts+bound, runtime_generation_masks_inputs_checked=True,
        prefixes=PREFIXES, independent_confirmation=False, LoRA_matched_scale=8,
        epoch=time.time(), original_wall_deadline_retained=True))


def queue():
    return common.read(RUN/'private/BENCHMARK146_QUEUE.json')


def point(t):
    return Path(queue()['reused'].get(t['edit_id'], str(worker.w0_path(t, 0))))


def train_single():
    part, gpu = int(os.environ['PARTITION']), int(os.environ['GPU'])
    q = queue()
    with common.lease(gpu):
        runtime, bindings = common.load(gpu)
        for t in q['tasks'][part::2]:
            done = RUN/'private/edits'/t['anonymous_edit']/'RETRO_SINGLE_COMPLETE.json'
            if done.exists():
                continue
            if t['edit_id'] not in q['reused']:
                worker.warmup(runtime, t)
            bank = worker.router(runtime, t)
            worker.evaluate(runtime, bindings, t, 0, ARM, 0, {t['edit_id']: point(t)}, bank, [t])
            common.write(done, dict(status='GENERATED_NOT_SCORED', epoch=time.time()))
            # Only this run's consumed intermediate phases; final W0 remains for the bank.
            if t['edit_id'] not in q['reused']:
                deleted = []
                for phase in ('native', 'A2'):
                    p = worker.w0_path(t, 0).parent/(phase+'.pt')
                    assert not p.is_symlink()
                    if p.exists():
                        deleted.append(dict(path=str(p), bytes=p.stat().st_size));p.unlink()
                common.write(done.parent/'INTERMEDIATE_DELETION.json', dict(files=deleted,
                    reason='Warmup and single complete; bank consumes retained W0 only'))
    common.write(RUN/'private'/('BENCHMARK146_SINGLE_'+str(part)+'.json'), dict(status='COMPLETE', epoch=time.time()))


def banks():
    gpu = int(os.environ['GPU']);tasks = queue()['tasks']
    assert all((RUN/'private'/('BENCHMARK146_SINGLE_'+str(p)+'.json')).exists() for p in (0, 1))
    with common.lease(gpu):
        runtime, bindings = common.load(gpu);bank=[];points={}
        for i,t in enumerate(tasks,1):
            bank += worker.router(runtime,t);points[t['edit_id']] = point(t)
            native = dict(t, events=[e for e in t['events'] if e['task']=='T0'])
            worker.evaluate(runtime,bindings,t,0,ARM,0,points,bank,[native],mode='insertion',prefix=i)
            if i in PREFIXES:
                worker.evaluate(runtime,bindings,t,0,ARM,0,points,bank,tasks[:i],mode='bank',prefix=i)
                common.write(RUN/'private'/('BENCHMARK146_PREFIX_'+str(i)+'.json'),dict(status='GENERATED_NOT_SCORED',epoch=time.time()))


def report():
    r = core.tool('report');records,scores=r.load();tasks=queue()['tasks'];panels=[]
    for mode,prefix in [('single_R0',1)]+[('bank_R0',n) for n in PREFIXES]:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            ts = tasks[:prefix] if mode=='bank_R0' else tasks
            m,c,g=r.panel(records,scores,'RETRO146',ARM,0,mode,task,ts,prefix)
            m.update(r.bootstrap(c,scores,g));panels.append(m)
    ledger=common.read(RUN/'RESOURCE_LEDGER.json')
    common.write(RUN/'public/BENCHMARK146_RESULTS.json',dict(panels=panels,retrospective=True,
        single_edits=146,sequential_prefixes=PREFIXES,parameters=7168,selected_version='W0_R0',
        independent_confirmation=False,medical_locality='NA_SCOPE_NOT_QUALIFIED',
        LoRA_matched_scale=8,full_system_comparison='NOT_IDENTITY_QUALIFIED',
        GPU_hours=common.used()/3600,Judge_attempts=ledger['Judge_attempts']))


def controller():
    import pipeline
    assert common.read(RUN/'public/BENCHMARK146_ADMISSION.json')['status']=='PASS'
    common.write(RUN/'public/PROGRESS.json',dict(status='TT88_RETROSPECTIVE146_RUNNING',
        core_training_complete=True,core_evaluation_complete=True,LoRA8_complete=True,
        benchmark146_complete=False,whole_task_complete=False))
    pipeline.wait([pipeline.launch('benchmark146.py','retro_single',5,0),pipeline.launch('benchmark146.py','retro_single',6,1)])
    pipeline.wait([pipeline.launch('benchmark146.py','retro_bank',7,0)])
    pipeline.score()
    pipeline.wait([pipeline.launch('benchmark146.py','retro_report')])
    common.write(RUN/'private/BENCHMARK146_COMPLETE.json',dict(epoch=time.time(),status='RESULTS_COMPLETE_REVIEW_PUBLICATION_PENDING'))
    common.write(RUN/'public/PROGRESS.json',dict(status='RESULTS_COMPLETE_REVIEW_PUBLICATION_PENDING',
        training_complete=True,evaluation_complete=True,development_candidate_locked=True,
        independent_confirmation_complete=False,benchmark146_complete=True,publication_complete=False,whole_task_complete=False))


if __name__ == '__main__':
    try:
        dict(retro_prepare=prepare,retro_single=train_single,retro_bank=banks,
             retro_report=report,retro_controller=controller)[os.environ['ACTION']]()
    except BaseException as error:
        common.write(RUN/'private'/('FAILURE_'+os.environ['ACTION']+'_'+os.environ.get('PARTITION','all')+'.json'),
                     dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),retry=False))
        raise
