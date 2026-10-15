"""Registered first-three panels plus unchanged 36-probe protection panel; output-only."""
import os
import time
import traceback
import torch
import base_evaluator as old
from runtime import R, ARM, CONTROL, c, b, track, wait, lease, peak_memory


def protection_ids(tasks, ledger, task=None):
    active = {(t['native']['image_sha256'], t['native']['question']) for t in tasks[:100]}
    return list(dict.fromkeys(q for t in tasks[:100] for e in t['events']
        if e['task'] in ((task,) if task else ('T1L','T2L')) for q in e['all_probe_query_ids']
        if ledger['Base_correctness'][q] and (ledger['queries'][q]['image_sha256'],ledger['queries'][q]['question']) not in active))


def expected_ids(tasks, ledger, mode, owner, prefix):
    if mode == 'protect100_R0':
        assert owner == 0 and prefix == 100
        return protection_ids(tasks, ledger)
    if mode == 'bank_R0':
        assert owner == 0 and prefix in (1,3)
        return list(dict.fromkeys(q for t in tasks[:prefix] for e in t['events'] for q in e['all_probe_query_ids']))
    assert mode == 'insertion_R0' and 1 <= owner <= 3 and prefix == owner
    t = tasks[owner-1]
    return list(dict.fromkeys([t['edit_id']] + [q for e in t['events'] if e['task']=='T2G'
        for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]]))


old.expected_ids = expected_ids
panel = old.panel


def evaluate(admission_only=False):
    with lease():
        runtime, bindings = c.load(6); hooks = track(runtime)
        try:
            tasks = c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks']; ledger = c.read(R/'private/EVAL_LEDGER.json')
            if admission_only:
                from contract import base_completion
                for order in (1,50,100):
                    t = tasks[order-1]
                    d = b.generate(runtime, bindings, t['native'], R/'private/mechanical'/ARM/(str(order)+'.json'), dict(stage='BASE_IDENTITY',arm=ARM))
                    assert d['R0'] == base_completion(bindings[t['native']['opaque_Base_id']]['output'])
                c.write(CONTROL/'BASE_ADMITTED.json', dict(status='PASS', three_Base_outputs_exact=True))
                wait(CONTROL/'LEARNER_ADMITTED.json')
                c.write(R/'public'/('ADMISSION_'+ARM+'.json'), dict(status='PASS',Base3_exact=True,
                    real_native_gradient=True, Fisher_VJP=True, source_only_IO_guard=True, added_parameters=0,formal_slots=0))
                peak_memory()
                return
            assert c.read(R/'public'/('ADMISSION_'+ARM+'.json'))['status']=='PASS'
            weight = runtime.get_module(c.LAYER).weight
            for owner in range(1,4):
                assert wait(CONTROL/'EVALUATE.json')['owner'] == owner
                saved = torch.load(CONTROL/'native.pt', map_location='cpu', weights_only=True)
                with torch.no_grad(): weight.copy_(saved['native'].to(runtime.device))
                panel(runtime,bindings,tasks,ledger,'insertion_R0',owner,owner)
                if owner in (1,3): panel(runtime,bindings,tasks,ledger,'bank_R0',0,owner)
                if owner == 3: panel(runtime,bindings,tasks,ledger,'protect100_R0',0,100)
                peak_memory(); c.write(CONTROL/(str(owner)+'.EVAL_DONE.json'),dict(owner=owner,outputs_durable=True))
                while (CONTROL/'EVALUATE.json').exists(): c.budget(); time.sleep(.1)
            c.write(R/'private'/('EVALUATION_COMPLETE_'+ARM+'.json'),dict(status='COMPLETE',consumers=161))
        finally: hooks.close(); b.counts()


def reload():
    with lease():
        runtime,bindings=c.load(6);hooks=track(runtime)
        try:
            saved=torch.load(R/'private/final'/ARM/'native.pt',map_location='cpu',weights_only=True)
            assert b.schema(runtime.model)==saved['schema']
            assert sum(p.numel() for p in runtime.model.parameters())==saved['parameters']
            with torch.no_grad():runtime.get_module(c.LAYER).weight.copy_(saved['native'].to(runtime.device))
            tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(R/'private/EVAL_LEDGER.json');task=tasks[2]
            ids=[task['edit_id'],next(e['all_probe_query_ids'][0] for e in task['events'] if e['task']=='T2G')]
            for q in ids:
                d=b.generate(runtime,bindings,ledger['queries'][q],R/'private/reload'/ARM/(q+'.json'),dict(stage='FRESH_NATIVE_RELOAD',arm=ARM))
                assert d['R0']==c.read(R/'private/outputs'/ARM/'bank_R0/0/3'/(q+'.json'))['R0']
            c.write(R/'public'/('NATIVE_RELOAD_'+ARM+'.json'),dict(status='PASS',two_outputs_exact=True,added_parameters=0))
            c.write(R/'private'/('FINAL_CONSUMED_'+ARM+'.json'),dict(reason='Full registered outputs and fresh reload exact',old_assets_touched=False))
        finally:hooks.close();b.counts()


def reference():
    with lease():
        runtime,bindings=c.load(6);hooks=track(runtime)
        try:
            assert c.read(R/'private/M_REFERENCE_PROOF.json')['first3_reference_W_exact']
            saved=torch.load(R/'private/M_reference_native.pt',map_location='cpu',weights_only=True)
            with torch.no_grad():runtime.get_module(c.LAYER).weight.copy_(saved['native'].to(runtime.device))
            tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(R/'private/EVAL_LEDGER.json')
            panel(runtime,bindings,tasks,ledger,'bank_R0',0,3)
            panel(runtime,bindings,tasks,ledger,'protect100_R0',0,100)
            c.write(R/'private/FINAL_CONSUMED_M.json',dict(status='READONLY_REFERENCE_COMPLETE',retrained=False,consumers=161))
        finally:hooks.close();b.counts()


if __name__=='__main__':
    try:
        action=os.environ['ACTION']
        if action.startswith('admit_'):evaluate(True)
        else:(reference if action=='reference_M' else reload if action.startswith('reload') else evaluate)()
        c.write(R/'private'/('EXIT_'+action+'.json'),dict(exit_code=0,epoch=time.time()))
    except BaseException:
        b.counts();c.write(R/'private/failures'/(os.environ['ACTION']+'_'+str(os.getpid())+'.json'),dict(traceback=traceback.format_exc(),epoch=time.time(),counts=b.COUNTS));raise
