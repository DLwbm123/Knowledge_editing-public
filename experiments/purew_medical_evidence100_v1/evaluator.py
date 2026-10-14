"""Separate output-only evaluator. No optimizer, gradients or feedback to learner."""
import gc
import json
import math
import os
from pathlib import Path
import time
import traceback
import torch
from runtime import R,ARM,CONTROL,c,b,track,wait,lease,peak_memory


def expected_ids(tasks,ledger,mode,owner,prefix):
    if mode=='bank_R0':
        return list(dict.fromkeys(q for t in tasks[:prefix] for e in t['events'] for q in e['all_probe_query_ids']))
    t=tasks[owner-1]
    return list(dict.fromkeys([t['edit_id']]+[q for e in t['events'] if e['task']=='T2G'
                         for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]]))


def panel(runtime,bindings,tasks,ledger,mode,owner,prefix):
    phase=dict(arm=ARM,mode=mode,owner=owner,prefix=prefix,
        protocol=c.digest(c.read(R/'private/FROZEN_PROTOCOL.json')),code=c.read(R/'private/GPU_SOURCE_VERSION.json')['commit'])
    dest=R/'private/outputs'/ARM/mode/str(owner)/str(prefix)
    ids=expected_ids(tasks,ledger,mode,owner,prefix)
    for q in ids:
        b.generate(runtime,bindings,ledger['queries'][q],dest/(q+'.json'),phase)
    c.write(dest/'COMPLETE.json',dict(phase=phase,outputs=len(ids)))


def diagnostics(runtime,tasks,owner,final=False):
    from dataclasses import replace
    from PIL import Image
    import numpy as np
    rows=c.read(R/'private/operator/source.json')['rows'];reviews=c.read(R/'private/operator/review.json')['rows']
    selected=[i for i in range(1,101) if reviews[i-1]['region_status']=='ELIGIBLE_COARSE_BOX'] if final else [owner]
    for order in selected:
        row=rows[order-1];review=reviews[order-1]
        if review['region_status']!='ELIGIBLE_COARSE_BOX':continue
        task=tasks[order-1];a=np.array(Image.open(row['image_file']).convert('RGB'))
        for i in review['selected_box_indices']:
            x,y,w,h=row['boxes'][i]['xywh'];a[math.floor(y):math.ceil(y+h),math.floor(x):math.ceil(x+w),:]=128
        path=R/'private/evaluation'/ARM/'removed.png';path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(a).save(path)
        removed=dict(task,native=dict(task['native'],image_path=str(path)))
        contrasts=[];norm=0.
        with torch.no_grad():
            for index,q in enumerate([None]+task['fit_questions']):
                z,y=b.logits(runtime,b.batch_cpu(runtime,task,q));zr,yr=b.logits(runtime,b.batch_cpu(runtime,removed,q))
                assert torch.equal(y,yr)
                D=((z-z.mean(-1,keepdim=True))-(zr-zr.mean(-1,keepdim=True)))/math.sqrt(len(y))
                coefficient=.5 if index==0 else .125;norm+=coefficient*float(D.square().sum())
                lp=lambda logits:torch.log_softmax(logits,-1).gather(-1,y[:,None]).mean()
                contrasts.append(float(lp(z)-lp(zr)))
        output=dict(order=order,contrast_logit_norm=math.sqrt(norm),target_meanlogp_contrasts=contrasts,
            scope='TRAINING_SOURCE_ONLY_NOT_CLINICAL_GROUNDING',vector_drift='NOT_STORED; norm magnitude drift only')
        if final:
            prior=c.read(R/'private/source_diagnostics'/ARM/'insertion'/(str(order)+'.json'))
            output.update(norm_magnitude_drift=output['contrast_logit_norm']-prior['contrast_logit_norm'],
                target_contrast_drifts=[x-y for x,y in zip(contrasts,prior['target_meanlogp_contrasts'])])
        c.write(R/'private/source_diagnostics'/ARM/('final' if final else 'insertion')/(str(order)+'.json'),output)
        path.unlink();del a,task,removed,z,zr,D,rows,reviews
        # Reload operator metadata for the next diagnostic without retaining image/feature tensors.
        if final:rows=c.read(R/'private/operator/source.json')['rows'];reviews=c.read(R/'private/operator/review.json')['rows']
    gc.collect();torch.cuda.empty_cache()


def evaluate():
    with lease():
        runtime,bindings=c.load(5);h=track(runtime)
        try:
            tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'][:100];ledger=c.read(R/'private/EVAL_LEDGER.json')
            for order in (1,50,100):
                t=tasks[order-1];d=b.generate(runtime,bindings,t['native'],R/'private/mechanical'/ARM/(str(order)+'.json'),dict(stage='BASE_IDENTITY',arm=ARM))
                from contract import base_completion
                assert d['R0']==base_completion(bindings[t['native']['opaque_Base_id']]['output'])
            c.write(CONTROL/'BASE_ADMITTED.json',dict(status='PASS',three_Base_outputs_exact=True))
            wait(CONTROL/'LEARNER_ADMITTED.json')
            c.write(R/'public'/('ADMISSION_'+ARM+'.json'),dict(status='PASS',Base3_exact=True,learner_native_gradient_and_intervention=True,
                current_case_IO_guard=True,independent_evaluator=True,added_parameters=0,security_boundary='Python I/O enforcement, not adversarial OS sandbox'))
            weight=runtime.get_module(c.LAYER).weight
            assert not any(p.requires_grad for p in runtime.model.parameters())
            for owner in range(1,101):
                req=wait(CONTROL/'EVALUATE.json');assert req['owner']==owner
                saved=torch.load(CONTROL/'native.pt',map_location='cpu',weights_only=True)
                assert set(saved)=={'native'} and saved['native'].dtype==torch.float16
                with torch.no_grad():weight.copy_(saved['native'].to(runtime.device))
                panel(runtime,bindings,tasks,ledger,'insertion_R0',owner,owner)
                if owner in (1,50,100):panel(runtime,bindings,tasks,ledger,'bank_R0',0,owner)
                diagnostics(runtime,tasks,owner)
                if owner==100:diagnostics(runtime,tasks,owner,True)
                peak_memory();c.write(CONTROL/(str(owner)+'.EVAL_DONE.json'),dict(owner=owner,outputs_durable=True))
                while (CONTROL/'EVALUATE.json').exists():time.sleep(.1)
            c.write(R/'private'/('EVALUATION_COMPLETE_'+ARM+'.json'),dict(status='COMPLETE',consumers=2181))
        finally:h.remove();b.counts()


def reload():
    with lease():
        runtime,bindings=c.load(5);h=track(runtime)
        try:
            saved=torch.load(R/'private/final'/ARM/'native.pt',map_location='cpu',weights_only=True)
            assert b.schema(runtime.model)==saved['schema']
            assert sum(p.numel() for p in runtime.model.parameters())==saved['parameters']
            with torch.no_grad():runtime.get_module(c.LAYER).weight.copy_(saved['native'].to(runtime.device))
            task=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'][99];ledger=c.read(R/'private/EVAL_LEDGER.json')
            ids=[task['edit_id'],next(e['all_probe_query_ids'][0] for e in task['events'] if e['task']=='T2G')]
            for q in ids:
                d=b.generate(runtime,bindings,ledger['queries'][q],R/'private/reload'/ARM/(q+'.json'),dict(stage='FRESH_NATIVE_RELOAD',arm=ARM))
                assert d['R0']==c.read(R/'private/outputs'/ARM/'bank_R0/0/100'/(q+'.json'))['R0']
            c.write(R/'public'/('NATIVE_RELOAD_'+ARM+'.json'),dict(status='PASS',two_outputs_exact=True,added_parameters=0,schema_exact=True))
            paths=[R/'private/final'/ARM/'native.pt',R/'private/active'/ARM/'state.pt',CONTROL/'native.pt']
            c.write(R/'private'/('FINAL_CONSUMED_'+ARM+'.json'),dict(reason='All outputs and fresh reload exact',bytes=sum(p.stat().st_size for p in paths if p.exists()),historical_deleted=0))
            for path in paths:
                assert R in path.parents and not path.is_symlink()
                if path.exists():path.unlink()
        finally:h.remove();b.counts()


if __name__=='__main__':
    try:
        (reload if os.environ['ACTION'].startswith('reload') else evaluate)()
        c.write(R/'private'/('EXIT_'+os.environ['ACTION']+'.json'),dict(exit_code=0,epoch=time.time()))
    except BaseException:
        b.counts();c.write(R/'private/failures'/(os.environ['ACTION']+'_'+str(os.getpid())+'.json'),dict(traceback=traceback.format_exc(),epoch=time.time(),counts=b.COUNTS));raise
