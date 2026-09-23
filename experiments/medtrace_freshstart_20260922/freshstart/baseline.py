"""Existing BalanceEdit paper-spec adaptation, with bounded integration and disk-backed experts."""
from dataclasses import asdict,replace
from pathlib import Path
import time
import torch
from freshstart.runtime import ROOT,read,write,check_budget
from freshstart.pipeline import record,input_key,score_request
from scripts.medtrace import stage15
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_cfact import state_hash
from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor


def run(runtime,tasks):
    root=ROOT/'formal-baseline';root.mkdir(exist_ok=True);runtime.run_root=root
    editor=BalanceEditPaperSpecEditor(runtime,inactive_store_dir=root/'private/weights')
    original,target=editor.wrapper.base,editor.target
    initialized=time.time();hashes={}
    def boundary(module,args):
        check_budget(training=True)
        if time.time()-initialized>1800:raise TimeoutError('30 minute baseline integration ceiling')
    handle=runtime.model.register_forward_pre_hook(boundary)
    def observe(task,row,prefix,label):
        rec=replace(record(task),question=row['question'],target=row['reference'],image_path=Path(row['image_path']))
        raw,_,binding=stage15.prepared(runtime,row,rec)
        with editor.route_generation(rec) as route:output=stage15.generate(runtime,raw,binding)
        consumer=digest(['baseline',prefix,label,input_key(row)])
        score=score_request(row,output,consumer)
        if row['role'].startswith('U'):
            base=read(ROOT/'private/preparation/BASE_OUTPUTS.json')[digest([input_key(row),row['reference']])]['output']
            score['teacher_judge_key']=score_request(dict(row,reference=base['raw_answer']),output,consumer+'/teacher','TEACHER_AGREEMENT')['judge_key']
        write(root/'private/consumers'/f'{consumer}.json',dict(arm='BALANCEDIT_PAPER_SPEC',prefix=prefix,event=label,row=row,
            route=asdict(route),parameter_hash=hashes.get(route.logical_edit_id),output=output,**score))
    try:
        evaluation=read(ROOT/'private/FRESH_EVALUATION.json')['candidate_packages']
        for prefix,task in enumerate(tasks,1):
            outcome=editor.apply_edit(record(task))
            if not outcome['finite_losses'] or not outcome['finite_gradients'] or outcome['steps']!=50:
                raise FloatingPointError('Baseline numerical gate failed')
            hashes[task['canonical_edit_id']]=state_hash(editor.wrapper.get_edit(task['canonical_edit_id']))
            write(root/'private/events'/f'{prefix}.json',outcome)
            observe(task,task['native'],prefix,'writer')
            if prefix in {5,19,32,45,len(tasks)}:
                for i,t in enumerate(tasks[:prefix]):
                    for row in [t['native']]+evaluation[i]['evaluation']:observe(t,row,prefix,f'panel/{i}/{row["role"]}')
            editor.save_editor_state(root/'private/active.tmp.pt')
            (root/'private/active.tmp.pt').replace(root/'private/active.pt')
            write(root/'BASELINE_STATUS.json',dict(status='RUNNING',prefix=prefix,N=len(tasks),classification='PAPER_SPEC_ADAPTATION_NOT_AUTHOR_CODE'))
        editor.load_editor_state(root/'private/active.pt')
        write(root/'BASELINE_STATUS.json',dict(status='TRAINING_AND_GENERATION_COMPLETE_SCORING_PENDING',prefix=len(tasks),N=len(tasks),
            classification='PAPER_SPEC_ADAPTATION_NOT_AUTHOR_CODE',support='native training; first existing fit used only for radius; 50 steps; different capacity',reload='PASS'))
    finally:
        handle.remove();editor.reset_editor_state();runtime.replace_module(target,original)
        if not runtime.base_guard.verify()['unchanged']:raise RuntimeError('Baseline changed frozen backbone')
