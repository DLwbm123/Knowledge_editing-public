"""Train four original-recipe C calibrations and MED experts; preserve old146."""
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(os.environ['EXPERIMENT_ROOT']);PARENT=Path(os.environ['PARENT_RUN'])
OLD=Path(os.environ['RAW_MED_ROOT']);CURVATURE=Path(os.environ['CURVATURE_ROOT'])
LOCK=json.loads((ROOT/'TRAIN_LOCK.json').read_text());CLOCK=json.loads((ROOT/'CLOCK.json').read_text())
COUNTS=dict(forwards=0,non_generation_forwards=0,backwards=0,operator_backwards=0,updates=0,generation_attempts=0)


def write(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def save(name,value):
    import torch
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    torch.save(value,tmp);tmp.replace(p)
def budget():
    assert time.time()<CLOCK['hard_end']-30 and not (ROOT/'TRAIN_STOP.json').exists()
    assert time.time()-json.loads((ROOT/'TRAIN_GUARD_HEALTH.json').read_text())['time']<45
    for name,limit in LOCK['budgets'].items():
        if name in COUNTS:assert COUNTS[name]<=limit,(name,COUNTS[name],limit)


def main():
    with (ROOT/'TRAIN_WORKER_ONCE').open('x'):pass
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Guard stop; preserve latest resume')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Original wall clock')))
    signal.alarm(max(1,int(CLOCK['hard_end']-time.time()-30)))
    sys.path[:0]=[str(ROOT/'tools'),str(PARENT/'private/tools'),str(PARENT/'private/source'),str(PARENT/'private/official_llava')]
    import torch
    import module_run as r
    import trajectory
    import training_math
    import med133_curvature as curvature
    from prefix_cache import PrefixCache
    from m3bench_repro.editors.llava_runtime import seed_everything
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor,balanced_radius,record_seed
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    r.c.budget=budget;r.COUNTS=COUNTS
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(LOCK['physical_GPU'])
    torch.cuda.set_per_process_memory_fraction(.78)
    data=json.loads((ROOT/'ADDITIONAL_INPUTS.json').read_text());expected=json.loads((ROOT/'EXPECTED_SOURCE_BINDINGS.json').read_text())
    assert len(data['tasks'])==len(expected)==4 and data['candidate_binding']==LOCK['candidate_binding']
    oldlock=json.loads((OLD/'PROTOCOL_LOCK.json').read_text());assert data['original146_input_binding']==oldlock['input_binding']
    assert json.loads((CURVATURE/'CURVATURE_REBUILD_PARITY.json').read_text())['status']=='PASS'
    assert json.loads((CURVATURE/'V132_CURVATURE_PARITY.json').read_text())['status']=='PASS'
    cf=json.loads((CURVATURE/'CURVATURE_BINDING.json').read_text())['contract']
    assert cf['mapping_binding']==oldlock['mapping_binding'] and cf['mask']=='m[:, :-1] = labels[:, 1:] != -100'
    prior=json.loads((OLD/'CURVATURE_BINDING.json').read_text())['contract']
    for key in ('mapping_binding','original_runtime','precision','mask','normalization','Base_seed','probe_seed_base','aliases'):
        assert cf[key]==prior[key],key
    fallback=r.p.load_state(CURVATURE/'curvature/COMMON_FALLBACK.pt')
    caches={name:dict(parts=[fallback,r.p.load_state(CURVATURE/'curvature'/f'{name}.pt')['mapped_part']]) for name in ('FULL_CTX','MED')}
    for cache in caches.values():
        for part in cache['parts']:
            for k in ('x','y'):part[k]=part[k].to('cuda')
    original_penalty=r.mm.operator_penalty
    r.mm.operator_penalty=lambda state,cache:curvature.penalty(state,cache,r.mm.factors) if 'parts' in cache else original_penalty(state,cache)
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    bindings=r.c.read(PARENT/'private/EVAL_BINDINGS.json')
    assert runtime.generation_config==next(iter(bindings.values()))['generation']
    runtime.run_root=ROOT/'runtime';runtime.run_root.mkdir(exist_ok=True)
    def count(_module,args,kwargs):
        budget();COUNTS['forwards']+=1;COUNTS['non_generation_forwards']+=1
    forward=runtime.model.register_forward_pre_hook(count,with_kwargs=True)
    for entry,want in zip(data['tasks'],expected):
        task=entry['task'];owner=task['order'];assert owner in (147,148,149,150)
        assert task['edit_id']==want['query_id'] and len(entry['source_rows'])==5
        assert not (ROOT/'candidates'/f'MED_{owner}.pt').exists()
        budget();record=r.c.record(task);runtime.model.half()
        try:
            seed_everything(record_seed(record.record_id,'balancedit'))
            editor=BalanceEditPaperSpecEditor(runtime)
            try:
                with torch.no_grad():
                    key,positive,negative,_=editor._anchors(record,runtime.run_root/'inputs/black_images')
                    radius=balanced_radius(key,positive,negative,alpha=.2,distance='euclidean')
                editor.router.add(task['edit_id'],key,radius,());router=editor.router.export_state()
            finally:
                base,target=editor.wrapper.base,editor.target
                editor.reset_editor_state();runtime.replace_module(target,base)
            prepared=r.s.prepare(runtime,bindings,task,entry['source_rows'],True)
        finally:r.old.fp32(runtime)
        assert prepared[0][4]['prompt_ids']==want['prompt_ids']
        assert prepared[0][4]['attention_mask']==want['attention_mask']
        assert prepared[0][4]['image_sha256']==want['image_sha256']
        assert len(router['entries'])==1 and router['entries'][0]['logical_edit_id']==task['edit_id']
        save(f'routers/{owner}.pt',router)
        batches=[r.old.batch_on(runtime,v[1]) for v in prepared]
        seed_everything(task['seed']);initial=r.p.tr.TT4(task['seed'],8,8).state_dict()
        assert sum(v.numel() for v in initial.values())==7168 and torch.count_nonzero(initial['G1'])==0
        expert=r.p.expert(initial,task['seed'],runtime.device)
        hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach()
        try:
            for b in batches:
                hook.set_teacher_routing(b.labels);mask=torch.zeros_like(b.labels,dtype=torch.bool);mask[:,:-1]=b.labels[:,1:]!=-100
                assert torch.equal(hook.token_mask,mask) and b.target_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id
            refs=[r.q.logits(runtime,hook,b,verify_base=True) for b in batches]
            expert.load_state_dict(initial)
            cbinding=dict(arm='C',seed=task['seed'],steps=160,parameters=7168,source_recipe=trajectory.contract.SOURCE_RECIPE,
                source_order=[v['query_id'] for v in entry['source_rows']],source_binding=r.c.digest(trajectory.contract.original.source_binding(entry['source_rows'])),
                scientific_lock=LOCK['candidate_binding'],source_version=LOCK['source_commit'],budget_binding=LOCK['budget_binding'])
            def checkpoint(state):
                save(f'resume/C_{owner}.pt',state)
                write('TRAIN_PROGRESS.json',dict(phase='C',owner=owner,step=state['step'],counts=COUNTS,time=time.time()))
                print('C',owner,state['step'],flush=True)
            result=trajectory.run(r,runtime,hook,expert,batches,'C',None,0.,None,cbinding,checkpoint)
            final=result['state'];gain,values,_,_=r.s.measure(runtime,hook,expert,final,batches,refs,COUNTS)
            gain=float(gain[0]);assert math.isfinite(gain) and gain>0
            with torch.no_grad():denominator=float(curvature.penalty(final,caches['FULL_CTX'],r.mm.factors))
            coefficient=training_math.coefficient(gain,denominator)
            targets=[v['operator_step_norm'] for v in result['curve']];assert len(targets)==160
            save(f'candidates/C_{owner}.pt',dict(BASE=initial,FINAL=final))
            write(f'calibration/{owner}.json',dict(C_gain=gain,C_FULL_CTX_penalty=denominator,
                coefficient=coefficient,targets=targets,source_only=True,curve=result['curve'],values=values))
        finally:hook.detach()
        seed_everything(task['seed']);state=r.mm.canonicalize(initial)[0]
        expert=r.p.expert(state,task['seed'],runtime.device)
        hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach()
        scalar=dict(step=0,second_moments=[0.,0.]);curve=[]
        try:
            with PrefixCache(runtime,batches):
                for step in range(1,161):
                    budget();before=r.old.snapshot(expert)
                    _,losses=trajectory.source_gradients(r,runtime,hook,expert,batches,step)
                    gradients={k:v.grad.detach().clone() for k,v in expert.named_parameters()}
                    selected,scalar,audit=training_math.step(r,before,gradients,caches['MED'],coefficient,scalar,targets[step-1],arm='M13P')
                    expert.load_state_dict(selected);COUNTS['updates']+=1;COUNTS['operator_backwards']+=1
                    curve.append(dict(step=step,source_NLL=losses,**audit))
                    if step%20==0:
                        save(f'resume/MED_{owner}.pt',dict(state=selected,scalar=scalar,step=step,curve=curve,
                            rng=r.c.rng(),numpy_rng=__import__('numpy').random.get_state(),counts=dict(COUNTS),
                            candidate_binding=LOCK['candidate_binding'],clock=CLOCK,owner=owner))
                        write('TRAIN_PROGRESS.json',dict(phase='MED',owner=owner,step=step,counts=COUNTS,time=time.time()))
                        print('MED',owner,step,flush=True)
            final=r.old.snapshot(expert);save(f'candidates/MED_{owner}.pt',final)
            loaded=torch.load(ROOT/'candidates'/f'MED_{owner}.pt',map_location='cpu',weights_only=True)
            assert all(torch.equal(v.cpu(),loaded[k]) for k,v in final.items())
            gain_med,source_values,_,_=r.s.measure(runtime,hook,expert,final,batches,refs,COUNTS)
            write(f'training/{owner}.json',dict(steps=160,coefficient=coefficient,curve=curve,save_load_exact=True,
                source_gain=gain_med.tolist(),source_values=source_values,mask_checked=True,zero_Base_exact=True,counts=dict(COUNTS),time=time.time()))
        finally:hook.detach()
    assert COUNTS['updates']==1280 and len(list((ROOT/'candidates').glob('MED_*.pt')))==4
    forward.remove();write('TRAIN_COMPLETE.json',dict(status='FOUR_MED_AND_FOUR_C_COMPLETE',time=time.time(),counts=COUNTS,
        original146_untouched=True,new_MED_experts=4,new_C_calibration_trajectories=4,actual_new_scientific_updates=1280,
        inherited_original_MED_updates=23360,cuda_max_memory_allocated=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write('TRAIN_FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),counts=COUNTS));raise
    finally:write('TRAIN_WORKER_EXIT.json',dict(time=time.time(),code=code,counts=COUNTS))
