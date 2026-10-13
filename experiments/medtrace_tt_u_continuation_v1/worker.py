"""Fixed-node single inference and actual incremental TT expert banks."""
import copy
from dataclasses import replace
import os
from pathlib import Path
import signal
import time
import traceback
import torch
from common import RUN,LAYER,read,write,save,digest,state_hash,rng,restore_rng,diagnostic_scope,budget,record,local_path,lease,load
from train import clone,teachers_for,continuation,activations,full_vocab_kl
from structures import TT4,optimizer_for

def tasks(cohort):return [t for t in read(RUN/'private/QUEUES.json')['tasks'] if t['cohort']==cohort]
def train_task(t,slot,rows):
    return dict(edit_id=t['edit_id'],order=t['order'],native=t['native'],fit_questions=t['fit_questions'],seed=t['seed']+slot*1000003,U_fit=rows,bindings=dict(cohort=t['cohort'],roles=digest(read(RUN/'private/U_ROLES.json')),runtime=digest(read(RUN/'private/RUNTIME_BINDING.json'))))
def w0_path(t,slot):
    if t['cohort']=='P1':return Path(next(x['path'] for x in read(RUN/'private/ASSET_REUSE_AUDIT.json')['W0'] if x['order']==t['order'] and x['slot']==slot))
    return RUN/'private/edits'/t['anonymous_edit']/'warmup/W0.pt'
def router(runtime,t):
    p=RUN/'private/edits'/t['anonymous_edit']/'ROUTER.pt'
    if p.exists():return torch.load(p,map_location='cpu',weights_only=True)['entries']
    if t['cohort']=='P1':
        old=Path(read(RUN/'PLAN_CONFIG.json')['parent_run'])/'private/edits'/('e'+str(t['order']).zfill(3))/'ROUTER.pt';s=torch.load(old,map_location='cpu',weights_only=True)
    else:
        from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor,balanced_radius,record_seed
        from m3bench_repro.editors.llava_runtime import seed_everything
        editor=BalanceEditPaperSpecEditor(runtime)
        try:
            seed_everything(record_seed(t['edit_id'],'balancedit'))
            with torch.no_grad():key,positive,negative,_=editor._anchors(record(t),runtime.run_root/'inputs/black_images');radius=balanced_radius(key,positive,negative,alpha=.2,distance='euclidean')
            editor.router.add(t['edit_id'],key,radius,());s=editor.router.export_state()
        finally:
            target,base=editor.target,editor.wrapper.base;editor.reset_editor_state();runtime.replace_module(target,base)
    assert len(s['entries'])==1 and s['entries'][0]['logical_edit_id']==t['edit_id'];save(p,s);return s['entries']
def warmup(runtime,t):
    from methods.medtrace.core import MedTraceLayerHook
    from m3bench_repro.editors.llava_runtime import seed_everything
    d=w0_path(t,0).parent;d.mkdir(parents=True,exist_ok=True);e=TT4(t['seed'],8,8).to(runtime.device)
    h=MedTraceLayerHook(runtime.get_module(LAYER),e);h.attach();rec=record(t)
    batches=[runtime.build_edit_batch(rec)]+[runtime.build_edit_batch(replace(rec,question=q)) for q in t['fit_questions']]
    assert all(b.target_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id for b in batches)
    try:
        for phase,steps in [('native',140),('A2',80),('W0',320)]:
            final=d/(phase+'.pt');latest=d/'latest.pt';seed_everything(t['seed'])
            o=torch.optim.AdamW(e.parameters(),lr=1e-3,weight_decay=0) if phase!='W0' else optimizer_for(e,runtime.model)
            binding=dict(task=train_task(t,0,[]),phase=phase,steps=steps,input_state=state_hash(e),execution=read(RUN/'private/GPU_SOURCE_VERSION.json'));curve=[];start=0
            if final.exists():
                s=torch.load(final,map_location='cpu',weights_only=True);assert s['binding']==binding;e.load_state_dict(s['expert']);continue
            if latest.exists():
                s=torch.load(latest,map_location='cpu',weights_only=True);assert s['binding']==binding;e.load_state_dict(s['expert']);o.load_state_dict(s['optimizer']);restore_rng(s);start=s['step'];curve=s['curve']
            began=time.time()
            for step in range(start+1,steps+1):
                budget();o.zero_grad(set_to_none=True);indices=[0] if phase=='native' else [0,1+(step-1)%4];values=[]
                for i in indices:h.set_teacher_routing(batches[i].labels);loss=runtime.compute_loss(batches[i]);(loss/len(indices)).backward();values.append(float(loss.detach()))
                norm=torch.nn.utils.clip_grad_norm_(e.parameters(),1.);assert torch.isfinite(norm);o.step();assert all(torch.isfinite(p).all() for p in e.parameters())
                curve.append(dict(step=step,indices=indices,CE=values,gradient_norm=float(norm),tokens=sum(len(batches[i].target_token_ids) for i in indices),forwards=len(indices),backwards=len(indices)))
                if step%20==0 or step==steps:save(latest,dict(binding=binding,expert=e.state_dict(),optimizer=o.state_dict(),step=step,curve=curve,**rng()));print('WARMUP',t['anonymous_edit'],phase,step,flush=True)
            save(final,dict(binding=binding,expert=e.state_dict(),step=steps,state_hash=state_hash(e)));write(d/(phase+'_TRAINING.json'),dict(status='COMPLETE',binding=binding,curve=curve,steps=steps,seconds=time.time()-began,actual_updates=steps,final_state_hash=state_hash(e)));latest.unlink()
    finally:h.detach()

def evaluate(runtime,bindings,owner,slot,arm,node,points,bank,panel_tasks,mode='single',prefix=1,forced=False,teacher_rows=None):
    from methods.medtrace.core import MedTraceLayerHook
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from m3bench_repro.editors.routing import MemoryRouter,decision_as_json
    ledger=read(RUN/'private/EVAL_LEDGER.json');rows={}
    for t in panel_tasks:
        for ev in t['events']:
            for qid in ev['all_probe_query_ids']:rows[qid]=ledger['queries'][qid]
    tm={}
    if teacher_rows:
        for teacher in teacher_rows:
            _,_,_,_,row,cache=teacher
            q=dict(row,query_id='U_'+digest(row),image_sha256=cache['binding']['input']['image'])
            rows[q['query_id']]=q;tm[q['query_id']]=teacher
    first=torch.load(next(iter(points.values())),map_location='cpu',weights_only=True);e=clone(first['expert'],owner['seed'],runtime.device).requires_grad_(False)
    h=MedTraceLayerHook(runtime.get_module(LAYER),e);h.attach();editor=BalanceEditPaperSpecEditor(runtime);editor.router=MemoryRouter.from_state(dict(distance='euclidean',entries=bank),device=runtime.device)
    weights={eid:dict(path=str(p),hash=torch.load(p,map_location='cpu',weights_only=True)['state_hash']) for eid,p in points.items()}
    phase=dict(mode=mode,prefix=prefix,arm=arm,node=node,slot=slot,forced=forced,weights=weights,router=[{k:v.tolist() if hasattr(v,'tolist') else v for k,v in x.items()} for x in bank],execution=read(RUN/'private/GPU_SOURCE_VERSION.json'),active_target_rule='Exact image identity + question for already inserted targets; excludes matching active targets from locality only')
    write(RUN/'private/bank_bindings'/(digest(phase)+'.json'),phase)
    try:
        for qid,q in rows.items():
            budget();panel=q.get('role','PANEL');dest=RUN/'private/outputs'/mode/owner['anonymous_edit']/('s'+str(slot))/arm/('n'+str(node))/('p'+str(prefix))/('forced' if forced else 'R0')/(digest(qid)+'.json')
            if dest.exists():assert read(dest)['binding']['phase']==phase;continue
            raw=runtime.adapter.prepare_inputs(Path(local_path(q['image_path'])),q['question'],None)
            if qid in tm:
                teacher=tm[qid];cache=teacher[-1];b=dict(question=q['question'],reference=q['reference'],image_sha256=q['image_sha256'],image_path=q['image_path'],prompt_ids=raw['input_ids'].tolist()[0],attention_mask=raw['attention_mask'].tolist()[0],runtime=next(iter(bindings.values()))['runtime'],generation=runtime.generation_config)
            else:
                b=bindings[q['opaque_Base_id']];assert raw['image_sha256']==b['image_sha256'] and raw['input_ids'].tolist()==[b['prompt_ids']] and raw['attention_mask'].tolist()==[b['attention_mask']]
            query=replace(record(owner),record_id='query',question=q['question'],target='',official_rephrase='',image_path=Path(local_path(q['image_path'])))
            h.clear_request_routing()
            with torch.inference_mode():decision=editor._route(query)
            selected=owner['edit_id'] if forced else decision.logical_edit_id if decision.activated else None
            if selected:
                assert selected in points;e.load_state_dict(torch.load(points[selected],map_location='cpu',weights_only=True)['expert'])
                began=time.time()
                with torch.inference_mode(),h.generation_request():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
            else:
                began=time.time()
                with torch.inference_mode():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
            out=dict(raw_answer=g.decoded_text,raw_token_ids=list(g.raw_token_ids));seconds=time.time()-began;kl=None;base_consistency=None
            if qid in tm:
                kwargs,labels,mask,logp,row,cache=tm[qid];h.clear_request_routing()
                if selected:h.set_teacher_routing(labels)
                with torch.no_grad():kl=float(full_vocab_kl(runtime.model(**kwargs).logits[mask],logp))
                base_consistency=out['raw_token_ids']==cache['tokens'];h.clear_request_routing()
            active={(t['native']['image_sha256'],t['native']['question']) for t in panel_tasks[:prefix] if mode=='bank'}
            bind=dict(input=q,judge_input=b,phase=phase,arm=arm,mode=mode+('_forced' if forced else '_R0'),prefix=prefix,owner_order=owner['order'],panel=panel)
            write(dest,dict(binding=bind,R0=out,route=decision_as_json(decision),effective_expert=selected,weight=weights.get(selected),execution_key=digest([bind,out]),seconds=seconds,TT_parameters_per_expert=7168,U_KL=kl,Base_token_consistency=base_consistency,active_target=(q['image_sha256'],q['question']) in active,diagnostic_only=forced))
    finally:
        h.detach();target,base=editor.target,editor.wrapper.base;editor.reset_editor_state();runtime.replace_module(target,base)

def fixed_x(runtime,t,slot,init,teachers):
    from methods.medtrace.core import MedTraceLayerHook
    rec=record(t);batches=[runtime.build_edit_batch(rec)]+[runtime.build_edit_batch(replace(rec,question=q)) for q in t['fit_questions']]
    e=clone(init,t['seed']+slot*1000003,runtime.device);h=MedTraceLayerHook(runtime.get_module(LAYER),e);h.attach()
    try:return activations(runtime,h,batches,teachers)
    finally:h.detach()

def single(runtime,bindings,t,slot):
    roles=read(RUN/'private/U_ROLES.json');fit=roles['FIT'];single=teachers_for(runtime,fit[:1]);multi=teachers_for(runtime,fit) if t['cohort']=='P2' and len(fit)>=2 else []
    cal=teachers_for(runtime,roles['CAL']);check=teachers_for(runtime,roles['CHECK']) if t['cohort']=='P2' else []
    if t['cohort']=='P2':warmup(runtime,t)
    w0=w0_path(t,slot);s=torch.load(w0,map_location='cpu',weights_only=True);assert set(s['expert'])=={'G1','G2','G3','G4'};init=s['expert'];bank=router(runtime,t)
    steps=320 if t['cohort']=='P1' else read(RUN/'private/T_STAR.json')['t_star'];nodes=[40,80,160,320] if t['cohort']=='P1' else [steps]
    root=RUN/'private/edits'/t['anonymous_edit']/('s'+str(slot));root.mkdir(parents=True,exist_ok=True);done=root/'COMPLETE.json'
    if done.exists():return
    x=fixed_x(runtime,t,slot,init,multi or single)
    points={'FROZEN_W0':w0};conditions={'CE_ONLY':[]}
    if single:conditions['CE_U_CLEAN' if t['cohort']=='P1' else 'CE_U_SINGLE']=single
    if multi:conditions['CE_U_MULTI']=multi
    evaluate(runtime,bindings,t,slot,'FROZEN_W0',0,{t['edit_id']:w0},bank,[t],teacher_rows=cal+(check if t['cohort']=='P2' else []))
    evaluate(runtime,bindings,t,slot,'FROZEN_W0',0,{t['edit_id']:w0},bank,[t],forced=True,teacher_rows=cal+check)
    for arm,teachers in conditions.items():
        rows=[x[4] for x in teachers];ct=train_task(t,slot,rows);d=root/arm
        continuation(runtime,ct,record(t),init,d,teachers,steps,nodes,x)
        for node in nodes:
            p=d/('step'+str(node)+'.pt');evaluate(runtime,bindings,t,slot,arm,node,{t['edit_id']:p},bank,[t],teacher_rows=(teachers+cal+check))
            if t['cohort']=='P2' or node in [80,320]:evaluate(runtime,bindings,t,slot,arm,node,{t['edit_id']:p},bank,[t],forced=True,teacher_rows=cal+check)
        points[arm]=d/('step'+str(steps)+'.pt')
    write(done,dict(status='GENERATED_NOT_SCORED',cohort=t['cohort'],slot=slot,points={k:str(v) for k,v in points.items()},steps=steps,conditions=list(points),fixed_activation_shape=list(x.shape)))
    print('UNIT_COMPLETE',t['anonymous_edit'],slot,flush=True)

def banks(runtime,bindings):
    ts=[t for t in tasks('P2') if (RUN/'private/edits'/t['anonymous_edit']/'s0/COMPLETE.json').exists()];assert len(ts) in (8,16,24)
    arms=read(RUN/'private/edits'/ts[0]['anonymous_edit']/'s0/COMPLETE.json')['conditions'];roles=read(RUN/'private/U_ROLES.json');check=teachers_for(runtime,roles['CHECK']);node=read(RUN/'private/T_STAR.json')['t_star']
    for arm in arms:
        bank=[];points={}
        for i,t in enumerate(ts,1):
            bank+=router(runtime,t);points[t['edit_id']]=Path(read(RUN/'private/edits'/t['anonymous_edit']/'s0/COMPLETE.json')['points'][arm])
            native=dict(t,events=[e for e in t['events'] if e['task']=='T0'])
            evaluate(runtime,bindings,t,0,arm,0 if arm=='FROZEN_W0' else node,points,bank,[native],mode='insertion',prefix=i)
            if i in (8,16,24):
                evaluate(runtime,bindings,t,0,arm,0 if arm=='FROZEN_W0' else node,points,bank,ts[:i],mode='bank',prefix=i,teacher_rows=check)
                for owner in ts[:i]:
                    # Corresponding expert diagnostic per owning edit, separate from R0.
                    evaluate(runtime,bindings,owner,0,arm,0 if arm=='FROZEN_W0' else node,points,bank,[owner],mode='bank',prefix=i,forced=True,teacher_rows=check)
        write(RUN/'private'/('BANK_'+arm+'_COMPLETE.json'),dict(status='GENERATED_NOT_SCORED',experts=len(ts),prefixes=[p for p in (8,16,24) if p<=len(ts)],real_incremental_insertion=True))

def main():
    gpu=int(os.environ['GPU']);action=os.environ['ACTION']
    with lease(gpu):
        runtime,bindings=load(gpu)
        if action=='mechanical':
            from qualification import gpu_check
            gpu_check(runtime,tasks('P1')[0]);return
        assert read(RUN/'private/GPU_MECHANICAL.json')['status']=='PASS'
        if action=='P1':
            for t in tasks('P1')[int(os.environ['PARTITION'])::2]:
                for slot in range(3):single(runtime,bindings,t,slot)
        elif action=='P2':
            ids=read(RUN/'private/QUEUES.json')['P2_blocks'][int(os.environ['BLOCK'])]
            for t in [t for t in tasks('P2') if t['edit_id'] in ids][int(os.environ['PARTITION'])::2]:single(runtime,bindings,t,0)
        elif action=='bank':banks(runtime,bindings)
        else:raise ValueError(action)
        write(RUN/'private'/('DONE_'+action+'_'+os.environ.get('BLOCK','all')+'_'+os.environ.get('PARTITION','0')+'.json'),dict(status='GENERATED_NOT_SCORED',epoch=time.time()))
if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Own bounded worker interrupted; preserve resume')))
    try:main()
    except BaseException as e:write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('GPU','cpu')+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time()));raise
