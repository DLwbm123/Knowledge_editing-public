"""Frozen source-only TT/full-weight banks; inherited native runtime and outputs."""
import os, sys, time, json, traceback
from pathlib import Path
from contextlib import contextmanager
sys.path.insert(0, str(Path(os.environ['RUN_ROOT']) / 'private/official_llava'))
import torch
import torch.nn.functional as F
import source_run as b
from tt_cores import TT4, optimizer_for
from methods.medtrace import MedTraceLayerHook
from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
from m3bench_repro.editors.routing import balanced_radius, decision_as_json
from purew_math import target_logits, preservation_kl
c,j,v,data,old,RUN=b.c,b.j,b.v,b.data,b.old,b.RUN
ARMS=('TT88_SOURCE','BALANCEDIT_SOURCE')

def tracked(runtime):
    counts=dict(forwards=0,backwards=0,updates=0,generations=0,generation_attempts=0,generation_forwards=0)
    path=RUN/'private/attempts'/f'{os.environ["ACTION"]}_{os.getpid()}_{time.time_ns()}.json'
    def persist():
        j.write(path,dict(counts,lock=b.lock()))
        totals={k:sum(c.read(p)[k] for p in path.parent.glob('*.json')) for k in counts}
        limits=c.read(RUN/'private/FROZEN_PROTOCOL.json')
        assert totals['generation_attempts']<=limits['maximum_generations']
        assert totals['backwards']<=limits['maximum_backwards']
        assert totals['updates']<=limits['maximum_formal_updates']
        assert totals['forwards']-totals['generation_forwards']<=limits['maximum_non_generation_forwards']
    def before(*_):
        counts['forwards']+=1
        if runtime._benchmark_generating:counts['generation_forwards']+=1
        persist()
    runtime._benchmark_generating=False
    handle=runtime.model.register_forward_pre_hook(before);persist()
    return counts,handle,persist

def record(row):return data.record(b.task(row),row)
def bank_path(arm,owner):return RUN/'private/banks'/arm/f'{owner:02d}.pt'

class Bank:
    def __init__(self,runtime,arm):
        self.runtime,self.arm=runtime,arm
        self.editor=BalanceEditPaperSpecEditor(runtime)
        assert self.editor.target=='model.layers.31.mlp.up_proj'
        self.experts={}
    def add(self,row,state):
        key=row['query_id'];assert state['lock']==b.lock() and state['owner']==row['owner']
        self.editor.router.add(key,state['route_key'].to(self.runtime.device),state['radius'],state['tokens'])
        if self.arm=='TT88_SOURCE':
            e=TT4(20261011+row['owner'],8,8).to(self.runtime.device);e.load_state_dict(state['expert']);e.requires_grad_(False)
            self.experts[key]=e
        else:
            e=self.editor.wrapper.add_edit(key);e.load_state_dict(state['expert']);self.editor.wrapper.train_only(None)
        return e
    def load(self,upto):
        for row in b.rows():
            if row['audit_role']=='NATIVE' and row['owner']<=upto:
                self.add(row,torch.load(bank_path(self.arm,row['owner']),map_location='cpu',weights_only=True))
    @contextmanager
    def routed(self,row):
        decision=self.editor._route(record(row));hook=None
        if self.arm=='TT88_SOURCE' and decision.logical_edit_id is not None:
            hook=MedTraceLayerHook(self.runtime.get_module(v.LAYER),self.experts[decision.logical_edit_id]);hook.attach()
        try:
            if hook:
                with hook.generation_request():yield decision
            else:
                with self.editor._activated(decision.logical_edit_id if self.arm=='BALANCEDIT_SOURCE' else None):yield decision
        finally:
            if hook:hook.detach()
    def summary(self):
        if self.arm=='TT88_SOURCE':
            return dict(added_parameters=7168*len(self.experts),weight_bytes=4*7168*len(self.experts),experts=len(self.experts),structure='FOUR_TT_CORES_RMS_ASSISTANT_PREDICTORS',routing_parameters=0,routing_key_elements=4096*len(self.experts),editor_branch=True)
        x=self.editor.state_summary()
        return dict(added_parameters=x['edited_parameter_count'],weight_bytes=x['edited_parameter_bytes'],experts=x['entry_count'],structure='FULL_FP32_UP_PROJ_BANK',routing_parameters=0,routing_key_elements=4096*x['entry_count'],editor_branch=True)
    def close(self):
        self.runtime.replace_module(self.editor.target,self.editor.wrapper.base)
        self.experts.clear()

def generate(runtime,bindings,row,stage,bank,count,persist):
    path=v.output(stage,row['owner'],row)
    if path.exists():
        saved=c.read(path);assert saved['lock']==b.lock() and saved['binding']['input']==row;return path
    attempt=RUN/'private/generation_attempts'/stage/str(row['owner'])/(row['query_id']+'.json')
    assert not attempt.exists(),'A failed generation attempt cannot be silently repeated'
    c.budget();item=v.prepare(runtime,bindings,[row])[0]
    j.write(attempt,dict(lock=b.lock(),epoch=time.time()))
    count['generation_attempts']+=1;persist()
    with torch.inference_mode(),bank.routed(row) as decision:
        runtime._benchmark_generating=True
        try:path=v.emit(runtime,item,stage,count)
        finally:runtime._benchmark_generating=False
    saved=c.read(path);saved['route']=decision_as_json(decision);j.write(path,saved);persist()
    return path

def train_expert(runtime,bindings,bank,row,count,persist,mechanical=False):
    arm=bank.arm;owner=row['owner'];key=row['query_id'];active=RUN/'private/active'/f'{arm}.pt'
    assert row['audit_role']=='NATIVE' and row['source_admission']=='PASS'
    item=v.prepare(runtime,bindings,[row],True)[0];batch=old.batch_on(runtime,item[1])
    torch.manual_seed(20261011+owner);torch.cuda.manual_seed_all(20261011+owner)
    keyvec,positive,negative,_=bank.editor._anchors(record(row),runtime.run_root/'black')
    assert torch.equal(keyvec,positive),'Source-only positive must equal original key'
    radius=balanced_radius(keyvec,positive,negative,alpha=.2,distance='euclidean')
    state=dict(lock=b.lock(),owner=owner,route_key=keyvec.detach().cpu(),radius=float(radius),tokens=batch.target_token_ids,phase=0,step=0,last_event=None,committed=0)
    if arm=='TT88_SOURCE':e=TT4(20261011+owner,8,8).to(runtime.device)
    else:e=bank.editor.wrapper.add_edit(key);bank.editor.wrapper.train_only(key);bank.editor.wrapper.set_active(key)
    hook=MedTraceLayerHook(runtime.get_module(v.LAYER),e) if arm=='TT88_SOURCE' else None
    if hook:hook.attach()
    phases=[('NATIVE',140),('FIT_SOURCE',80),('W0',320)] if hook else [('CE',50)]
    if mechanical:phases=[('NATIVE' if hook else 'CE',1)]
    saved=torch.load(active,map_location='cpu',weights_only=True) if active.exists() and not mechanical else None
    if saved:
        assert saved['lock']==b.lock() and saved['owner']==owner
        state=saved;e.load_state_dict(state['expert']);j.ensure_event(RUN/'private/training_diagnostics'/arm,state)
    frozen=[(p,p._version) for p in runtime.model.parameters() if not any(p is z for z in e.parameters())]
    initial=[p.detach().clone() for p in e.parameters()]
    try:
        for phase,(name,steps) in enumerate(phases):
            if phase<state['phase']:continue
            torch.manual_seed(20261011+owner);torch.cuda.manual_seed_all(20261011+owner)
            opt=optimizer_for(e,runtime.model) if name=='W0' else torch.optim.AdamW(e.parameters(),lr=.001,weight_decay=0) if hook else torch.optim.Adam(e.parameters(),lr=.01)
            if saved and phase==state['phase'] and 'optimizer' in state:
                opt.load_state_dict(state['optimizer']);c.restore_rng(state['rng'])
            for step in range(state['step']+1,steps+1):
                c.budget();opt.zero_grad(set_to_none=True)
                if hook:hook.set_teacher_routing(batch.labels)
                z,y=target_logits(runtime.model,batch);loss=F.cross_entropy(z,y)
                count['backwards']+=1;persist();loss.backward();terms={'CE':float(loss.detach())};del z,loss
                if name=='W0':
                    index=((owner-1)*320+step-1)%61
                    basis=data.q.protection_batch(runtime,data.q.split()[0][index],data.tasks()[0])
                    hook.set_teacher_routing(basis.labels)
                    z,_=target_logits(runtime.model,basis)
                    ref=torch.load(RUN/'private/cache/protection'/f'{index}.pt',map_location='cpu',weights_only=True)
                    assert ref['lock']==b.lock()
                    loss=.01*preservation_kl(z,ref['reference'].to(runtime.device));count['backwards']+=1;persist();loss.backward();terms['KL_WEIGHTED']=float(loss.detach());del z,loss
                norm=torch.nn.utils.clip_grad_norm_(e.parameters(),1.)
                assert torch.isfinite(norm) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in e.parameters())
                before=[p.detach().clone() for p in e.parameters()];opt.step()
                assert all(torch.isfinite(p).all() for p in e.parameters()) and all(p._version==version for p,version in frozen)
                update=state['committed']+1
                event=dict(update=(owner-1)*(540 if hook else 50)+update,owner=owner,phase=name,step=step,losses=terms,gradient_norm=float(norm),parameter_update_norm=sum(float((p-a).norm())**2 for p,a in zip(e.parameters(),before))**.5)
                state.update(phase=phase,step=step,committed=update,last_event=event,expert={k:x.detach().cpu() for k,x in e.state_dict().items()},optimizer=opt.state_dict(),rng=c.rng())
                if not mechanical:
                    c.save(active,state)
                    with active.open('rb') as f:os.fsync(f.fileno())
                    j.ensure_event(RUN/'private/training_diagnostics'/arm,state);count['updates']+=1;persist()
                if step%20==0:
                    print('UPDATE',arm,owner,name,step,flush=True);j.write(RUN/'private/progress'/f'{arm}.json',event)
            state.update(phase=phase+1,step=0);state.pop('optimizer',None);saved=None
            if not mechanical:c.save(active,state)
        assert any(not torch.equal(p,a) for p,a in zip(e.parameters(),initial))
        state['expert']={k:x.detach().cpu() for k,x in e.state_dict().items()};state.pop('rng',None);state.pop('optimizer',None)
        if not mechanical:
            c.save(bank_path(arm,owner),state);j.write(RUN/'private/expert_receipts'/arm/f'{owner:02d}.json',dict(owner=owner,steps=state['committed'],bytes=bank_path(arm,owner).stat().st_size,parameters=sum(p.numel() for p in e.parameters()),lock=b.lock()))
            active.unlink()
        return state
    finally:
        if hook:hook.detach()
        else:bank.editor.wrapper.set_active(None);bank.editor.wrapper.train_only(None)

def admission():
    if b.complete('BASELINE_ADMISSION.json'):return
    b.set_tasks(b.rows());c.available=v.admission
    with c.lease(5):
        runtime,bindings=c.load(5);v.fp32_native(runtime);count,h,persist=tracked(runtime)
        try:
            native=next(r for r in b.rows() if r['audit_role']=='NATIVE')
            for index,row in enumerate(data.q.split()[0]):
                path=RUN/'private/cache/protection'/f'{index}.pt'
                if path.exists():assert torch.load(path,weights_only=True)['lock']==b.lock();continue
                batch=data.q.protection_batch(runtime,row,data.tasks()[0])
                with torch.no_grad():z,_=target_logits(runtime.model,batch)
                c.save(path,dict(reference=z.cpu(),lock=b.lock()))
            reports=[]
            for arm in ARMS:
                bank=Bank(runtime,arm)
                identity=generate(runtime,bindings,native,f'{arm}_ZERO_IDENTITY',bank,count,persist)
                assert c.read(identity)['R0']==c.read(v.output('BASE',native['owner'],native))['R0']
                state=train_expert(runtime,bindings,bank,native,count,persist,True)
                # Actual trained state survives serialization without retaining a formal edit.
                path=RUN/'private/admission'/f'{arm}.pt';c.save(path,state);reload=torch.load(path,map_location='cpu',weights_only=True)
                assert all(torch.equal(x,reload['expert'][k]) for k,x in state['expert'].items())
                if arm=='BALANCEDIT_SOURCE':bank.editor.wrapper.clear()
                bank.add(native,reload)
                probe=generate(runtime,bindings,native,f'{arm}_MECHANICAL',bank,count,persist)
                assert c.read(probe)['route']['logical_edit_id']==native['query_id']
                reports.append(bank.summary());path.unlink();bank.close()
            assert not any(p.requires_grad or p.grad is not None for p in runtime.model.parameters())
            b.done('BASELINE_ADMISSION.json',status='PASS',exact_disabled_Base_identity=True,real_gradient_writes=True,save_load_exact=True,reports=reports)
        finally:h.remove();persist()

def train():
    arm=ARMS[int(os.environ['PARTITION'])]
    if b.complete(f'TRAIN_COMPLETE_{arm}.json'):return
    assert b.complete('BASELINE_ADMISSION.json')
    b.set_tasks(b.rows());c.available=v.admission
    with j.exclusive(RUN/'private/locks'/f'{arm}.lock'),c.lease(5):
        runtime,bindings=c.load(5);v.fp32_native(runtime);count,h,persist=tracked(runtime);bank=Bank(runtime,arm)
        try:
            for owner in range(1,25):
                row=next(x for x in b.rows() if x['owner']==owner and x['audit_role']=='NATIVE')
                path=bank_path(arm,owner)
                if path.exists():state=torch.load(path,map_location='cpu',weights_only=True)
                else:
                    state=train_expert(runtime,bindings,bank,row,count,persist)
                    if arm=='BALANCEDIT_SOURCE':
                        bank.editor.router.add(row['query_id'],state['route_key'].to(runtime.device),state['radius'],state['tokens'])
                if arm=='TT88_SOURCE' or path.exists() and row['query_id'] not in bank.editor.wrapper.logical_to_slot:bank.add(row,state)
                assert len(bank.editor.router)==owner
                for query in b.rows():
                    if query['audit_role'] in ('NATIVE','T2G') and query['owner']<=owner:
                        generate(runtime,bindings,query,f'{arm}_PREFIX_{owner}',bank,count,persist)
                    elif owner in (1,8,16,24) and query['audit_role'] in ('HELDOUT','NEW_HELDOUT'):
                        generate(runtime,bindings,query,f'{arm}_PROTECT_{owner}',bank,count,persist)
                j.write(RUN/'private/prefix_receipts'/arm/f'{owner:02d}.json',dict(lock=b.lock(),prefix=owner,summary=bank.summary()))
            for row in b.rows():
                prior=c.read(v.output(f'{arm}_PREFIX_24' if row['audit_role'] in ('NATIVE','T2G') else f'{arm}_PROTECT_24',row['owner'],row))
                j.write(v.output(arm,row['owner'],row),dict(prior,stage=arm,reused_generation='PREFIX24'))
            b.done(f'TRAIN_COMPLETE_{arm}.json',committed_updates=12960 if arm=='TT88_SOURCE' else 1200,summary=bank.summary())
        finally:bank.close();h.remove();persist()

def performance():
    if b.complete('PERFORMANCE_COMPLETE.json'):return
    b.set_tasks(b.rows());c.available=v.admission
    with c.lease(5):
        runtime,bindings=c.load(5);v.fp32_native(runtime);count,h,persist=tracked(runtime)
        try:
            nativeW=runtime.get_module(v.LAYER).weight;original=nativeW.detach().clone()
            # Persisted queue has four T2G per owner; choose first in queue, independent of labels.
            queries=[]
            for owner in (1,8,16,24):
                queries.extend([next(r for r in b.rows() if r['owner']==owner and r['audit_role']==role) for role in ('NATIVE','T2G')])
            held=[r for r in b.rows() if r['audit_role']=='NEW_HELDOUT'];queries.extend(held[i] for i in (0,10,20,30));assert len(queries)==12
            results={}
            for arm in ('BASE','U','H_SOURCE')+ARMS:
                bank=None
                if arm in ARMS:
                    bank=Bank(runtime,arm);bank.load(24)
                    row=next(r for r in b.rows() if r['owner']==24 and r['audit_role']=='NATIVE')
                    probe=generate(runtime,bindings,row,arm+'_RELOAD',bank,count,persist)
                    assert c.read(probe)['R0']==c.read(v.output(arm,24,row))['R0']
                    b.done('RELOAD_'+arm+'.json',native_fresh_reload_tokens_equal=True,summary=bank.summary())
                elif arm!='BASE':
                    saved=torch.load(RUN/'private/parent_candidates'/f'{arm}.pt',map_location='cpu',weights_only=True)
                    assert saved['lock']==c.read(RUN/'private/PARENT_BINDING.json')['scientific_lock']
                    with torch.no_grad():nativeW.copy_(saved['weight'].to(runtime.device))
                events=[];torch.cuda.reset_peak_memory_stats()
                for slot in range(37):
                    eventpath=RUN/'private/performance'/arm/f'{slot:02d}.json'
                    if eventpath.exists():events.append(c.read(eventpath));continue
                    row=queries[0] if slot==0 else queries[(slot-1)%12]
                    attempt=eventpath.with_suffix('.attempt.json');assert not attempt.exists(),'No timing retry after an attempted generation'
                    j.write(attempt,dict(lock=b.lock(),epoch=time.time()))
                    count['generation_attempts']+=1;persist();torch.cuda.synchronize();start=time.perf_counter()
                    # Disk counter writes are suspended during timing; forwards are counted in RAM.
                    h.remove();forwards=[0];timer=runtime.model.register_forward_pre_hook(lambda *_:forwards.__setitem__(0,forwards[0]+1))
                    
                    # Wall prefill/decode includes native forward completion, including TT hooks.
                    wall=[]
                    pre=runtime.model.register_forward_pre_hook(lambda *_:wall.append(time.perf_counter()))
                    def after_forward(*_):
                        torch.cuda.synchronize();wall.append(time.perf_counter())
                    post=runtime.model.register_forward_hook(after_forward)
                    try:
                        item=v.prepare(runtime,bindings,[row])[0];prepared=time.perf_counter()
                        if bank:
                            with torch.inference_mode(),bank.routed(row):
                                route=time.perf_counter();out=runtime.adapter.generate_prepared_with_result(item[2],runtime.generation_config)
                        else:
                            route=prepared
                            with torch.inference_mode():out=runtime.adapter.generate_prepared_with_result(item[2],runtime.generation_config)
                        torch.cuda.synchronize();finish=time.perf_counter()
                        if slot==0 and arm in ('U','H_SOURCE'):
                            parent=Path(c.read(RUN/'private/FROZEN_PROTOCOL.json')['source_parent'])
                            original_output=c.read(parent/'private/outputs'/arm/str(row['owner'])/(row['query_id']+'.json'))
                            assert list(out.raw_token_ids)==original_output['R0']['raw_token_ids']
                    finally:
                        timer.remove();pre.remove();post.remove()
                        def before(*_):
                            count['forwards']+=1
                            if runtime._benchmark_generating:count['generation_forwards']+=1
                        h=runtime.model.register_forward_pre_hook(before)
                    count['forwards']+=forwards[0];count['generation_forwards']+=max(0,forwards[0]-(1 if bank else 0));count['generations']+=1;persist()
                    event=dict(slot=slot,warmup=slot==0,query_slot=0 if slot==0 else (slot-1)%12,seconds=finish-start,prepare_seconds=prepared-start,route_seconds=route-prepared,tokens=len(out['raw_token_ids']) if isinstance(out,dict) else len(out.raw_token_ids),forward_seconds=[wall[i+1]-wall[i] for i in range(2 if bank else 0,len(wall),2)],forwards=forwards[0])
                    j.write(eventpath,event);events.append(event)
                summary=bank.summary() if bank else dict(added_parameters=0,editor_branch=False,fresh_reload_tokens_equal=arm!='BASE')
                summary.update(native_parameter_count=7566219264,base_native_weight_bytes=4*7566219264,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),serialized_bank_bytes=sum(p.stat().st_size for p in (RUN/'private/banks'/arm).glob('*.pt')) if bank else 0)
                results[arm]=dict(summary=summary,events=events)
                if bank:bank.close()
                with torch.no_grad():nativeW.copy_(original)
            j.write(RUN/'public/PERFORMANCE.json',results);b.done('PERFORMANCE_COMPLETE.json',methods=5,queries=12,repeats=3)
        finally:h.remove();persist()

def controller():
    import pipeline
    with j.exclusive(RUN/'private/locks/controller.lock'):
        stages=[('baseline_admission',0,'BASELINE_ADMISSION.json'),('baseline_train',0,'TRAIN_COMPLETE_TT88_SOURCE.json'),('baseline_train',1,'TRAIN_COMPLETE_BALANCEDIT_SOURCE.json'),('baseline_performance',0,'PERFORMANCE_COMPLETE.json')]
        for action,part,receipt in stages:
            if b.complete(receipt):continue
            path=RUN/'private'/f'START_CHAIN_{action}_{part}.json'
            prior=c.read(path) if path.exists() else None
            if prior and j.alive(prior):start=prior
            elif prior:raise RuntimeError('Owned stage ended without terminal receipt; retained evidence requires engineering review')
            else:
                pipeline.launch('bench.py',action,5,part,os.environ['TRAIN_PYTHON']);start=c.read(path)
            while j.alive(start):c.budget();time.sleep(20)
            assert b.complete(receipt),f'{action} stopped without completion'
        paths=list((RUN/'private/banks').glob('*/*.pt'))+list((RUN/'private/parent_candidates').glob('*.pt'))+list((RUN/'private/cache/protection').glob('*.pt'))
        receipt=dict(files=len(paths),paths=list(map(str,paths)),bytes=sum(x.stat().st_size for x in paths),last_consumers=['all prefixes','fresh bank reload','bound efficacy outputs','fixed inference timing'],lock=b.lock())
        j.write(RUN/'private/DELETION_AUTHORIZED.json',receipt)
        for path in paths:path.unlink()
        j.write(RUN/'private/DELETION.json',receipt);b.done('GENERATION_COMPLETE.json',status='COMPLETE')

if __name__=='__main__':
    try:
        {'baseline_admission':admission,'baseline_train':train,'baseline_performance':performance,'baseline_controller':controller}[os.environ['ACTION']]()
    except BaseException as e:
        j.write(RUN/'private/failures'/f'{os.environ["ACTION"]}_{os.getpid()}_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),lock=b.lock()));raise
