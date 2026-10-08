"""Finite paper-module transfer campaign over the frozen exposed development data."""
import os
import time
import traceback
import random
import json
import subprocess
from pathlib import Path
from contextlib import contextmanager
from dataclasses import replace
import core
import structure as s

c,tr,w,RUN,BASE=core.common,core.train,core.worker,core.RUN,core.BASE
PARENT=Path(os.environ['PAPER_PARENT'])
import torch
from methods.medtrace.core import MedTraceLayerHook
A_ROUTES=('R0','NORM_MEAN','MODAL','INTRINSIC_ALL','INTRINSIC_MODAL')
ARMS=('TT_CE','TT_ALIGN','TT_FIXED_A')
ROUTES=('R0','MODAL','INTRINSIC_MODAL')

def tasks(n=146):
    return c.read(RUN/'private/BENCHMARK146_QUEUE.json')['tasks'] if n==146 else [t for t in c.read(BASE/'private/QUEUES.json')['tasks'] if t['cohort']=='P2']

def initial(t):
    q=c.read(RUN/'private/BENCHMARK146_QUEUE.json')
    return Path(q['reused'].get(t['edit_id'],str(PARENT/'private/edits'/t['anonymous_edit']/'warmup/W0.pt')))

def point(t,arm):
    assert arm in ('W0',)+ARMS, 'Only TT is authorized'
    if arm=='W0':return initial(t)
    if arm=='TT_CE':return BASE/'private/edits'/t['anonymous_edit']/'s0/CE_ONLY/step160.pt'
    return RUN/'private/weights'/arm/t['anonymous_edit']/'step160.pt'

def load_state(p):return torch.load(p,map_location='cpu',weights_only=True)

def expert(state,seed,device='cpu'):
    assert set(state)=={'G1','G2','G3','G4'}, 'Only TT cores are authorized'
    e=tr.clone(state,seed,'cpu')
    e.load_state_dict(state);return e.to(device)

def optimizer(e,base):
    assert not any(p.requires_grad for p in base.parameters())
    ip,op=e.parameter_groups()
    return torch.optim.Adam([{'params':[p for p in ip if p.requires_grad],'lr':1e-4},{'params':[p for p in op if p.requires_grad],'lr':1e-3}],betas=(.9,.999),eps=1e-8,weight_decay=0)

@contextmanager
def lease(gpu):
    import fcntl
    assert gpu in (5,6,7)
    uuid,free=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    assert uuid==c.read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)] and int(free)>=24000
    with (RUN/'private'/('lease.'+os.environ['ACTION']+'.'+os.environ.get('PARTITION','0'))).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entry=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],gpu_uuid=uuid,started_epoch=time.time(),action=os.environ['ACTION'],part=os.environ.get('PARTITION'))
        with c.resources() as r:r['gpu_sessions'].append(entry)
        try:yield
        finally:
            with c.resources() as r:
                x=next(x for x in r['gpu_sessions'] if x['pid']==entry['pid'] and x['started_epoch']==entry['started_epoch']);x.update(ended_epoch=time.time(),resident_seconds=time.time()-entry['started_epoch']);r['gpu_seconds_used']+=x['resident_seconds']

def done(name,data=None):c.write(RUN/'private'/(name+'.json'),dict(status='COMPLETE',epoch=time.time(),**(data or {})))
def progress(stage):c.write(RUN/'public/PROGRESS.json',dict(status=stage,complete=False,epoch=time.time()))
def feature_path(row):return RUN/'private/features'/(c.digest([row['image_sha256'],row['question']])+'.pt')
def features(row,route=False):return load_state(feature_path(row))['route_vectors' if route else 'vectors']
def train_rows(t):return [dict(t['native'],question=q) for q in [t['native']['question']]+t['fit_questions']]
def queries(n):return c.read(RUN/'private/PAPER_QUERIES.json')[str(n)]
def lock():return c.digest(c.read(RUN/'private/PAPER_LOCK.json'))
def amendment():return c.digest(c.read(RUN/'private/TT_ONLY_AMENDMENT.json'))
def execution(stage):return c.read(RUN/'private'/('GPU_SOURCE_VERSION.json' if stage=='A' else 'PAPER_TT_SOURCE_VERSION.json'))

def cpucheck():
    result=s.selfcheck()
    base=torch.nn.Linear(1,1).requires_grad_(False)
    e=tr.clone(tr.TT4(3,8,8).state_dict(),3,'cpu')
    with torch.no_grad():e.G1.normal_(0,.01)
    for value in e.parameter_groups()[0]:value.requires_grad_(False)
    saved=[v.clone() for v in e.parameter_groups()[0]]
    opt=optimizer(e,base);e.residual(torch.randn(2,14336)).square().mean().backward();opt.step()
    assert all(torch.equal(a,b) and b.grad is None for a,b in zip(saved,e.parameter_groups()[0]))
    assert all(p.grad is not None for p in e.parameter_groups()[1])
    other=tr.TT4(4,8,8)
    with torch.no_grad():other.G1.normal_(0,.01)
    mixture=Mixture([e,other]);mixture.ids=[0,1];mixture.weights=[.3,.7]
    x=torch.randn(2,14336)
    assert torch.equal(mixture.residual(x),.3*e.residual(x)+.7*other.residual(x))
    other.zero_grad(set_to_none=True)
    (-s.input_score(other.factors()[1],torch.randn(3,14336))).backward()
    assert other.G3.grad.norm()>0 and other.G4.grad.norm()>0 and other.G1.grad is None and other.G2.grad is None
    result['checks']+=['TT intrinsic routing gradient reaches only input cores','TT input factors unchanged after optimizer update','mixture exact weighted residual']
    return result

def prepare():
    ts=tasks();small=tasks(24);assert len(ts)==146 and len(small)==24
    ledger=c.read(RUN/'private/EVAL_LEDGER.json');qs={}
    import sqlite3
    old={};check={}
    for folder,label,method,n in [('judge_astra_medium','A_W0_R0','TT88_W0_RETRO146',146),('judge_combo24_astra_medium','B_TT_CE_R0','CE_ONLY',24)]:
        db=sqlite3.connect('file:'+str(PARENT/'private'/folder/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        rows=db.execute("SELECT c.*,p.binding AS payload_binding FROM consumer c JOIN payload p ON p.key=c.payload_key WHERE c.method=? AND c.mode='bank_R0' AND c.prefix=?",(method,n)).fetchall()
        assert len(rows)==(1509 if n==146 else 283)
        # One complete bank binding proves the reused phase's weights and original router.
        first=c.read(rows[0]['path']);phase=first['binding']['phase']
        assert set(phase['weights'])=={x['edit_id'] for x in tasks(n)}
        expected={x['edit_id']:dict(path=str(point(x,'W0' if n==146 else 'TT_CE')),hash=load_state(point(x,'W0' if n==146 else 'TT_CE'))['state_hash']) for x in tasks(n)}
        assert phase['weights']==expected
        expected_router=[]
        for task in tasks(n):
            for entry in load_state(initial(task).parent.parent/'ROUTER.pt')['entries']:
                expected_router.append({k:v.tolist() if hasattr(v,'tolist') else v for k,v in entry.items()})
        assert phase['router']==expected_router,'Reused R0 requires identical frozen router entries'
        c.write(RUN/'private'/('REUSE_PHASE_'+str(n)+'.json'),dict(phase=phase,first_output_binding=rows[0]['output_binding']))
        for item in rows:
            full=json.loads(item['payload_binding']);record=dict(path=item['path'],payload=full,original_output_binding=item['output_binding'])
            if item['folder']=='CHECK':
                d=c.read(item['path']);check[item['query_id']]=d['binding']['input'];record.update(U_KL=d['U_KL'],Base_token_consistency=d['Base_token_consistency'])
            old[label+'|'+item['query_id']]=record
        db.close()
    assert len(check)==4
    for n in (146,24):
        ids={q for t in tasks(n) for e in t['events'] for q in e['all_probe_query_ids']}
        qs[str(n)]={q:ledger['queries'][q] for q in sorted(ids)};qs[str(n)].update(check)
    assert len(qs['146'])==1513 and len(qs['24'])==283
    c.write(RUN/'private/PAPER_QUERIES.json',qs)
    rows={}
    for row in list(qs['146'].values())+[r for t in ts for r in train_rows(t)]:rows[str(feature_path(row))]=row
    c.write(RUN/'private/FEATURE_ROWS.json',list(rows.values()))
    assert len(old)==1792
    c.write(RUN/'private/REUSE_OUTPUTS.json',old)
    result=cpucheck();result.update(feature_inputs=len(rows),consumers=10395,GPUs=[5,6,7])
    c.write(RUN/'public/ADMISSION.json',result)

def mechanical(runtime):
    task=tasks(24)[0];record=c.record(task)
    batch=runtime.build_edit_batch(record)
    rows=[]
    for kind in ('TT_FIXED_A',):
        e=expert(load_state(initial(task))['expert'],task['seed'],runtime.device)
        if kind=='TT_FIXED_A':
            for parameter in e.parameter_groups()[0]:parameter.requires_grad_(False)
        fixed=[p.detach().clone() for p in e.parameter_groups()[0]]
        opt=optimizer(e,runtime.model);hook=MedTraceLayerHook(runtime.get_module(c.LAYER),e);hook.attach()
        try:
            hook.set_teacher_routing(batch.labels);loss=runtime.compute_loss(batch);assert torch.isfinite(loss);loss.backward()
            params=[p for p in e.parameters() if p.requires_grad];norm=torch.nn.utils.clip_grad_norm_(params,1.)
            assert torch.isfinite(norm) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            assert not any(p.grad is not None for p in runtime.model.parameters());opt.step()
            if kind=='TT_FIXED_A':assert all(torch.equal(a,b) for a,b in zip(fixed,e.parameter_groups()[0]))
            rows.append(dict(kind=kind,loss=float(loss),gradient_norm=float(norm),Base_gradient=False))
        finally:hook.detach()
    peak=torch.cuda.max_memory_allocated();assert 2*peak+4*1024**3 < torch.cuda.get_device_properties(0).total_memory
    c.write(RUN/'public/GPU_MECHANICAL.json',dict(status='PASS',cases=rows,peak_allocated_bytes=peak,two_workers_with_reserve=True,formal_weights_unchanged=True))

def cache():
    from llava.constants import IMAGE_TOKEN_INDEX
    gpu,part=int(os.environ['GPU']),int(os.environ['PARTITION'])
    rows=c.read(RUN/'private/FEATURE_ROWS.json')
    with lease(gpu):
        runtime,_=c.load(gpu)
        route_layer=runtime.target_lock['balancedit']['targets'][0]
        if part==0 and not (RUN/'public/GPU_MECHANICAL.json').exists():mechanical(runtime)
        for i,row in enumerate(rows[part::3]):
            path=feature_path(row)
            if path.exists():continue
            raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
            assert raw['image_sha256']==row['image_sha256']
            with torch.no_grad():
                embeds,att,pos,_=runtime._expand_multimodal(raw_input_ids=raw['input_ids'],attention_mask=raw['attention_mask'],labels=torch.full_like(raw['input_ids'],-100),images=raw['images'])
                masks=s.token_masks(raw['input_ids'][0],embeds.shape[1],None if att is None else att[0],IMAGE_TOKEN_INDEX)
                captured=[];route_captured=[]
                route_handle=runtime.get_module(route_layer).register_forward_pre_hook(lambda _,args:route_captured.append(s.pool(args[0][0],masks)))
                handle=runtime.get_module(c.LAYER).register_forward_pre_hook(lambda _,args:captured.append(s.pool(args[0][0],masks)))
                try:runtime.model(inputs_embeds=embeds,attention_mask=att,position_ids=pos,labels=None,use_cache=False,return_dict=True)
                finally:handle.remove();route_handle.remove()
            assert len(captured)==len(route_captured)==1
            vec,counts=captured[0];route_vec,route_counts=route_captured[0]
            assert counts==route_counts and vec.shape[1]==14336 and route_vec.shape[1]==4096
            if i==0:
                rec=replace(c.record(tasks()[0]),question=row['question'],image_path=Path(c.local_path(row['image_path'])),target='',official_rephrase='')
                reference=runtime.extract_layer_input_key(runtime.build_question_batch(rec),module_path=route_layer,pooling='mean')
                assert torch.equal(route_vec[0],reference),'Original R0 mean key differs'
                done('CACHE_NATIVE_TEST_'+str(part),dict(exact_original_mean=True,modality_counts=counts))
            c.save(path,dict(vectors=vec.cpu(),route_vectors=route_vec.cpu(),counts=counts,binding=dict(route_layer=route_layer,write_layer=c.LAYER,image=row['image_sha256'],question=row['question'],prompt_ids=raw['input_ids'][0].tolist(),runtime=c.read(RUN/'private/RUNTIME_BINDING.json'),generation=runtime.generation_config,all_edits_off=True)))
            if i%20==0:print('FEATURE',part,i,flush=True)
    done('FEATURE_'+str(part))

def alignment(e,t,ts,negatives,x):
    # Cached negative experts are fixed W0; the current expert retains its gradient.
    scores=[]
    for other,a in zip(ts,negatives):
        aa=e.factors()[1] if other['edit_id']==t['edit_id'] else a
        scores.append(s.projection(aa,x[:,1:].reshape(-1,14336)).reshape(5,2).mean(1))
    logits=torch.stack(scores,1)/.1
    compatible=[i for i,o in enumerate(ts) if o['edit_id']==t['edit_id'] or (o['native']['image_sha256'],o['native']['reference'])==(t['native']['image_sha256'],t['native']['reference'])]
    return (torch.logsumexp(logits,1)-torch.logsumexp(logits[:,compatible],1)).mean()

def fit():
    gpu,part=int(os.environ['GPU']),int(os.environ['PARTITION']);stage=os.environ['FIT_STAGE'];ts=tasks(24)
    from m3bench_repro.editors.llava_runtime import seed_everything
    assert stage=='fit', 'CP warmup is canceled'
    with lease(gpu):
        runtime,_=c.load(gpu)
        neg={}
        jobs=[(t,arm) for arm in ARMS if arm!='TT_CE' for t in ts][part::6]
        for t,arm in jobs:
            dest=point(t,arm);receipt=dest.parent/'TRAINING.json'
            if receipt.exists():assert dest.exists();continue
            warm=initial(t)
            state=load_state(warm);e=expert(state['expert'],t['seed'],runtime.device)
            if arm=='TT_FIXED_A':
                for p in e.parameter_groups()[0]:p.requires_grad_(False)
            opt=optimizer(e,runtime.model);rec=c.record(t)
            batches=[runtime.build_edit_batch(rec)]+[runtime.build_edit_batch(replace(rec,question=q)) for q in t['fit_questions']]
            assert all(b.target_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id for b in batches)
            order=list(range(1,5));random.Random(t['seed']).shuffle(order);seed_everything(t['seed'])
            hook=MedTraceLayerHook(runtime.get_module(c.LAYER),e);hook.attach()
            binding=dict(task=w.train_task(t,0,[]),arm=arm,W0=str(warm),W0_state=state['state_hash'],steps=160,fit_order=order,lock=lock(),tt_only_amendment=amendment(),execution=execution('B'),alignment_weight=.5 if arm.endswith('ALIGN') else 0.)
            latest=dest.parent/'latest.pt';curve=[];start=0;began=time.time()
            try:
                if latest.exists():
                    saved=load_state(latest);assert saved['binding']==binding;e.load_state_dict(saved['expert']);opt.load_state_dict(saved['optimizer']);c.restore_rng(saved);curve=saved['curve'];start=saved['step']
                if arm.endswith('ALIGN'):
                    kind=arm.split('_')[0]
                    if kind not in neg:
                        neg[kind]=[expert(load_state(initial(o))['expert'],o['seed'],runtime.device).factors()[1].detach() for o in ts]
                align_x=torch.stack([features(row) for row in train_rows(t)]).to(runtime.device) if arm.endswith('ALIGN') else None
                frozen=[p.detach().clone() for p in e.parameter_groups()[0]] if arm=='TT_FIXED_A' else []
                for step in range(start+1,161):
                    c.budget();opt.zero_grad(set_to_none=True);values=[]
                    for b in (batches[0],batches[order[(step-1)%4]]):
                        hook.set_teacher_routing(b.labels);loss=runtime.compute_loss(b);assert torch.isfinite(loss);(.5*loss).backward();values.append(float(loss.detach()))
                    al=None
                    if arm.endswith('ALIGN'):
                        loss=alignment(e,t,ts,neg[arm.split('_')[0]],align_x);assert torch.isfinite(loss);(.5*loss).backward();al=float(loss.detach())
                    params=[p for p in e.parameters() if p.requires_grad]
                    norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                    assert not any(p.grad is not None for p in runtime.model.parameters());opt.step();assert all(torch.isfinite(p).all() for p in e.parameters())
                    curve.append(dict(step=step,CE=values,alignment=al,preclip_norm=float(norm)))
                    if step%20==0:
                        c.save(latest,dict(binding=binding,expert=e.state_dict(),optimizer=opt.state_dict(),curve=curve,step=step,**c.rng()));print('TRAIN',arm,t['order'],step,flush=True)
                assert all(torch.equal(a,b) for a,b in zip(frozen,e.parameter_groups()[0]))
                c.save(dest,dict(binding=binding,expert=e.state_dict(),step=160,state_hash=c.state_hash(e)))
                c.write(receipt,dict(status='COMPLETE',binding=binding,curve=curve,updates=160,seconds=time.time()-began,parameter_count=sum(p.numel() for p in e.parameters()),trainable=sum(p.numel() for p in e.parameters() if p.requires_grad),Base_gradient=False));latest.unlink()
            finally:hook.detach()
    done(stage.upper()+'_'+str(part))

class Mixture(torch.nn.Module):
    def __init__(self,experts):super().__init__();self.experts=torch.nn.ModuleList(experts);self.ids=[];self.weights=[]
    def residual(self,x):return sum(weight*self.experts[i].residual(x) for i,weight in zip(self.ids,self.weights))


def route_setup(ts,es,route,device):
    if route=='R0':
        from m3bench_repro.editors.routing import MemoryRouter
        entries=[]
        for t in ts:
            p=initial(t).parent.parent/'ROUTER.pt';entries+=load_state(p)['entries']
        return MemoryRouter.from_state(dict(distance='euclidean',entries=entries),device=device),None
    keys=[torch.stack([features(row,route=not route.startswith('INTRINSIC')) for row in train_rows(t)]).to(device) for t in ts]
    aa=[e.factors()[1].detach() for e in es];threshold=[]
    for a,k in zip(aa,keys):
        if route.startswith('INTRINSIC'):threshold.append(min(float(s.input_score(a,x,route.endswith('MODAL'))) for x in k))
        else:threshold.append(min(float(s.key_score(k[i],torch.cat([k[:i],k[i+1:]]),route=='MODAL').max()) for i in range(5)))
    return (keys,aa),torch.tensor(threshold,device=device)


def routing(row,route,setup,threshold,ts,device,mix):
    x=features(row,route=not route.startswith('INTRINSIC')).to(device)
    if route=='R0':
        from m3bench_repro.editors.routing import decision_as_json
        decision=setup.route(x[0]);ids=[next(i for i,t in enumerate(ts) if t['edit_id']==decision.logical_edit_id)] if decision.activated else []
        return ids,[1.] if ids else [],decision_as_json(decision)
    keys,aa=setup
    scores=torch.stack([s.input_score(a,x,route.endswith('MODAL')) for a in aa]) if route.startswith('INTRINSIC') else torch.stack([s.key_score(x,k,route=='MODAL').max() for k in keys])
    ids,weights,adjusted=s.select(scores,threshold,mix)
    return ids,weights,dict(activated=bool(ids),logical_edit_id=ts[ids[0]]['edit_id'] if ids else None,selected_experts=[ts[i]['edit_id'] for i in ids],mix_weights=weights,maximum_adjusted=float(adjusted.max()),scores=scores.tolist(),rule=route)


def evaluate():
    gpu,part=int(os.environ['GPU']),int(os.environ['PARTITION']);stage=os.environ['EVAL_STAGE'];n=146 if stage=='A' else 24;ts=tasks(n)
    jobs=[('W0',r,False) for r in A_ROUTES] if stage=='A' else [(a,r,False) for a in ARMS for r in ROUTES] if stage=='B' else [(a,'INTRINSIC_MODAL',True) for a in ('TT_ALIGN',)]
    shard_count=3 if stage=='A' else 6
    reuse=c.read(RUN/'private/REUSE_OUTPUTS.json')
    rows=queries(n);active={(t['native']['image_sha256'],t['native']['question']) for t in ts}
    with lease(gpu):
        runtime,bindings=c.load(gpu);teachers=tr.teachers_for(runtime,c.read(BASE/'private/U_ROLES.json')['CHECK']);tm={'U_'+c.digest(x[4]):x for x in teachers}
        for arm,route,mix in jobs:
            label=stage+'_'+arm+'_'+route+('_MIX' if mix else '')
            es=[expert(load_state(point(t,arm))['expert'],t['seed'],runtime.device).requires_grad_(False) for t in ts]
            mixture=Mixture(es);hook=MedTraceLayerHook(runtime.get_module(c.LAYER),mixture);hook.attach()
            setup,threshold=route_setup(ts,es,route,runtime.device)
            weights={t['edit_id']:dict(path=str(point(t,arm)),hash=load_state(point(t,arm))['state_hash']) for t in ts}
            phase=dict(mode='bank',prefix=n,arm=label,node=0 if n==146 else 160,slot=0,weights=weights,execution=execution(stage),paper_lock=lock(),tt_only_amendment=amendment() if stage!='A' else None,route=route,mix=mix,thresholds=threshold.tolist() if threshold is not None else None,calibration='five training inputs only; fixed pseudoinverse rtol1e-6; nonnegative adjusted score tolerance1e-6')
            c.write(RUN/'private/bank_bindings'/(c.digest(phase)+'.json'),phase)
            try:
                for qid,row in list(rows.items())[part::shard_count]:
                    c.budget();dest=RUN/'private/outputs'/label/(c.digest(qid)+'.json')
                    if dest.exists():assert c.read(dest)['binding']['phase']==phase;continue
                    ids,mw,decision=routing(row,route,setup,threshold,ts,runtime.device,mix);mixture.ids,mixture.weights=ids,mw;hook.clear_request_routing()
                    old=reuse.get(label+'|'+qid)
                    if old:
                        full=old['payload'];assert full['query_id']==qid and full['image_sha256']==row['image_sha256'] and full['question']==row['question'] and full['generation']==runtime.generation_config
                        out=full['output'];b=bindings[row['opaque_Base_id']] if qid not in tm else {k:full[k] for k in ('question','reference','image_sha256','image_path','prompt_ids','attention_mask','runtime','generation')}
                        assert all(b[k]==full[k] for k in ('question','reference','image_sha256','image_path','prompt_ids','attention_mask','runtime','generation'))
                        kl=old.get('U_KL');same=old.get('Base_token_consistency');seconds=0.
                    else:
                        raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
                        if qid in tm:
                            cache=tm[qid][-1];b=dict(question=row['question'],reference=row['reference'],image_sha256=raw['image_sha256'],image_path=row['image_path'],prompt_ids=raw['input_ids'][0].tolist(),attention_mask=raw['attention_mask'][0].tolist(),runtime=next(iter(bindings.values()))['runtime'],generation=runtime.generation_config)
                        else:
                            b=bindings[row['opaque_Base_id']];assert raw['image_sha256']==b['image_sha256'] and raw['input_ids'].tolist()==[b['prompt_ids']] and raw['attention_mask'].tolist()==[b['attention_mask']]
                        began=time.time()
                        with torch.inference_mode():
                            if ids:
                                with hook.generation_request():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                            else:g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                        out=dict(raw_answer=g.decoded_text,raw_token_ids=list(g.raw_token_ids));seconds=time.time()-began;kl=same=None
                        if qid in tm:
                            kwargs,labels,mask,logp,_,cache=tm[qid];hook.clear_request_routing()
                            if ids:hook.set_teacher_routing(labels)
                            with torch.no_grad():kl=float(tr.full_vocab_kl(runtime.model(**kwargs).logits[mask],logp))
                            same=out['raw_token_ids']==cache['tokens'];hook.clear_request_routing()
                    bind=dict(input=row,judge_input=b,phase=phase,arm=label,mode='bank_R0',prefix=n,owner_order=ts[-1]['order'],panel=row.get('role','PANEL'))
                    c.write(dest,dict(binding=bind,R0=out,route=decision,effective_expert=ts[ids[0]]['edit_id'] if ids else None,selected_experts=[ts[i]['edit_id'] for i in ids],mix_weights=mw,research_lock=lock(),seconds=seconds,TT_parameters_per_expert=sum(p.numel() for p in es[0].parameters()),U_KL=kl,Base_token_consistency=same,active_target=(row['image_sha256'],row['question']) in active,diagnostic_only=False,reused_output=old['path'] if old else None,reused_output_binding=old['original_output_binding'] if old else None))
                print('EVAL_CONDITION',label,part,flush=True)
            finally:hook.detach()
    done('EVAL_'+stage+'_'+str(part))


def controller():
    import pipeline as p
    assert c.read(RUN/'private/TT_ONLY_AMENDMENT.json')['CP_enabled'] is False
    assert all((RUN/'private'/('FEATURE_'+str(i)+'.json')).exists() for i in range(3))
    progress('A_FROZEN146_ROUTING_TT_ONLY_TAIL')
    # Adopt the recorded, already-running TT workers without regenerating A.
    while True:
        pending=[i for i in range(3) if not (RUN/'private'/('EVAL_A_'+str(i)+'.json')).exists()]
        if not pending:break
        c.budget()
        for i in pending:
            st=c.read(RUN/'private'/('START_CHAIN_paper_eval_'+str(i)+'.json'))
            proc=Path('/proc/'+str(st['pid'])+'/stat')
            alive=proc.exists() and proc.read_text().split()[21]==st['start_ticks'] and proc.read_text().split()[2]!='Z'
            assert alive or (RUN/'private'/('EVAL_A_'+str(i)+'.json')).exists(), 'Recorded A worker failed; inspect before recovery'
        time.sleep(5)
    p.wait([p.launch('paper_queue.py','paper_ingest')])
    progress('B_TT_FIXED_TRAINING_COMPARISONS')
    os.environ['FIT_STAGE']='fit';p.wait([p.launch('paper.py','paper_fit',5+i%3,i) for i in range(6)])
    progress('B_TT_FINAL24_BANKS')
    os.environ['EVAL_STAGE']='B';p.wait([p.launch('paper.py','paper_eval',5+i%3,i) for i in range(6)])
    p.wait([p.launch('paper_queue.py','paper_ingest')])
    progress('D_SPARSE_MIXTURE')
    os.environ['EVAL_STAGE']='D';p.wait([p.launch('paper.py','paper_eval',5+i%3,i) for i in range(6)])
    done('GENERATION_COMPLETE');p.wait([p.launch('paper_queue.py','paper_ingest')])
    removed=[]
    for folder in ('weights','features','teacher'):
        for path in (RUN/'private'/folder).rglob('*.pt'):
            assert not path.is_symlink() and path.resolve().is_relative_to((RUN/'private'/folder).resolve())
            removed.append(dict(path=str(path),bytes=path.stat().st_size));path.unlink()
    done('DELETION',dict(files=removed,all_consumers_complete=True,bindings_durable=True))
    progress('ASTRA_SCORING')
    root=RUN/'private/judge_paper_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(30)
    p.wait([p.launch('paper_report.py','paper_report')])

if __name__=='__main__':
    try:{'paper_prepare':prepare,'paper_cache':cache,'paper_fit':fit,'paper_eval':evaluate,'paper_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        if os.environ['ACTION']=='paper_controller':c.write(RUN/'private/CONTROLLER_FAILURE.json',dict(error=repr(error),epoch=time.time()))
        c.write(RUN/'private'/('FAILURE_'+os.environ['ACTION']+'_'+os.environ.get('PARTITION','all')+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),retry=False));raise
