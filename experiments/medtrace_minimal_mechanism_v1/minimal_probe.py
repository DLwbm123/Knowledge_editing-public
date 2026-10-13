"""Finite probes of retained states. No training continuation or heldout reads."""
import os
import time
import traceback
from pathlib import Path
import torch
import scope as s
import minimal_math as mm

c,p,q,old,RUN = s.c,s.p,s.q,s.old,s.RUN
PARENT = Path(os.environ['DIAGNOSTIC_PARENT'])
COUNTS = dict(forwards=0,backwards=0,updates=0,generations=0,Judge=0)


def forward_count(*_):
    c.budget()
    COUNTS['forwards'] += 1
    assert COUNTS['forwards'] <= 900


def run():
    from m3bench_repro.editors.llava_runtime import seed_everything
    assert not (RUN/'START.json').exists(), 'No implicit retry'
    c.write(RUN/'START.json',dict(epoch=time.time(),pid=os.getpid()))
    checks=mm.selfcheck(); tasks=s.tasks(); assert [t['order'] for t in tasks]==list(range(1,9))
    rows={t['order']:s.source(t) for t in tasks}; assert all(len(v)==7 for v in rows.values())
    states={t['order']:p.load_state(PARENT/'private/candidates/C'/f"{t['order']}.pt")['states'] for t in tasks}
    assert all(set(v)=={'BASE','FINAL'} for v in states.values())
    # Resolve donor identities before reading any model prediction.
    donors={}
    for t in tasks:
        owner=t['order']
        for i,row in enumerate(rows[owner]):
            donors[owner,i]=next(j for offset in range(1,8) for j in [1+(owner-1+offset)%8]
                if rows[j][0]['image_path']!=row['image_path'] and rows[j][0].get('image_sha256')!=row.get('image_sha256'))
    c.write(RUN/'private/INPUT_BINDINGS.json',dict(rows=rows,donors={f'{a}:{b}':v for (a,b),v in donors.items()}))
    c.write(RUN/'public/ADMISSION.json',dict(status='PASS',selfcheck=checks,owners=8,source_questions=56,
        states=['BASE','FINAL'],representations=list(mm.MODES),expected_forwards=864,expected_backwards=144,
        expected_ephemeral_steps=48,new_checkpoints=0,new_generations=0,new_Judge=0))
    gpu=6
    with p.lease(gpu):
        runtime,_=c.load(gpu); frozen={}; spans={}; image_ids={}
        for t in tasks:
            for i,row in enumerate(rows[t['order']]):
                batch=runtime.build_edit_batch(s.record(t,row))
                if row.get('image_sha256'): assert batch.image_sha256==row['image_sha256']
                image_ids[t['order'],i]=batch.image_sha256
                frozen[t['order'],i]=old.cpu(batch)
                spans[t['order'],i]=(batch.image_token_start,batch.image_token_end)
        assert all(image_ids[owner,i]!=image_ids[donor,0] for (owner,i),donor in donors.items())
        old.fp32(runtime); handle=runtime.model.register_forward_pre_hook(forward_count)
        try:
            for t in tasks:
                owner=t['order']; batches=[old.batch_on(runtime,frozen[owner,i]) for i in range(7)]
                expert=p.expert(states[owner]['BASE'],t['seed'],runtime.device)
                hook=q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert); hook.attach()
                try:
                    for state_name in ('BASE','FINAL'):
                        before={k:v.to(runtime.device) for k,v in states[owner][state_name].items()}
                        expert.load_state_dict(before); references=[q.logits(runtime,hook,b) for b in batches]
                        repeat,target=q.logits(runtime,hook,batches[0]); assert torch.equal(target,references[0][1])
                        noise=mm.metrics(references[0][0],repeat,target); visual=[]
                        for i,(ref,target) in enumerate(references):
                            conditions={'ORIGINAL':mm.metrics(ref,ref,target)}
                            donor=donors[owner,i]; a,b=spans[donor,0]
                            donor_emb=frozen[donor,0][0]['inputs_embeds'][:,a:b]
                            for mode in ('SWAPPED_VISUAL','ZERO_VISUAL'):
                                value=mm.visual_input(frozen[owner,i],spans[owner,i],mode,donor_emb)
                                logits,y=q.logits(runtime,hook,old.batch_on(runtime,value)); assert torch.equal(y,target)
                                conditions[mode]=mm.metrics(ref,logits,target)
                            visual.append(dict(slot=i,role=rows[owner][i]['audit_role'],donor_owner=donor,conditions=conditions))
                        gauge_rows=[]; identity_delta=None
                        for mode in mm.MODES:
                            seed_everything(t['seed']); start=mm.gauge(before,mode); expert.load_state_dict(start)
                            if mode=='IDENTITY': parity=mm.metrics(references[0][0],references[0][0],references[0][1])
                            else:
                                logits,y=q.logits(runtime,hook,batches[0]); assert torch.equal(y,references[0][1])
                                torch.testing.assert_close(logits,references[0][0],rtol=1e-5,atol=2e-4)
                                parity=mm.metrics(references[0][0],logits,y)
                            # An operator check also covers input directions absent from the native question.
                            probe_x=torch.randn(3,14336,generator=torch.Generator().manual_seed(20261011),dtype=torch.float64)
                            b0,a0=mm.factors({k:v.detach().cpu().double() for k,v in before.items()})
                            b1,a1=mm.factors({k:v.detach().cpu().double() for k,v in start.items()})
                            torch.testing.assert_close((probe_x@a1.T)@b1.T,(probe_x@a0.T)@b0.T,rtol=1e-10,atol=1e-10)
                            opt=p.optimizer(expert,runtime.model); opt.zero_grad(set_to_none=True)
                            for i,weight in ((0,.25),(1,.25),(5,.5)):
                                hook.set_teacher_routing(batches[i].labels); loss=runtime.compute_loss(batches[i])
                                assert torch.isfinite(loss); (weight*loss).backward(); COUNTS['backwards']+=1
                                assert COUNTS['backwards']<=160
                            norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
                            assert torch.isfinite(norm) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in expert.parameters())
                            opt.step(); groups=q.d.apply_direction(opt,{id(v):start[k] for k,v in expert.named_parameters()},'RAW')
                            COUNTS['updates']+=1; assert COUNTS['updates']<=48
                            after=old.snapshot(expert); update=mm.delta(start,after)
                            if mode=='IDENTITY': identity_delta=update
                            values=[]
                            for batch,(ref,target) in zip(batches,references):
                                logits,y=q.logits(runtime,hook,batch); assert torch.equal(y,target)
                                values.append(mm.metrics(ref,logits,target))
                            gauge_rows.append(dict(mode=mode,before_native_parity=parity,gradient_norm_before_clip=float(norm),
                                groups=groups,operator_update=mm.compare_maps(update,identity_delta),source=values,
                                native_FIT_gain=-.5*values[0]['delta_NLL']-.125*sum(v['delta_NLL'] for v in values[1:5]),
                                GFIT_gain=-.5*sum(v['delta_NLL'] for v in values[5:])))
                            expert.load_state_dict(before)
                        assert not any(v.grad is not None for v in runtime.model.parameters())
                        c.write(RUN/'public/rows'/f'{owner}_{state_name}.json',dict(owner=owner,state=state_name,
                            repeat_noise=noise,visual=visual,gauge=gauge_rows,counts_so_far=dict(COUNTS)))
                        print('PROBE_COMPLETE',owner,state_name,COUNTS,flush=True)
                finally:
                    hook.detach(); del expert,batches
        finally: handle.remove()
        assert COUNTS==dict(forwards=864,backwards=144,updates=48,generations=0,Judge=0), COUNTS
    c.write(RUN/'public/COMPLETE.json',dict(status='COMPLETE',counts=COUNTS,
        GPU_hours=c.used()/3600,source_only=True,historical_exposure=True,independent_confirmation=False,
        fresh_optimizer=True,persistent_checkpoints=0,epoch=time.time()))


if __name__=='__main__':
    try:
        run(); c.write(RUN/'EXIT.json',dict(exit_code=0,epoch=time.time()))
    except BaseException as error:
        c.write(RUN/'FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),counts=COUNTS,epoch=time.time()))
        c.write(RUN/'EXIT.json',dict(exit_code=1,epoch=time.time())); raise
