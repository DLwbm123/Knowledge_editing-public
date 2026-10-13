"""One bounded check covers update, diagnostic, replay and TT expansion semantics."""
import copy
import torch
import updates
from geometry import selfcheck,flat,assign,match
from direction import clone,embed,step as direction_step


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def expansion_check(runtime,record,state,expanded,seed,dest):
    from methods.medtrace import MedTraceLayerHook
    from audit import write
    import stage_worker as w
    batch=runtime.build_edit_batch(record);raw=runtime.adapter.prepare_inputs(record.image_path,record.question,None)
    original=clone(state,seed,runtime.device).requires_grad_(True);expanded=expanded.to(runtime.device).requires_grad_(True)
    hs=[];gs=[];logits=[];residual=[];gradients=[]
    for e in (original,expanded):
        hook=MedTraceLayerHook(runtime.get_module(w.LAYER),e);hook.attach()
        captured=[]
        def take(_module,args):captured.append(args[0].detach().float()[:, -1:, :].clone())
        handle=runtime.get_module(w.LAYER).register_forward_pre_hook(take)
        def capture_logits(_module,_args,output):
            logits.append(output.logits[:,:-1][batch.labels[:,1:]!=-100].detach().float().clone())
        lh=runtime.model.register_forward_hook(capture_logits)
        try:
            # build_edit_batch exposes model kwargs in the original runtime.
            hook.set_teacher_routing(batch.labels)
            loss=runtime.compute_loss(batch);loss.backward()
            lh.remove()
            with torch.no_grad():residual.append(e.residual(captured[0]).detach())
            hs.append(float(loss.detach()));gradients.append({k:p.grad.detach().clone() for k,p in e.named_parameters()})
            out,diag=w.generate(runtime,hook,raw,True);gs.append(out)
        finally:lh.remove();handle.remove();hook.detach()
    assert torch.allclose(residual[0],residual[1],rtol=2e-5,atol=2e-5) and abs(hs[0]-hs[1])<=2e-5 and gs[0]==gs[1]
    assert len(logits)==2 and torch.allclose(logits[0],logits[1],rtol=1e-3,atol=1e-2)
    gr=gradients[1]
    new_norms=([float(gr['G1'][:,:,4:].norm()),float(gr['G3'][:,:,4:].norm())] if expanded.G1.shape[2]==8 else [float(gr['G2'][:,:,4:].norm())])
    assert all(n>0 for n in new_norms),'New TT pathway has no opening gradient'
    write(dest,dict(status='PASS',residual_max_abs=float((residual[0]-residual[1]).abs().max()),logit_max_abs=float((logits[0]-logits[1]).abs().max()),reference_loss_difference=abs(hs[0]-hs[1]),generated_tokens_equal=True,opening_gradient_norms=new_norms,parameters=sum(p.numel() for p in expanded.parameters()),temporary_AB_bytes=sum(f.numel()*f.element_size() for f in expanded.factors())))


def check(runtime,t,cfg):
    import stage_worker as w
    from audit import write
    from methods.medtrace import MedTraceLayerHook
    from structures import optimizer_for,TT4
    from scripts.medtrace import stage18_cfact as cf
    from scripts.medtrace.run_selective_write import save
    from m3bench_repro.editors.llava_runtime import seed_everything
    synthetic=selfcheck();record=w.legacy.record_for(t);state=w.load_state(w.path_for(t,'TT88',0,warm=True))['expert']
    teachers,batches,hb,fit,schedule=w.training_inputs(runtime,t,cfg,record)
    e=clone(state,t['seed'],runtime.device).requires_grad_(True);opt=optimizer_for(e,runtime.model);hook=MedTraceLayerHook(runtime.get_module(w.LAYER),e);hook.attach()
    raw=runtime.adapter.prepare_inputs(record.image_path,record.question,None)
    try:
        seed_everything(t['seed']);initial=flat(e);rng=updates.rng();hs=updates.hook_state(hook)
        def restore():assign(e,initial);opt.state.clear();updates.restore_rng(rng);updates.restore_hook(hook,hs)
        def update():return w.base.logged_update(runtime,hook,e,opt,batches[0],batches[fit[0]],teachers[0],(hb[schedule[0]],w.H[t['edit_id']][schedule[0]]),1.,False)
        # Original update versus diagnostic instrumentation: exact actual Adam state.
        old=cf.update(runtime,hook,e,opt,batches[0],batches[fit[0]],teachers[0],(hb[schedule[0]],w.H[t['edit_id']][schedule[0]]),extra_weight=1.)
        reference=flat(e);refopt=copy.deepcopy(opt.state_dict());restore()
        activations=updates.capture_training_activations(runtime,hook,batches,teachers,w.LAYER)
        updates.diagnostic(runtime,hook,e,[batches[0],batches[fit[0]]],teachers,hb[schedule[0]],1.,activations)
        update();assert torch.equal(reference,flat(e)) and equal_tree(refopt,opt.state_dict())
        restore();before_opt=copy.deepcopy(opt.state_dict())
        # Shared guard must explicitly reject finite infeasibility and preserve Adam.
        item=updates.guarded_candidate(e,opt,update,lambda:[1.]*6,[0.]*6)
        assert not item['guard']['accepted'] and torch.equal(flat(e),initial) and equal_tree(before_opt,opt.state_dict());restore()
        # Direction-policy rejection with a deliberately impossible nonlinear check.
        original_protection=updates.protection
        updates.protection=lambda *args:[1e6]*6
        try:item=direction_step(runtime,hook,e,opt,update,batches,teachers,hb[schedule[0]],[1e5]*6,activations,'DIR')
        finally:updates.protection=original_protection
        assert not item['accepted'] and torch.equal(flat(e),initial) and equal_tree(before_opt,opt.state_dict()) and equal_tree(updates.rng(),rng) and equal_tree(updates.hook_state(hook),hs)
        restore()
        # Wide constraints admit exact unprojected candidate and optimizer state.
        item=direction_step(runtime,hook,e,opt,update,batches,teachers,hb[schedule[0]],[1e5]*6,activations,'DIR')
        assert item['accepted'] and torch.equal(flat(e),reference) and equal_tree(refopt,opt.state_dict())
        point=w.RUN/'private/mechanical/latest.pt'
        save(point,dict(binding=dict(check='replay'),expert=e.state_dict(),optimizer=opt.state_dict(),step=1,curve=[item],hook_state=updates.hook_state(hook),**cf.rng_state()))
        expected=(flat(e),copy.deepcopy(opt.state_dict()),updates.rng(),updates.hook_state(hook));restore()
        w.base.resume(point,dict(check='replay'),e,opt);updates.restore_hook(hook,torch.load(point,map_location='cpu',weights_only=True)['hook_state'])
        assert torch.equal(flat(e),expected[0]) and equal_tree(opt.state_dict(),expected[1]) and equal_tree(updates.rng(),expected[2]) and equal_tree(updates.hook_state(hook),expected[3])
        point.unlink();restore()
        # A target from alpha=.5 verifies nonlinear matching and moment commitment.
        update();d=flat(e)-initial
        with torch.no_grad():assign(e,initial);fb=e.residual(activations).detach();assign(e,initial+.5*d);target=float((e.residual(activations)-fb).norm())
        restore();item=direction_step(runtime,hook,e,opt,update,batches,teachers,hb[schedule[0]],[1e5]*6,activations,'MATCH',target)
        assert item['match']['relative_error']<1e-4 and equal_tree(opt.state_dict(),refopt)
        restore();item=direction_step(runtime,hook,e,opt,update,batches,teachers,hb[schedule[0]],[1e5]*6,activations,'MATCH',0.)
        assert not item['nonzero'] and torch.equal(flat(e),initial) and equal_tree(opt.state_dict(),before_opt) and equal_tree(updates.rng(),rng) and equal_tree(updates.hook_state(hook),hs)
        a,_=w.generate(runtime,hook,raw,True);b,_=w.generate(runtime,hook,raw,True);assert a==b
        zero=TT4(t['seed'],8,8).to(runtime.device);hook.expert=zero
        off,_=w.generate(runtime,hook,raw,False);on,_=w.generate(runtime,hook,raw,True);assert off==on
        hook.expert=e;assert set(dict(e.named_parameters()))=={'G1','G2','G3','G4'} and not any(p.requires_grad for p in runtime.model.parameters())
    finally:hook.detach()
    state44=w.load_state(w.path_for(t,'TT44',0,warm=True))['expert']
    for kind in ('OUTER','MIDDLE'):expansion_check(runtime,record,state44,embed(state44,kind,t['seed']),t['seed'],w.RUN/'private/mechanical'/f'EMBED_{kind}.json')
    write(w.RUN/'private/GPU_MECHANICAL.json',dict(status='PASS',synthetic=synthetic,actual_Adam_parity=True,diagnostic_neutrality=True,reject_full_restore=True,accept_candidate_moments_once=True,match_scale_moments_once=True,zero_match_full_restore=True,save_reload_TT_optimizer_RNG_hook=True,repeated_generation=True,zero_Base=True,TT_only=True,embedding_qualifications=True,formal_continuation_started=False))
