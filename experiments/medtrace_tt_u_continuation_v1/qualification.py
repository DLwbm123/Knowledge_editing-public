"""Finite CPU checks followed by one fixed-editor real GPU smoke."""
import ast
import copy
import io
from pathlib import Path
import torch
from common import RUN,LAYER,read,write,rng,restore_rng,diagnostic_scope,record,state_hash
from structures import TT4,optimizer_for
from train import clone,update,teachers_for,activations,grad
from roles import validate_roles

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def cpu_check():
    torch.set_num_threads(2)
    for name in ('train.py','common.py','roles.py'):
        tree=ast.parse((RUN/'private/tools'/name).read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        assert not any('stage18' in x or 'legacy_worker' in x or 'direction' in x for x in imports)
        assert not any(isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value in ('H','H_fit','H_AVAILABLE.json','H_eval') for n in ast.walk(tree))
    assert not (RUN/'private/H_AVAILABLE.json').exists()
    roles=read(RUN/'private/U_ROLES.json');validate_roles(roles,set())
    if roles['FIT']:
        bad=copy.deepcopy(roles);bad['CAL']=[dict(bad['FIT'][0],role='CAL')]
        try:validate_roles(bad,set())
        except AssertionError:pass
        else:raise AssertionError('Cross-role duplicate accepted')
        try:validate_roles(roles,{(roles['FIT'][0]['dataset'],roles['FIT'][0]['image_id'])})
        except AssertionError:pass
        else:raise AssertionError('Future/evaluation source accepted')
    from methods.medtrace.selective_write import full_vocab_kl,predictor_mask
    logits=torch.tensor([[.5,1.,-.5]],requires_grad=True);teacher=torch.tensor([[1.,-.2,.4]]).log_softmax(-1)
    expected=(teacher.exp()*(teacher-logits.log_softmax(-1))).sum();assert torch.equal(full_vocab_kl(logits,teacher),expected)
    labels=torch.tensor([[-100,-100,2,3]]);attention=torch.ones_like(labels);mask=predictor_mask(labels,attention);assert mask.tolist()==[[False,True,True,False]]
    base=torch.nn.Linear(1,1).requires_grad_(False);e=TT4(17,8,8);assert sum(p.numel() for p in e.parameters())==7168 and set(e.state_dict())=={'G1','G2','G3','G4'}
    x=torch.randn(2,14336);target=torch.randn(2,4096);opt=optimizer_for(e,base)
    def step(e,o):o.zero_grad();(e.residual(x)-target).square().mean().backward();torch.nn.utils.clip_grad_norm_(e.parameters(),1);o.step()
    step(e,opt);buf=io.BytesIO();torch.save(dict(e=e.state_dict(),o=opt.state_dict()),buf);buf.seek(0);s=torch.load(buf,weights_only=True)
    ee=TT4(17,8,8);oo=optimizer_for(ee,base);ee.load_state_dict(s['e']);oo.load_state_dict(s['o']);step(e,opt);step(ee,oo);assert equal(e.state_dict(),ee.state_dict()) and equal(opt.state_dict(),oo.state_dict())
    assert all(p.grad is None for p in base.parameters()) and all(p.grad is not None for p in e.parameters())
    write(RUN/'private/CPU_MECHANICAL.json',dict(status='PASS',no_optional_payload_files=True,no_optional_module_imports=True,role_overlap_rejected=True,future_eval_source_rejected=True,KL_direction='Base||student',predictor_mask=True,save_load_next_step=True,parameter_count=7168,source_shapes_unchanged=True))

def gpu_check(runtime,t):
    from methods.medtrace.core import MedTraceLayerHook
    from train import full_vocab_kl
    from common import save
    torch.set_num_threads(4)
    roles=read(RUN/'private/U_ROLES.json');teacher=teachers_for(runtime,roles['FIT'][:1]);cal=teachers_for(runtime,roles['CAL'])
    p=Path(read(RUN/'private/ASSET_REUSE_AUDIT.json')['W0'][0]['path']);init=torch.load(p,map_location='cpu',weights_only=True)['expert'];rec=record(t)
    from dataclasses import replace
    batches=[runtime.build_edit_batch(rec)]+[runtime.build_edit_batch(replace(rec,question=q)) for q in t['fit_questions']]
    for b in batches:assert b.target_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id
    frozen=[(p,p._version,p.data_ptr()) for p in runtime.model.parameters()]
    e=clone(init,t['seed'],runtime.device);o=optimizer_for(e,runtime.model);h=MedTraceLayerHook(runtime.get_module(LAYER),e);h.attach();repeat=[]
    try:
        before=rng();hs=dict(enabled=h.enabled,token_mask=h.token_mask);x=activations(runtime,h,batches,teacher)
        assert equal(before,rng()) and h.enabled==hs['enabled'] and h.token_mask is None and all(p.grad is None for p in e.parameters())
        # CE_ONLY skipping the optional forward has identical next CE update semantics.
        ref=clone(init,t['seed'],runtime.device);ro=optimizer_for(ref,runtime.model)
        if teacher:
            with diagnostic_scope(h),torch.no_grad():
                kwargs,labels,mask,logp,*_=teacher[0];h.set_teacher_routing(labels);runtime.model(**kwargs)
        a=update(runtime,h,e,o,batches[0],batches[1]);after=rng();h.expert=ref;restore_rng(before);b=update(runtime,h,ref,ro,batches[0],batches[1]);assert equal(e.state_dict(),ref.state_dict()) and equal(o.state_dict(),ro.state_dict()) and equal(after,rng())
        assert 'U' not in a['terms'];h.expert=e
        state=copy.deepcopy(e.state_dict());opt=copy.deepcopy(o.state_dict());r=rng()
        for _ in range(2):activations(runtime,h,batches,teacher)
        assert equal(r,rng()) and equal(state,e.state_dict()) and equal(opt,o.state_dict())
        chosen=teacher[0] if teacher else None;a=update(runtime,h,e,o,batches[0],batches[1],chosen)
        assert not teacher or a['terms']['U']['weighted_gradient_norm']>0,'Real U update contribution must be evidenced'
        expected=copy.deepcopy(e.state_dict());expectedopt=copy.deepcopy(o.state_dict());expectedrng=rng()
        path=RUN/'private/smoke/latest.pt';save(path,dict(e=state,o=opt,**r));loaded=torch.load(path,map_location='cpu',weights_only=True)
        e.load_state_dict(loaded['e']);o.load_state_dict(loaded['o']);restore_rng(loaded);update(runtime,h,e,o,batches[0],batches[1],chosen)
        assert equal(expected,e.state_dict()) and equal(expectedopt,o.state_dict()) and equal(expectedrng,rng())
        with diagnostic_scope(h),torch.no_grad():
            e.load_state_dict(init)
            for _ in range(2):
                vals=[]
                for kwargs,labels,mask,logp,*_ in cal:h.set_teacher_routing(labels);vals.append(float(full_vocab_kl(runtime.model(**kwargs).logits[mask],logp)))
                repeat.append(vals)
        error=max([abs(a-b) for a,b in zip(*repeat)],default=0.)
        h.clear_request_routing();raw=runtime.adapter.prepare_inputs(rec.image_path,rec.question,None)
        with torch.inference_mode(),h.generation_request():out=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
        saved=RUN/'private/smoke/cores.pt';save(saved,dict(expert=e.state_dict(),step=0,state_hash=state_hash(e)));reloaded=torch.load(saved,map_location='cpu',weights_only=True);e.load_state_dict(reloaded['expert'])
        with torch.inference_mode(),h.generation_request():again=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
        assert list(out.raw_token_ids)==list(again.raw_token_ids)
        assert all(p._version==v and p.data_ptr()==ptr and p.grad is None and not p.requires_grad for p,v,ptr in frozen)
        write(RUN/'private/GPU_MECHANICAL.json',dict(status='PASS',order=t['order'],no_payload_file_dependency=True,Base_frozen=True,CE_ONLY_no_U_grad=True,U_real_gradient=bool(teacher),teacher_graph_detached=True,teacher_all_edits_off=True,EOS_and_prefix_mask=True,diagnostic_rng_hook_grad_optimizer_parity=True,save_load_next_update=True,save_load_generation=True,numerical_repeat_error=error,CAL_groups=len(cal),teacher_generation_untruncated=True,smoke_not_formal=True))
        write(RUN/'public/P0_CHECKS.json',read(RUN/'private/GPU_MECHANICAL.json'))
    finally:h.detach()
