"""One deep TT expert, original source/FIT optimizer plus admitted replay CE."""
import copy
import math
from pathlib import Path
import torch
import trajectory
from train_pair import state_digest, save_new


def train(runtime,r,task,source_batches,replay_batches,role_receipt,output,budget):
    if role_receipt.get('status')!='FIT_CONTENT_AUTHORIZED' or task['order'] not in role_receipt.get('owners',[]):
        raise ValueError('Independent FIT content admission missing')
    if len(source_batches)!=5 or not replay_batches:
        raise ValueError('Source and legal replay batches required')
    if any(p.requires_grad for p in runtime.model.parameters()): raise ValueError('Base not frozen')
    if len(runtime.model.model.layers)!=32: raise ValueError('Unexpected decoder')
    path='model.layers.30.mlp.down_proj'; module=runtime.get_module(path)
    if (module.in_features,module.out_features)!=(14336,4096): raise ValueError('Incompatible deep module')
    from m3bench_repro.editors.llava_runtime import seed_everything
    seed_everything(task['seed'])
    initial=r.p.tr.TT4(task['seed'],8,8).state_dict()
    if sum(t.numel() for t in initial.values())!=7168 or torch.count_nonzero(initial['G1']):
        raise ValueError('TT initialization mismatch')
    initial={k:v.detach().cpu().clone() for k,v in initial.items()}
    expert=r.p.expert(copy.deepcopy(initial),task['seed'],runtime.device)
    if state_digest(expert.state_dict())!=state_digest(initial): raise ValueError('initialization mismatch')
    versions={name:p._version for name,p in runtime.model.named_parameters()}
    hook=r.q.scoped.replay.MedTraceLayerHook(module,expert);hook.attach()
    try:
        for b in source_batches: budget();r.q.logits(runtime,hook,b,verify_base=True)
        optimizer=r.p.optimizer(expert,runtime.model);curve=[]
        for step in range(1,161):
            budget();before=r.old.snapshot(expert)
            _,source_losses=trajectory.source_gradients(r,runtime,hook,expert,source_batches,step)
            # Replay batches stay on the CPU between steps; only the scheduled
            # batch is transferred to the model device.
            b=r.old.batch_on(runtime,replay_batches[(step-1)%len(replay_batches)])
            hook.set_teacher_routing(b.labels)
            loss=runtime.compute_loss(b)
            if not torch.isfinite(loss): raise ValueError('nonfinite replay CE')
            loss.backward();r.COUNTS['backwards']+=1
            norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
            if not torch.isfinite(norm): raise ValueError('nonfinite gradient')
            optimizer.step();r.COUNTS['updates']+=1
            after=r.old.snapshot(expert)
            if step==1 and all(torch.equal(after[k],before[k]) for k in before): raise ValueError('zero first update')
            delta=math.sqrt(sum(float((after[k].double()-before[k].double()).square().sum()) for k in before))
            curve.append(dict(step=step,source_nll=source_losses,replay_nll=float(loss.detach()),
                              replay_index=(step-1)%len(replay_batches),grad_norm=float(norm),update_norm=delta))
        final=r.old.snapshot(expert)
        save_new(Path(output),dict(state=final,initial_digest=state_digest(initial),module=path,steps=160,curve=curve,
                                   replay_weight=1.,replay_schedule='fixed_allowed_order_cyclic'))
        budget()
        active,target_tokens=r.q.logits(runtime,hook,source_batches[0])
        hook.enabled=False
        try:
            with torch.inference_mode():
                out=runtime.model(**source_batches[0].forward_kwargs())
                mask=source_batches[0].labels[:,1:]!=-100
                base=out.logits[:,:-1][mask].detach().cpu().double()
                base_tokens=source_batches[0].labels[:,1:][mask].cpu()
        finally:
            hook.enabled=True
        if not torch.equal(target_tokens,base_tokens): raise ValueError('Response token bindings differ')
        response_norm=float((active-base).norm())
        if not math.isfinite(response_norm) or response_norm==0: raise ValueError('No finite expert response')
        reloaded=torch.load(output,map_location='cpu',weights_only=True)
        if any(not torch.equal(v.cpu(),reloaded['state'][k]) for k,v in final.items()): raise ValueError('state reload mismatch')
        expert.load_state_dict(reloaded['state'])
        restored,restored_tokens=r.q.logits(runtime,hook,source_batches[0])
        if not torch.equal(active,restored) or not torch.equal(target_tokens,restored_tokens):
            raise ValueError('Reload changed model output')
        if any(p._version!=versions[name] for name,p in runtime.model.named_parameters()): raise ValueError('Base parameter modified')
        return {'module':path,'steps':160,'parameters':7168,'initial_digest':state_digest(initial),
                'final_digest':state_digest(final),'save_reload_state_equal':True,'save_reload_output_equal':True,
                'response_norm':response_norm,'replay_batches':len(replay_batches)}
    finally: hook.detach()
