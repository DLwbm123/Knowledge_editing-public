"""One-input engineering diagnosis of backward aggregation, no candidate update."""
import torch
import token_probe as token
q,c,p,RUN=token.q,token.c,token.p,token.RUN


def main():
    task=q.d.selected()[0];row=token.prior.split()[0][0]
    with p.lease(2):
        runtime,_=c.load(2)
        expert=p.expert(q.d.start_state(task),task['seed'],runtime.device)
        hook=q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
        before={k:v.detach().clone() for k,v in expert.state_dict().items()}
        try:
            batch=q.protection_batch(runtime,row,task);hook.set_teacher_routing(batch.labels)
            out=runtime.model(**batch.forward_kwargs());losses=token.token_losses(out.logits,batch.labels)
            parameters=[dict(expert.named_parameters())[k] for k in q.KEYS]
            flatten=lambda gs:torch.cat([g.detach().cpu().double().flatten() for g in gs])
            pieces=[flatten(torch.autograd.grad(loss,parameters,retain_graph=True)) for loss in losses]
            sequence=flatten(torch.autograd.grad(out.loss,parameters,retain_graph=True))
            manual=flatten(torch.autograd.grad(losses.mean(),parameters,retain_graph=True))
            automatic_logits=torch.autograd.grad(out.loss,out.logits,retain_graph=True)[0]
            manual_logits=torch.autograd.grad(losses.mean(),out.logits,retain_graph=True)[0]
            sum_logits=torch.autograd.grad(losses.sum(),out.logits)[0]/len(losses)
            error=lambda x,y:float((x.double()-y.double()).norm()/(y.double().norm()+1e-30))
            result=dict(status='COMPLETE',engineering_only=True,scientific_candidates=0,
                forwards=1,backwards=len(losses)+5,tokens=len(losses),
                model_dtype=str(next(runtime.model.parameters()).dtype),logit_dtype=str(out.logits.dtype),
                expert_dtype=str(next(expert.parameters()).dtype),embed_dtype=str(batch.forward_kwargs()['inputs_embeds'].dtype),
                loss_mean_error=abs(float(losses.mean()-out.loss)),
                logits_mean_gradient_error=error(manual_logits,automatic_logits),
                logits_sum_gradient_error=error(sum_logits,automatic_logits),
                single_manual_mean_vs_original_parameter_gradient_error=error(manual,sequence),
                mean_of_individual_parameter_gradients_error=error(torch.stack(pieces).mean(0),sequence),
                state_unchanged=all(torch.equal(v,expert.state_dict()[k]) for k,v in before.items()),
                old_tolerance_unchanged=1e-3)
            assert result['state_unchanged']
            c.write(RUN/'public/PRECISION_DIAGNOSTIC.json',result)
            print(result,flush=True)
        finally:hook.detach()
    p.done('PRECISION_COMPLETE')


if __name__=='__main__':main()
