"""Bounded native CrispEdit-Seq/Adam comparison; private frozen config in RUN_DIR."""
import gc
import gzip
import json
import os
from pathlib import Path
import signal
import time
import traceback
import torch
from torch.func import functional_call
from experiments.alphaedit_stage17 import runtime as native
from experiments.alphaedit_stage17.run import configure, consume, dump, hooks, space
from experiments.alphaedit_vlm.core import answer_nll
from .core import factors, combine, projection, optimizer_class


def atomic_state(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def text_batches(rt,c,start,count):
    seen=0
    with gzip.open(c['corpus_path'],'rt') as stream:
        for index,line in enumerate(stream):
            if index<start:continue
            if seen>=count:break
            text=json.loads(line)['text']
            ids=rt.adapter.tokenizer(text,return_tensors='pt',truncation=True,max_length=512)['input_ids'].cuda()
            labels=ids.clone();labels[labels==0]=-100
            if rt.adapter.tokenizer.pad_token_id is not None:
                labels[labels==rt.adapter.tokenizer.pad_token_id]=-100
            if not (labels[:,1:]!=-100).any():raise ValueError('empty fixed corpus document')
            seen+=1
            yield dict(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False),labels
    if seen!=count:raise ValueError('fixed corpus too short')


def text_loss(rt,c):
    total=0.;tokens=0
    with torch.no_grad():
        for inp,lab in text_batches(rt,c,c['statistics_documents'],c['text_holdout_documents']):
            n=int((lab[:,1:]!=-100).sum());total+=float(answer_nll(rt.model(**inp).logits,lab))*n;tokens+=n
    return dict(nll=total/tokens,tokens=tokens,documents=c['text_holdout_documents'])


def run(c,root,save):
    configure();rt=native.load(c,root);model=rt.model;torch.manual_seed(c['seed'])
    layers=c['layers'];names={l:f'model.layers.{l}.mlp.down_proj.weight' for l in layers}
    weights={l:model.get_submodule(names[l].rsplit('.',1)[0]).weight for l in layers}
    original={l:w.detach().cpu().clone() for l,w in weights.items()}
    frozen={n:p._version for n,p in model.named_parameters() if n not in names.values()}
    buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    def verify():
        if hooks(model) or frozen!={n:p._version for n,p in model.named_parameters() if n not in names.values()}:
            raise RuntimeError('noneditable parameter mutation or retained hook')
        if any(not torch.equal(b,buffers[n]) for n,b in model.named_buffers()):raise RuntimeError('buffer mutation')
    def restore():
        with torch.no_grad():
            for l,w in weights.items():w.copy_(original[l].to(w.device))
    def panel(folder):
        folder.mkdir(exist_ok=True)
        for i,(qid,q) in enumerate(c['queries'].items(),1):
            path=folder/f'{i:03d}.private.json'
            if path.exists():
                if json.loads(path.read_text())['query_id']!=qid:raise ValueError('panel identity changed')
                continue
            save(dict(state='GENERATING',panel=str(folder.relative_to(root)),completed=i-1,intended=len(c['queries'])))
            dump(path,dict(query_id=qid,**native.generated(rt,q)))

    # Base outputs also enforce the original image and prompt bindings before editing.
    panel(root/'base')
    if not (root/'base/TEXT_LOSS.json').exists():dump(root/'base/TEXT_LOSS.json',text_loss(rt,c))
    cap_path=root/'capability.private.pt'
    if not (root/'crisp/COMPLETE.json').exists() and not cap_path.exists():
        save(dict(state='CAPABILITY_STATISTICS',completed=0,intended=c['statistics_documents']))
        def progress(n,tokens):
            if n%10==0:save(dict(state='CAPABILITY_STATISTICS',completed=n,intended=c['statistics_documents'],tokens=tokens))
        cap=factors(model,text_batches(rt,c,0,c['statistics_documents']),layers,progress)
        verify();space(root);atomic_state(cap_path,cap)
        dump(root/'CAPABILITY.json',dict(documents=c['statistics_documents'],tokens=cap[layers[0]]['N'],
             estimator='upstream empirical summed-CE gradient outer products',layers=layers))
        del cap;gc.collect();torch.cuda.empty_cache()

    for arm in c['arms']:
        folder=root/arm;folder.mkdir(exist_ok=True)
        if (folder/'COMPLETE.json').exists():continue
        restore()
        cap=torch.load(cap_path,map_location='cpu',weights_only=True) if arm=='crisp' else None
        masters={l:torch.nn.Parameter(weights[l].detach().float().clone()) for l in layers}
        optimizer=None;history=None;done=0;resume=folder/'resume.private.pt';pending_state=None
        if resume.exists():
            pending_state=torch.load(resume,map_location='cpu',weights_only=True)
            done=pending_state['completed'];history=pending_state['history']
            with torch.no_grad():
                for l,w in masters.items():w.copy_(pending_state['masters'][l].to(w.device));weights[l].copy_(w.to(weights[l].dtype))
        for index,task in enumerate(c['tasks'],1):
            if index<=done:continue
            started=time.monotonic();space(root)
            save(dict(state='PROJECTION' if arm=='crisp' else 'EDITING',arm=arm,completed=index-1,intended=len(c['tasks'])))
            cache={};spectrum=[]
            if arm=='crisp':
                combined=combine(cap,history) if history is not None else cap
                for l in layers:
                    save(dict(state='PROJECTION',arm=arm,active=index,layer=l))
                    pair=combined[l]
                    cache[masters[l]],receipt=projection(pair['A'].cuda(),pair['B'].cuda(),c['energy'])
                    spectrum.append(dict(layer=l,**receipt))
                if optimizer is None:optimizer=optimizer_class(c['upstream_dir'])(list(masters.values()),cache,lr=c['lr'],weight_decay=0)
                elif pending_state is None:optimizer.reset_cache(cache)
                del combined
            elif optimizer is None:optimizer=torch.optim.Adam(list(masters.values()),lr=c['lr'],weight_decay=0)
            if pending_state is not None:
                optimizer.load_state_dict(pending_state['optimizer'])
                if arm=='crisp':optimizer.reset_cache(cache)
                pending_state=None;gc.collect()
            q=c['queries'][task['edit_id']];inp,labels=native.prepared(rt,q)
            trace=[];before={l:w.detach().cpu().clone() for l,w in weights.items()}
            for step in range(c['steps']):
                save(dict(state='EDITING',arm=arm,completed=index-1,active=index,step=step,intended_steps=c['steps']))
                optimizer.zero_grad(set_to_none=True)
                replacements={names[l]:w.to(weights[l].dtype) for l,w in masters.items()}
                loss=answer_nll(functional_call(model,replacements,(),inp).logits,labels)
                value=float(loss.detach())
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite native objective')
                row=dict(step=step,nll=value)
                if value<c['stop_loss']:
                    row['updated']=False;trace.append(row);del loss,replacements;break
                (loss*c['loss_scale']).backward()
                for w in masters.values():
                    w.grad.div_(c['loss_scale'])
                    if not torch.isfinite(w.grad).all():raise FloatingPointError('nonfinite editing gradient')
                norm=sum(float(w.grad.norm())**2 for w in masters.values())**.5
                if not norm>0:raise RuntimeError('zero native gradient')
                optimizer.step()
                projected_norm=sum(float(w.grad.norm())**2 for w in masters.values())**.5
                with torch.no_grad():
                    for l,w in masters.items():
                        if not torch.isfinite(w).all():raise FloatingPointError('nonfinite master W')
                        weights[l].copy_(w.to(weights[l].dtype))
                        if not torch.isfinite(weights[l]).all():raise FloatingPointError('nonfinite deployed W')
                row.update(updated=True,gradient_norm=norm,projected_gradient_norm=projected_norm)
                trace.append(row);del loss,replacements
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():post=float(answer_nll(model(**inp).logits,labels))
            verify()
            dump(folder/f'{index:03d}.TARGET.private.json',dict(query_id=task['edit_id'],**native.generated(rt,q)))
            # Official sequential cache branch: post-edit factors, token-weighted history.
            if arm=='crisp':
                save(dict(state='EDIT_STATISTICS',arm=arm,active=index))
                history=combine(history,factors(model,[(inp,labels)],layers))
                verify()
            writes=[]
            for l,w in weights.items():
                actual=w.detach().cpu().float()-before[l].float()
                writes.append(dict(layer=l,norm=float(actual.norm()),changed_elements=int((actual!=0).sum())))
            receipt=dict(index=index,arm=arm,trace=trace,nll_after=post,writes=writes,projection=spectrum,
                hooks=0,frozen_parameters=True,frozen_buffers=True,history_tokens=history[layers[0]]['N'] if history else 0,
                seconds=time.monotonic()-started)
            dump(folder/f'{index:03d}.RECEIPT.json',receipt)
            # Projection bases are reconstructed from factors, not duplicated in every state.
            opt_state=optimizer.state_dict()
            opt_state['param_groups']=[{k:v for k,v in group.items() if 'projection_cache' not in k} for group in opt_state['param_groups']]
            atomic_state(resume,dict(completed=index,masters={l:w.detach().cpu() for l,w in masters.items()},
                optimizer=opt_state,history=history))
            del inp,labels,before,actual,opt_state;gc.collect();torch.cuda.empty_cache()
        panel(folder/'final');dump(folder/'TEXT_LOSS.json',text_loss(rt,c))
        matrices={l:folder/f'active-{l}.private.pt' for l in layers}
        for l,path in matrices.items():atomic_state(path,weights[l].detach().cpu())
        q=c['queries'][c['tasks'][-1]['edit_id']]
        expected=native.logits(rt,q);generation=native.generated(rt,q)
        save(dict(state='CLEAN_RELOAD',arm=arm,completed=len(c['tasks'])))
        result=consume(c,folder,matrices,q);difference=float((expected-result['logits']).abs().max())
        if difference or generation!=result['generation'] or result['hooks'] or result['algorithm_loaded']:
            raise RuntimeError('fresh native reload mismatch')
        verify()
        dump(folder/'COMPLETE.json',dict(state='GENERATED_NOT_SCORED',edits=len(c['tasks']),queries=len(c['queries']),
             native_reload_logit_error=difference,native_reload_tokens_identical=True))
        paths=[resume,*matrices.values()]
        if arm=='crisp':paths.append(cap_path)
        size=sum(p.stat().st_size for p in paths)
        for p in paths:p.unlink()
        dump(folder/'DELETION.json',dict(files=len(paths),bytes=size,retained_copy=False,
            consumers=['all_ordered_edits','insertion_and_final_generation','text_holdout','native_reload'],
            reconstruction='recompute frozen statistics and rerun fixed edits'))
        del optimizer,masters,history,cap,cache;gc.collect();torch.cuda.empty_cache()
    dump(root/'COMPLETE.json',dict(state='GENERATED_NOT_SCORED',arms=c['arms'],edits_per_arm=len(c['tasks']),queries=len(c['queries'])))
    save(dict(state='GENERATED_NOT_SCORED',arms_completed=len(c['arms'])))


def main():
    root=Path(os.environ['RUN_DIR']);c=json.loads((root/'CONFIG.private.json').read_text())
    def save(status):dump(root/'STATUS.json',dict(**status,updated_epoch=time.time()))
    def alarm(*_):raise TimeoutError('original campaign wall deadline reached')
    if time.time()>=c['deadline_epoch']:raise TimeoutError('original deadline already expired')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(c['deadline_epoch']-time.time())))
    try:save(dict(state='LOADING'));run(c,root,save)
    except BaseException as error:
        dump(root/'FAILURE.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        save(dict(state='FAILED',type=type(error).__name__));raise
