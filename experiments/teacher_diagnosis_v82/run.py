"""Finite source-only teacher diagnosis; no fitting, resampling or heldout inputs."""
import json
import os
from pathlib import Path
import signal
import time
import traceback
import v81 as v


def short_context(fact, arm):
    record = {k:fact['medical_record'][k] for k in
              ('entity','relation','values','uncertainty','scope')}
    parts = [f'{k}: {json.dumps(value,ensure_ascii=False)}' for k,value in record.items()]
    if arm=='S1': body=' '.join(parts)
    elif arm=='S2': body=json.dumps({f'item_{i+1}':x for i,x in enumerate(parts)},ensure_ascii=False)
    else: body=json.dumps(record,ensure_ascii=False)
    return ('Use only these source facts. Return one short answer covering every value and '
            'qualifier. Add no other factual claim or explanation.\nSource facts:\n'+body)


def variants(row, fact):
    yield 'Base', row
    for arm in ('S1','S2','S3'):
        yield 'old_'+arm, dict(row,question=row['question']+'\n\n'+v.contexts(fact,arm))
    for arm in ('S1','S2','S3'):
        yield 'short_'+arm, dict(row,question=short_context(fact,arm)+'\n\nQuestion: '+row['question'])


def greedy(item):
    torch=v.torch;v.generating=True;v.counts['teacher_generations']+=1
    ids=[]
    with torch.inference_mode():
        out=v.runtime.model(**v.pref(item[3]))
        for _ in range(128):
            if len(ids)==1:
                prefix=v.pref(item[3])['inputs_embeds']
                embedded=v.runtime.model.get_input_embeddings()(torch.tensor([ids],device=v.runtime.device))
                check=v.runtime.model(inputs_embeds=torch.cat([prefix,embedded],1),use_cache=False,return_dict=True)
                torch.testing.assert_close(out.logits[0,-1],check.logits[0,-1],rtol=1e-5,atol=5e-4)
                del prefix,embedded,check
            token=int(out.logits[0,-1].argmax());ids.append(token)
            if token==v.runtime.adapter.tokenizer.eos_token_id:break
            past=out.past_key_values;del out
            out=v.runtime.model(input_ids=torch.tensor([[token]],device=v.runtime.device),
                past_key_values=past,use_cache=True,return_dict=True)
            del past
        del out
    v.generating=False
    return dict(raw_token_ids=ids,response=v.runtime.adapter.tokenizer.decode(ids,skip_special_tokens=True).strip(),
        EOS=ids[-1]==v.runtime.adapter.tokenizer.eos_token_id,at_cap=len(ids)==128)


def diagnose(task):
    torch=v.torch;owner=task['order'];fact=v.read(v.ROOT/'FACTS.json')[str(owner)]
    rows=v.read(v.ROOT/'SOURCE_ROWS.json')[str(owner)]
    for view,row in enumerate(rows):
        named=list(variants(row,fact));items=v.prepare(task,[r for _,r in named],True)
        batch=items[0][1];labels=batch[1]
        ids=labels[labels!=-100].tolist()
        assert ids[-1]==v.runtime.adapter.tokenizer.eos_token_id
        base_log=None;stats={};base_ids=items[0][2]['input_ids'].tolist()
        for (name,variant),item in zip(named,items):
            dest=v.ROOT/'diagnosis'/str(owner)/str(view)/(name+'.json')
            assert not dest.exists(),'No implicit repeated diagnostic'
            raw=item[2];pref=v.pref(item[3]);n=pref['inputs_embeds'].shape[1]
            if name!='Base':
                assert raw['input_ids'].tolist()!=base_ids
                decoded=v.runtime.adapter.tokenizer.decode([i for i in raw['input_ids'][0].tolist() if i>=0])
                assert all(value.lower() in decoded.lower() for value in fact['medical_record']['values'])
            with torch.inference_mode():
                target=torch.tensor([ids],device=v.runtime.device)
                embedding=v.runtime.model.get_input_embeddings()(target).half().float()
                full=torch.cat([pref['inputs_embeds'],embedding],1)
                out=v.runtime.model(inputs_embeds=full,use_cache=False,return_dict=True)
                log=out.logits[0,n-1:n+len(ids)-1].float().log_softmax(-1)
                assert log.shape[0]==len(ids)
                if name=='Base':base_log=log.clone()
                gold=log.gather(1,target[0,:,None]).squeeze(1)
                prob=log.exp();eos=v.runtime.adapter.tokenizer.eos_token_id
                mix=.5*base_log.exp()+.5*prob
                stats[name]=dict(mean_reference_NLL=float(-gold.mean()),reference_token_NLL=(-gold).cpu().tolist(),
                    first_EOS_probability=float(prob[0,eos]),first_reference_probability=float(gold[0].exp()),
                    conditional_to_Base_KL=float((prob*(log-base_log)).sum(-1).mean()),
                    mean_entropy=float(-(prob*log).sum(-1).mean()),
                    mixed_reference_NLL=float(-mix.gather(1,target[0,:,None]).log().mean()),
                    mixed_first_EOS_probability=float(mix[0,eos]),
                    text_prompt_tokens=raw['input_ids'].shape[1],expanded_prompt_tokens=n,
                    tokenizer_model_max_length=getattr(v.runtime.llava_model().config,'tokenizer_model_max_length',None),
                    last_prompt_text=raw['prompt'][-180:],facts_in_tokenized_prompt=name!='Base')
                del out,log,prob,mix,full,embedding
            record=dict(owner=owner,view=view,condition=name,question=row['question'],reference=row['reference'],
                        stats=stats[name],source_only=True)
            if view==0:record.update(greedy(item))
            v.write(dest,record);v.progress(owner=owner,view=view,condition=name)
        del base_log,items


def main():
    with (v.ROOT/f'ONCE_{v.PHASE}_{v.PART}').open('x'):pass
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Diagnostic deadline')))
    signal.alarm(max(1,int(v.limits['hard_end']-time.time()-300)))
    v.setup()
    for task in v.read(v.ROOT/'TASKS.json')[v.PART::2]:diagnose(task)
    v.write(v.ROOT/f'COMPLETE_{v.PHASE}_{v.PART}.json',dict(status='PASS',counts=v.counts,time=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as e:
        code=1;v.write(v.ROOT/f'FAILURE_{v.PHASE}_{v.PART}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),counts=v.counts));raise
    finally:
        v.progress(exit_code=code)
        v.write(v.ROOT/f'EXIT_{v.PHASE}_{v.PART}.json',dict(exit_code=code,counts=v.counts,time=time.time()))
