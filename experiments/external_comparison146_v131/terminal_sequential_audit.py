"""Read-only terminal audit; no model loading, generation, optimization or Judge."""
import collections
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import statistics
import time


def read(p):return json.loads(Path(p).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def identity(row):return json.dumps([row[k] for k in ('method','mode','prefix','owner')]+[row['query']['query_id']],separators=(',',':'))


def main():
    root=Path(os.environ['V131_ROOT']);out=root/'MEMIT_MM';lock=read(root/'PROTOCOL_LOCK.json')
    plan={identity(v):v for v in read(root/'EVAL_PLAN.json') if v['method']=='MEMIT_MM'}
    assert len(plan)==7494
    terminal=read(out/'INCOMPLETE_GENERATION.json');aborted={v['consumer_id'] for v in terminal['aborted_consumers']}
    assert len(aborted)==4 and all(not v['formal_retry_allowed'] for v in terminal['aborted_consumers'])
    db=sqlite3.connect('file:'+str(root/'judge_v131/queue.sqlite')+'?mode=ro',uri=True)
    prompt=db.execute("SELECT value FROM meta WHERE key='prompt'").fetchone()[0]
    records={};byphase=collections.defaultdict(list);scope=collections.Counter();executions=collections.Counter()
    for cid,path,mode,n,owner,qid,payload_key,state,correct,effort,record,binding in db.execute(
        "SELECT c.id,c.path,c.mode,c.prefix,c.owner,c.query_id,c.payload_key,p.state,p.correct,p.effort,p.record,p.binding "
        "FROM consumer c JOIN payload p ON c.payload_key=p.key WHERE c.method='MEMIT_MM'"):
        assert cid in plan and cid not in aborted and cid not in records
        v=read(path);row=plan[cid];query=row['query'];ji=v['binding']['judge_input'];answer=v['R0']
        assert v['consumer_id']==cid and (v['mode'],v['prefix'],v['owner'])==(mode,n,owner)
        assert all(v['binding']['input'][k]==query[k] for k in ('query_id','question','reference','image_sha256'))
        assert ji['question']==query['question'] and ji['reference']==query['reference'] and ji['image_sha256']==query['image_sha256']
        assert state in ('FORMAT_VALID','REUSED') and correct in (0,1) and effort=='high'
        expected=digest(dict(query_id=qid,question=ji['question'],reference=ji['reference'],image_sha256=ji['image_sha256'],
            prompt_ids=ji['prompt_ids'],output=answer,judge=dict(model='gpt-6-astra',reasoning_effort=effort,
                protocol=lock['scoring']['protocol'],prompt=prompt)))
        assert payload_key==expected
        visible=json.loads(record);saved=json.loads(binding)
        assert visible==dict(opaque_query_id=expected,question=ji['question'],gold_answer=ji['reference'],raw_base_answer=answer['raw_answer'])
        assert saved['query_id']==qid and saved['output']==answer
        # A shared Judge payload can have an earlier consumer's runtime;
        # each prediction keeps its own runtime in the immutable output record.
        assert all(saved['binding'][k]==ji[k] for k in ('question','reference','image_sha256','prompt_ids'))
        assert ji['runtime']['actual_precision']=='MATH_FP32_FROZEN_FP16_PREFILL'
        if saved['binding']['runtime']!=ji['runtime']:scope['shared_score_other_runtime_provenance']+=1
        assert v['actual_inserted_prefix']==(1 if mode=='single' else n)
        assert v['cumulative_parameter_state']==(mode!='single') and v['no_external_router']
        tokens=answer['raw_token_ids'];assert tokens
        records[cid]=v;byphase[(mode,n if mode!='single' else 0)].append((len(tokens),correct,tokens[-1]==2))
        scope['consumer_binding_and_score_checks']+=1;executions[v['execution']]+=1
    assert set(records)==set(plan)-aborted
    group=[]
    for (mode,n),values in sorted(byphase.items()):
        if mode=='insertion':continue
        group.append(dict(mode=mode,prefix=n,n=len(values),all_query_correct=sum(v[1] for v in values),
            mean_tokens=statistics.mean(v[0] for v in values),at_cap=sum(v[0]>=1024 for v in values),EOS=sum(v[2] for v in values)))
    first={mode:{plan[cid]['query']['query_id']:v for cid,v in records.items() if plan[cid]['mode']==mode and
                  (plan[cid]['owner']==1 if mode=='single' else plan[cid]['prefix']==1)} for mode in ('single','bank')}
    assert set(first['single'])==set(first['bank'])
    exact_tokens=sum(first['single'][q]['R0']['raw_token_ids']==first['bank'][q]['R0']['raw_token_ids'] for q in first['single'])
    exact_bindings=sum(first['single'][q]['binding']['judge_input']==first['bank'][q]['binding']['judge_input'] for q in first['single'])
    a,b=read(out/'training/single_1.json'),read(out/'training/sequential_1.json')
    assert a['latent']==b['latent'] and a['writes']==b['writes']
    training=[];count_updates=0
    for mode in ('single','sequential'):
        for owner in range(1,147):
            v=read(out/'training'/f'{mode}_{owner}.json');assert v['save_load_exact']
            if mode=='sequential':assert v['actual_inserted']==owner and v['latent_recomputed_under_current_weights']
            assert [w['layer'] for w in v['writes']]==[4,5,6,7,8]
            count_updates+=sum('gradient_norm' in v for v in v['latent']['trace'])
            if mode=='sequential' and owner in (1,2,3,4,5,6,10,50,100,146):
                a=v['latent'];training.append(dict(owner=owner,initial_norm=a['initial_norm'],delta_norm=a['delta_norm'],
                    initial_NLL=a['trace'][0]['NLL'],final_NLL=a['trace'][-1]['NLL'],
                    mean_write_norm=statistics.mean(w['update_norm'] for w in v['writes'])))
    assert count_updates==terminal['counts']['updates']
    insertions=[(plan[cid]['prefix'],len(v['R0']['raw_token_ids'])) for cid,v in records.items() if plan[cid]['mode']=='insertion']
    replay=read(out/'implementation_audit3/RESULT.json');dense=read(out/'implementation_audit3/DENSE_REFERENCE_RESULT.json')
    assert replay['status']=='COMPLETE' and replay['inverse_initialization_backend']=='GPU_BEFORE_MODEL'
    assert replay['maximum_write_error']==0 and all(v['prefix_equal'] for v in replay['generation_checks'])
    assert dense['status']=='PASS' and len(dense['rows'])==10
    # Reuse the two fixed existing keys per layer; do not run more model forwards.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    old=torch.load(out/'implementation_audit3/SOLVE_RECORDS.pt',map_location='cpu',weights_only=True)
    by={(v['owner'],v['layer']):v for v in old};geometry=[]
    for layer in range(4,9):
        a,b=by[1,layer],by[6,layer];x,y=a['key'].double(),b['key'].double();u,v=a['adjusted'].double(),b['adjusted'].double()
        geometry.append(dict(layer=layer,key_cosine=float((x@y)/(x.norm()*y.norm())),self_1=float(x@u),self_6=float(y@v),
            cross_1_to_6=float(y@u),cross_relative_1_to_6=float((y@u)/(x@u)),
            note='Only layer4 inputs are invariant to earlier writes at4..8; higher-layer keys use different cumulative states'))
    result=dict(status='AUDIT_COMPLETE_NO_NEW_MODEL_CALLS',time=time.time(),
        coverage=dict(planned=7494,valid=7490,known_generation_UNKNOWN=4,score_UNKNOWN=0),
        bindings=dict(status='PASS',**scope),first_edit=dict(source_optimizer_and_writes_exact=True,
            overlapping_queries=len(first['single']),tokens_exact=exact_tokens,full_judge_bindings_exact=exact_bindings),
        saved_training=dict(single=146,sequential=146,actual_updates=count_updates,counts_match=True,save_load_receipts_all=True),
        generation=group,first_insertion_at_cap=min(n for n,t in insertions if t>=1024),
        existing_independent_replay=dict(edits=len(replay['replay']),physical_writes=30,max_abs_write_error=0,
            native_prefix_teacher_exact=all(v['prefix_vs_teacher_max_abs']==0 for v in replay['replay']),
            generation_prefixes_exact=all(v['prefix_equal'] for v in replay['generation_checks']),
            direct_solve_checks=10,max_equation_residual=max(v['relative_equation_residual'] for v in dense['rows'])),
        observed_instability=training,existing_key_geometry=geometry,
        interpretation=dict(mechanical_loop_bug_found=False,unique_cause_proven=False,
            strong_candidates=['FixedMM fact lookup at expanded last chat-prompt token instead of native subject_last',
                'Strong layer4 key interference for the two pre-existing recorded keys',
                'Current-activation-relative clamp grows as cumulative hidden states inflate',
                'Text-only Wiki moment distribution omits image/chat-token inputs; site/layers were transferred without VLM causal qualification'],
            implementation_label='Fixed project MM online adaptation; not a validated native MEMIT VLM implementation',
            not_an_authorized_hyperparameter_search=True),cost=dict(new_GPU=0,new_model_forwards=0,new_model_backwards=0,new_edits=0,new_generations=0,new_Judge=0),
        limitations=['Existing GPU replay independently covers first6 edits only, not all146 physical writes',
            'Same-source replay proves execution consistency but cannot exclude a shared algorithm/adaptation error',
            'Two-key CPU analysis does not prove global146 key collapse or unique causality',
            'No new causal intervention, VLM layer localization or alternative hyperparameters run'])
    print(json.dumps(result))

if __name__=='__main__':main()
