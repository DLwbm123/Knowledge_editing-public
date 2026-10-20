"""CPU-only R0 versus same-DEV-supervision SIMPLE and CONDITIONAL evaluation."""
import json,time,sys,os
from pathlib import Path
from collections import Counter,defaultdict
import torch
from route_fit_v22 import fit_predict
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def count(labels,base):
    assert len(labels)==len(base)
    return dict(n=len(labels),correct=sum(v=='CORRECT' for v in labels),Base_correct=sum(v=='CORRECT' for v in base),lost_correct=sum(b=='CORRECT' and v!='CORRECT' for b,v in zip(base,labels)),explicit_wrong_after_Base_correct=sum(b=='CORRECT' and v=='INCORRECT' for b,v in zip(base,labels)),Base_incorrect=sum(b=='INCORRECT' for b in base),repaired=sum(b=='INCORRECT' and v=='CORRECT' for b,v in zip(base,labels)),labels=dict(Counter(labels)))
def run(root):
    started=time.time()
    def budget():
        assert not (root/'STOP').exists() and time.time()<read(root/'RESOURCE_LIMITS.json')['hard_end']-300
        assert time.time()-started<3600,'Finite CPU fit/aggregation budget'
    budget()
    assert not (root/'STOP').exists() and time.time()<read(root/'RESOURCE_LIMITS.json')['hard_end']-300
    assert read(root/'ROUTE_CPU_ADMISSION.json')['status']=='PASS'
    plan=read(root/'ROUTE_PLAN_FROZEN.json');cells=read(root/'BANK_CELLS.json')['cells'];mapping=read(root/'BANK_BLIND_MAPPING.json');packet=read(root/'BANK_BLIND_PACKET.json')
    verdict=dict(read(root/'BLIND_REUSE.json'));verdict.update(read(root/'BLIND_VERDICTS.json'));verdict.update(read(root/'BANK_BLIND_VERDICTS.json'))
    assert set(verdict)=={x['code'] for x in packet['items']}
    assert all(v['label'] in ('CORRECT','INCORRECT','UNCERTAIN','ABSTAIN') and v.get('reason') for v in verdict.values())
    experts=[t['key'] for t in plan['tasks']];qids=sorted(plan['queries']);keys={}
    for i in range(5):
        shard=torch.load(root/str(i)/'QUERY_KEYS.pt',map_location='cpu',weights_only=True)
        for q in qids:
            if q in shard:assert q not in keys;keys[q]=shard[q]
    assert set(keys)==set(qids)
    env=read(root/'PARENT_ENV.json');os.environ.update({k:str(v) for k,v in env.items()});p=Path(env['PARENT_RUN'])
    sys.path[:0]=[str(p/'private/tools'),str(p/'private/source'),str(p/'private/official_llava')]
    import module_run
    from m3bench_repro.editors.routing import MemoryRouter
    router=MemoryRouter('euclidean');maxdiff=0.
    for task in plan['tasks']:
        anchor=torch.load(root/str((task['order']-1)%5)/task['key']/'R0_ANCHORS.pt',map_location='cpu',weights_only=True)['BALANCEDIT']
        assert anchor['distance']=='euclidean' and len(anchor['entries'])==1
        entry=anchor['entries'][0];assert entry['logical_edit_id']==task['key']
        difference=float((keys[task['source']['query_id']]['joint'].float().flatten()-entry['key'].float().flatten()).abs().max());maxdiff=max(maxdiff,difference)
        assert difference<=1e-6,'Saved query/native R0 feature mismatch: preserve failure, no new extraction'
        router.add(task['key'],entry['key'],entry['radius'],entry['label'])
    route={q:router.route(keys[q]['joint']) for q in qids}
    r0={q:experts.index(v.logical_edit_id) if v.activated else None for q,v in route.items()}
    basepaths={q:str(Path('base')/str(i%5)/(q+'.json')) for i,q in enumerate(sorted(read(root/'QUERY_ROWS.json')))}
    labels={('BASE',q):verdict[mapping[basepaths[q]]]['label'] for q in qids}
    for pair,path in cells.items():
        e,q=pair.split('|');labels[e,q]=verdict[mapping[path]]['label']
    dev={q for q in qids if plan['query_roles'][q]=='DEV'}
    # Pass a physically restricted dictionary to fitting; no held-out labels available there.
    with (root/'ROUTE_FIT_ONCE').open('x'):pass
    fit=fit_predict(plan,keys,{pair:v for pair,v in labels.items() if pair[1] in dev},r0,budget)
    selections=dict(OFF={q:None for q in qids},R0=r0)
    for name,v in fit['methods'].items():selections[name]=v['selected']
    report={};config=read(root/'DATA_ADMISSION.json')
    def output(selection,q):return labels['BASE' if selection[q] is None else experts[selection[q]],q]
    for name,selection in selections.items():
        if selection is None:report[name]=dict(status='NA_NO_DEV_WORKPOINT');continue
        table={}
        for split,ids in [('ALL_DEVELOPMENT',qids),('DEV',sorted(dev)),('HELDOUT_DEVELOPMENT',[q for q in qids if q not in dev])]:
            table[split]=count([output(selection,q) for q in ids],[labels['BASE',q] for q in ids])
        byrole=defaultdict(list)
        for task in config['tasks']:
            for row in task['eval_rows']:
                for role in row['roles']:byrole[role].append(row['query_id'])
        table['benchmark_role_consumers']={role:count([output(selection,q) for q in ids],[labels['BASE',q] for q in ids]) for role,ids in byrole.items()}
        table['activation_relation']=dict(Counter('OFF' if selection[q] is None else plan['relations'][experts[selection[q]]+'|'+q] for q in qids));report[name]=table
    owner=defaultdict(list)
    for task in config['tasks']:
        for row in task['eval_rows']:
            for role in row['roles']:owner[role].append((task['key'],row['query_id']))
    forced={role:count([labels[e,q] for e,q in pairs],[labels['BASE',q] for e,q in pairs]) for role,pairs in owner.items()}
    upper={}
    for split,ids in [('ALL_DEVELOPMENT',qids),('HELDOUT_DEVELOPMENT',[q for q in qids if q not in dev])]:
        upper[split]=dict(n=len(ids),any_expert_or_Base_correct=sum(any(labels[e,q]=='CORRECT' for e in ['BASE']+experts) for q in ids),applicable_expert_or_Base_correct=sum(labels['BASE',q]=='CORRECT' or any(labels[e,q]=='CORRECT' and plan['relations'][e+'|'+q]=='YES' for e in experts) for q in ids),oracle_only_not_a_router=True)
    write(root/'ROUTE_PRIVATE_FIT.json',fit);write(root/'ROUTE_PRIVATE_SELECTIONS.json',dict(selections=selections,R0={q:vars(v) for q,v in route.items()}))
    budget()
    result=dict(status='COMPLETE_DEVELOPMENT_NOT_CLINICAL_CONFIRMATION',methods=report,designated_expert_forced=forced,capability_oracles=upper,native_key_max_difference=maxdiff,new_model_calls=0,CPU_process_wall_seconds=time.time()-started)
    write(root/'ROUTE_RESULTS.json',result);return result
if __name__=='__main__':
    import traceback,signal
    def expired(*_):raise TimeoutError('Frozen one-hour CPU evaluation limit')
    signal.signal(signal.SIGALRM,expired);signal.alarm(3590)
    torch.set_num_threads(1);root=Path(sys.argv[1]);started=time.time();code=0
    try:run(root)
    except BaseException:
        code=1
        write(root/'ROUTE_FIT_FAILURE.json',dict(error=traceback.format_exc(),time=time.time(),new_model_calls=0));raise
    finally:write(root/'ROUTE_EXIT.json',dict(code=code,CPU_process_wall_seconds=time.time()-started,new_model_calls=0))
