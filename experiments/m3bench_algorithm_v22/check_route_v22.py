"""Synthetic end-to-end CPU test, using the original router and real consumers."""
import json,time,tempfile,copy
from pathlib import Path
import torch
from route_fit_v22 import selfcheck
from route_evaluate_v22 import run,write,read

def check(parent_env):
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='q.22.route.',dir='/data/bmw/tmp') as tmp:
        top=Path(tmp)
        def fixture(root,flip=False,missing=False,bad_key=False):
            root.mkdir();qids=['n0','n1','d0','d1','h0'];experts=['E0','E1']
            keys={q:{k:torch.tensor([1.+i,.2+i/10,.5]) for k in ('joint','image','text')} for i,q in enumerate(qids)}
            tasks=[dict(key=e,order=i+1,source=dict(query_id='n'+str(i)),eval_rows=[dict(query_id='n'+str(i),roles=['T0']),dict(query_id='d'+str(i),roles=['T1L'])]) for i,e in enumerate(experts)]
            plan=dict(tasks=tasks,queries={q:{} for q in qids},query_roles={q:'DEV' if q in ('d0','d1') else 'HELDOUT_DEVELOPMENT' for q in qids},query_groups={q:q for q in qids},relations={e+'|'+q:('YES' if q=='n'+str(i) else 'NO' if q.startswith('d') else 'UNKNOWN') for i,e in enumerate(experts) for q in qids})
            write(root/'ROUTE_PLAN_FROZEN.json',plan);write(root/'DATA_ADMISSION.json',dict(tasks=tasks));write(root/'QUERY_ROWS.json',{q:{} for q in qids});write(root/'PARENT_ENV.json',parent_env)
            write(root/'RESOURCE_LIMITS.json',dict(hard_end=time.time()+3600));write(root/'ROUTE_CPU_ADMISSION.json',dict(status='PASS'))
            labels={('BASE',q):'INCORRECT' for q in qids};labels['BASE','d1']='CORRECT'
            for e in experts:
                for q in qids:labels[e,q]='CORRECT' if e=='E0' and q=='d0' or e=='E1' and q=='d1' else 'INCORRECT'
            if flip:
                for pair in labels:
                    if pair[1] not in ('d0','d1'):labels[pair]='CORRECT' if labels[pair]=='INCORRECT' else 'INCORRECT'
            paths={};cells={};verdict={};items=[]
            for i,(pair,label) in enumerate(sorted(labels.items())):
                e,q=pair;code='B%06d'%i
                path=str(Path('base')/str(sorted(qids).index(q)%5)/(q+'.json')) if e=='BASE' else str(Path('bank/outputs')/e/(q+'.json'))
                paths[path]=code;verdict[code]=dict(label=label,reason='Synthetic fixed label');items.append(dict(code=code))
                if e!='BASE':cells[e+'|'+q]=path
            if missing:verdict.pop(next(iter(verdict)))
            write(root/'BANK_CELLS.json',dict(cells=cells));write(root/'BANK_BLIND_MAPPING.json',paths);write(root/'BANK_BLIND_PACKET.json',dict(items=items));write(root/'BLIND_REUSE.json',{});write(root/'BLIND_VERDICTS.json',verdict);write(root/'BANK_BLIND_VERDICTS.json',{})
            for i in range(5):
                folder=root/str(i);folder.mkdir();torch.save({q:keys[q] for j,q in enumerate(qids) if j%5==i},folder/'QUERY_KEYS.pt')
            for i,e in enumerate(experts):
                folder=root/str(i)/e;folder.mkdir();key=keys['n'+str(i)]['joint'].clone()
                if bad_key and i==0:key[0]+=.1
                torch.save(dict(BALANCEDIT=dict(distance='euclidean',entries=[dict(logical_edit_id=e,key=key,radius=.01,label=[])])),folder/'R0_ANCHORS.pt')
        a=top/'a';b=top/'b';fixture(a);fixture(b,flip=True)
        first=run(a);second=run(b)
        from m3bench_repro.editors.routing import MemoryRouter
        router=MemoryRouter('euclidean');router.add('nearest_but_off',torch.tensor([0.,0.,1.]),0.);router.add('farther_eligible',torch.tensor([1.,0.,1.]),10.)
        decision=router.route(torch.tensor([.2,0.,1.]));assert not decision.activated and decision.nearest_logical_edit_id=='nearest_but_off'
        assert read(a/'ROUTE_PRIVATE_FIT.json')==read(b/'ROUTE_PRIVATE_FIT.json')
        assert read(a/'ROUTE_PRIVATE_SELECTIONS.json')==read(b/'ROUTE_PRIVATE_SELECTIONS.json')
        assert first['new_model_calls']==second['new_model_calls']==0 and first['native_key_max_difference']==0
        for name,kw in [('missing',dict(missing=True)),('bad_key',dict(bad_key=True))]:
            p=top/name;fixture(p,**kw)
            try:run(p)
            except AssertionError:assert not (p/'ROUTE_FIT_ONCE').exists()
            else:raise AssertionError('Missing judgments or native parity mismatch accepted')
        return dict(status='PASS',math=selfcheck(),full_original_router_and_eval=True,R0_nearest_then_own_radius=True,heldout_answers_cannot_change_fit_threshold_or_decisions=True,missing_scores_rejected_before_fit=True,native_feature_mismatch_rejected_before_fit=True,new_model_calls=0)
if __name__=='__main__':
    import sys
    print(json.dumps(check(read(Path(sys.argv[1])/'PARENT_ENV.json'))))
