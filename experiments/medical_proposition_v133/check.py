"""CPU checks for changed membership, reuse identity and MED-only parity."""
import copy
import json
import os
from pathlib import Path
import sys
import torch


def main():
    root,parent=Path(os.environ['EXPERIMENT_ROOT']),Path(os.environ['PARENT_RUN'])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_available()
    sys.path[:0]=[str(root/'tools'),str(parent/'private/tools'),str(parent/'private/source'),str(parent/'private/official_llava')]
    import med133_curvature as curvature
    import med133_evaluate as evaluate
    import module_run as r
    from m3bench_repro.editors.routing import MemoryRouter
    result=curvature.selfcheck()
    g=torch.Generator().manual_seed(133)
    paths=[dict(x=torch.randn(n,7,generator=g),full=torch.randn(n,8,5,generator=g),score=torch.randn(n,5,generator=g)) for n in (3,4,2,5)]
    _,old,_,audit=curvature.anchor(paths,[-1.,-2.,-3.,-2.5],2,132)
    _,new,newaudit=curvature.med_anchor(paths,[-1.,-2.,-3.,-2.5],2)
    assert torch.equal(old['x'],new['x']) and torch.equal(old['y'],new['y'])
    assert audit['full_trace']==newaudit['full_trace']
    f=r.c.read(root/'INPUTS_FROZEN.json');entries=[dict(logical_edit_id=str(i),key=[float(i),0.],radius=.1) for i in range(1,147)]
    for mode,n,owner in [('single',1,146),('bank',1,0),('bank',50,0),('bank',100,0),('bank',146,0)]+[('insertion',i,i) for i in range(1,147)]:
        row=dict(mode=mode,prefix=n,owner=owner);selected=evaluate.membership(entries,row)
        assert len(selected)==(1 if mode=='single' else n)
        router=MemoryRouter.from_state(dict(distance='euclidean',entries=selected),device='cpu')
        d=router.route(torch.tensor(selected[-1]['key']))
        assert d.activated and d.logical_edit_id==selected[-1]['logical_edit_id']
    state=dict(G=torch.ones(2));a=evaluate.state_id(state);state['G'][0]+=1
    assert a!=evaluate.state_id(state)
    binding=dict(input=dict(prompt_ids=[1,2],runtime='same'),effective_state='x',generation={'do_sample':False},source_code='fixtureA')
    for changed in [dict(binding,effective_state='y'),dict(binding,input=dict(prompt_ids=[1,3],runtime='same')),dict(binding,generation={'do_sample':True}),dict(binding,source_code='fixtureB')]:
        assert r.c.digest(binding)!=r.c.digest(changed)
    import med133_report as report
    assert report.no_regression([(True,True),(None,True)]) is None
    assert report.no_regression([(False,True),(None,True)]) is False
    assert report.no_regression([(True,True),(False,False)]) is True
    assert report.display_metric(report.summary({1:[None]})).startswith('UNKNOWN')
    assert report.display_metric(report.summary({1:[]})).startswith('NA')
    fake=[];decisions={}
    for row in f['plan']:
        q=row['query'];active={(t['native']['image_sha256'],t['native']['question']) for t in f['tasks'][:row['prefix']]}
        for arm in ('MED','LEGACY','BASE'):
            key=str(len(fake));decisions[key]=dict(core_correct=True,full_supported=arm!='BASE')
            fake.append(dict(arm=arm,mode=row['mode'],prefix=row['prefix'],owner=row['owner'],query_id=q['query_id'],
                payload_key=key,question=q['question'],image_sha256=q['image_sha256'],EOS=True,at_cap=False,
                active_target=row['mode']!='single' and (q['image_sha256'],q['question']) in active))
    result_report,_,_=report.summarize(f,fake,decisions,set(),'B',{})
    assert result_report['panels']['bank_146']['MED']['T2L']['core_correct']['n']==54
    assert result_report['protection']['bank_146']['all_nonnative_evaluation']['full_supported']['zero_new_damage'] is None
    assert result_report['insertion_final_retention']['core_correct']['inserted_correct']==146
    assert all(v['forgotten']==0 for v in result_report['cross_prefix_common_owners'])
    # A missing answer cannot silently become an incorrect or a complete score.
    missing=next(v['payload_key'] for v in fake if v['arm']=='MED' and v['mode']=='single')
    del decisions[missing]
    incomplete,_,_=report.summarize(f,[v for v in fake if v['mode']=='single'],decisions,{missing},'A',{})
    assert incomplete['status']=='INCOMPLETE_SCORING' and incomplete['grading']['unknown_consumers']==1
    result.update(original_T2L_denominator54=True,empty_FULL_support_is_NA=True,unknown_preserved=True,
        same_owner_prefix_comparison=True,insertions_distinct_from_final_retention=True)
    result.update(status='PASS',MED_exact_V132=True,real_prefix_membership=True,changed_state_input_generation_invalidates_cache=True,
        no_GPU_calls=True,no_Judge=True,consumers=len(f['plan']),updates=23360)
    (root/'CPU_ADMISSION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
