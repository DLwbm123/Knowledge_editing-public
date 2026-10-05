"""CPU data, initializer and finite-score qualification before any native GPU work."""
import os,sys,time,json,subprocess,tempfile
from pathlib import Path
from fractions import Fraction
RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
from audit import read,write,digest

def main():
    began=time.time();plan=read(RUN/'PLAN_CONFIG.json');ledger=read(RUN/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json')
    assert digest({k:v for k,v in ledger.items() if k!='freeze_id'})==ledger['freeze_id']
    H=read(RUN/'private/H_AVAILABLE.json');parent=Path(plan['parent_run'])
    assert H==read(parent/'private/H_AVAILABLE.json') and not read(parent/'private/H_SEMANTIC_BINDING_AUDIT.json')['blocked']
    tasks=[t for t in ledger['tasks'] if t['edit_id'] in H]
    assert len(ledger['main_T0'])==146 and [t['order'] for t in tasks]==plan['H8_orders'] and sum(map(len,H.values()))==12
    from admit import relocate
    paths=set()
    for t in tasks:
        assert len(t['fit_questions'])==4 and t['U_fit'] and t['fit_status']=='SUPPORTED'
        paths.add(relocate(t['native']['image_path']));paths.update(relocate(u['image_path']) for u in t['U_fit'])
        for ev in t['events']:paths.update(relocate(ledger['queries'][q]['image_path']) for q in ev['all_probe_query_ids'])
        paths.update(h['image_path'] for h in H[t['edit_id']])
    assert all(Path(x).is_file() for x in paths)
    lock=read(RUN/'private/SOURCE_COMMIT.json')
    for p,commit in [(RUN/'private/source',lock['commit']),(RUN/'private/official_llava',lock['official_source_commit'])]:
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()==commit
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=p,text=True).strip()
    import importlib.util,torch
    spec=importlib.util.spec_from_file_location("cp_init_worker",RUN/"private/tools/worker.py")
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    assert tuple(plan['arms'])==w.ARMS and set(w.legacy.GPUS)=={5,6}
    e=w.direct_zero(123,'cpu');f=w.direct_zero(123,'cpu');g=w.direct_zero(124,'cpu')
    assert torch.equal(e.A,f.A) and not torch.equal(e.A,g.A) and torch.count_nonzero(e.B)==0
    x=torch.randn(3,14336);assert torch.count_nonzero(e.residual(x))==0
    torch.testing.assert_close(e.A.norm(dim=1),torch.ones(4))
    loss=(e.residual(x)-torch.ones(3,4096)).square().mean();loss.backward()
    assert e.B.grad.norm()>0 and e.A.grad is not None
    with torch.no_grad():e.B.normal_()
    before=e.residual(x).detach();e.normalize_factors_();torch.testing.assert_close(before,e.residual(x))
    assert sum(p.numel() for p in e.parameters())==73728
    import reporting as r
    scores={'x':None,'y':1,'z':0};groups={'e1':'s','e2':'s'}
    a={'e1':{'x':Fraction(1)},'e2':{'y':Fraction(1)}};b={'e1':{'x':Fraction(1)},'e2':{'z':Fraction(1)}}
    assert r.paired(a,b,scores,groups)['macro_delta_bounds']==[.5,.5]
    assert r.paired(a,a,scores,groups)['macro_delta_bounds']==[0.,0.]
    assert r.metric([],scores)[0]['macro_bounds'] is None
    assert len(r.PAIRS)==4 and r.ARMS==w.ARMS
    fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py');os.close(fd)
    Path(entry).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['TASK_CHECK'],run_name='__main__')\n")
    subprocess.run([sys.executable,entry],env=dict(os.environ,TASK_CHECK=str(RUN/'private/tools/qwen_selfcheck.py')),stdout=(RUN/'logs/score_selfcheck.log').open('w'),stderr=subprocess.STDOUT,check=True)
    import qwen_queue as q
    db=q.connect();q.initialize(db)
    # Validate recursive inheritance across both completed epochs, including inherited rows.
    q.parent_result({'synthetic_absent_key':True});assert q._PARENT_CACHE and all(row['status'] in ('FORMAT_VALID','MISSING') for row,saved in q._PARENT_CACHE.values())
    db.close()
    spec=importlib.util.spec_from_file_location("cp_init_controller",RUN/"private/tools/controller.py")
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    controller.guard()
    write(RUN/'private/SCORING_IMPLEMENTATION_ADMISSION.json',dict(status='PASS',four_arms=True,full_key_inheritance=True,inherited_catalog_size=len(q._PARENT_CACHE),synthetic_missing_no_retry=True,shared_missing_exact_bounds=True,bootstrap10000=True,formal_attempts=read(RUN/'RESOURCE_LEDGER.json')['Judge_attempts']))
    write(RUN/'private/CPU_ADMISSION.json',dict(status='PASS',original_N=146,evaluated_N=8,H_relations=12,paths_checked=len(paths),CPU_initializer_checks='PASS',source_clean=True,GPU_mechanical='PENDING',seconds=time.time()-began))
    with w.legacy.locked_ledger() as cost:cost['CPU_admission_seconds']=time.time()-began
    print('CPU_ADMISSION_PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        write(RUN/'private/CPU_ADMISSION_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time()));raise
