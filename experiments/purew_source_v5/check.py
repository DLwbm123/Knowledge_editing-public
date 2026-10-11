"""Small checks for the new role contract and exact shared-norm recovery."""
import ast,json,tempfile
from types import SimpleNamespace
from pathlib import Path
import torch
import cohort,journal,shared_math

def run():
    torch.set_num_threads(1)
    role=cohort.selfcheck();recovery=journal.selfcheck();math=shared_math.selfcheck()
    rows=[dict(query_id=str(i),source_group=str(i),owner=0,question='Where is the lesion?',reference='source',audit_role='CANDIDATE',evaluation_questions=cohort.reformulations('Where is the lesion?')[0],variant_kind='LOCATION_REWRITE') for i in range(84)]
    packet=dict(candidates=rows,new_protection=[])
    chosen,selected=cohort.select(packet,{str(i):0 for i in range(84)})
    assert len(chosen)==24 and len(selected)==120 and len(cohort.training_rows(selected))==24
    for scores in ({str(i):None if i==0 else 0 for i in range(84)},{str(i):1 for i in range(84)}):
        try:cohort.select(packet,scores)
        except ValueError:pass
        else:raise AssertionError('Incomplete source qualification was accepted')
    torch.manual_seed(51);initial=torch.randn(2,6)*.01;x=torch.randn(3,6);y=torch.randn(3,2)
    u=initial.clone();references=[]
    for step in range(18):
        w=u.detach().requires_grad_();loss=(x@w.T-y).square().mean();g=torch.autograd.grad(loss,w)[0]
        proposed=min(.01*float(g.norm()),.05);u,actual=shared_math.apply_step(u,shared_math.shared_step(g,proposed),proposed)
        references.append(dict(actual=actual,proposal=proposed,weight=u.clone()))
    def trial(interrupt):
        w=initial.clone()
        for step,r in enumerate(references):
            z=w.detach().requires_grad_();loss=(x@z.T-y).square().mean()
            if step>=6:loss=loss+.5*(x[:1]@z.T-y[:1]).square().mean()
            g=torch.autograd.grad(loss,z)[0];target=r['proposal'] if step<6 else r['actual']
            w,actual=shared_math.apply_step(w,shared_math.shared_step(g,target),target)
            assert abs(actual-r['actual'])<=1e-6
            if step<6:assert torch.equal(w,r['weight'])
            if interrupt and step==8:
                with tempfile.TemporaryDirectory() as folder:
                    p=Path(folder)/'resume.pt';torch.save(w,p);w=torch.load(p,weights_only=True)
        return w
    assert torch.equal(trial(False),trial(True))
    stage=next(n for n in ast.walk(ast.parse((Path(__file__).parent/'run.py').read_text())) if isinstance(n,ast.FunctionDef) and n.name=='stage')
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);(root/'private').mkdir();(root/'private/START_CHAIN_source_base_0.json').write_text('{}')
        checks=iter((False,True));alive=iter((True,False))
        def forbidden(*args):raise AssertionError('Completed or live stage relaunched')
        namespace=dict(RUN=root,complete=lambda _:next(checks),j=SimpleNamespace(alive=lambda _:next(alive)),c=SimpleNamespace(read=lambda _:dict(owned=True),budget=forbidden),pipeline=SimpleNamespace(launch=forbidden),time=SimpleNamespace(sleep=forbidden))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[stage],type_ignores=[])),'controller_stage','exec'),namespace)
        namespace['stage']('source_base',0,'BASE_COMPLETE.json')
        namespace['complete']=lambda _:True
        namespace['stage']('source_base',0,'BASE_COMPLETE.json')
    for p in Path(__file__).parent.glob('*.py'):compile(p.read_text(),str(p),'exec')
    return dict(status='PASS',test_scope='CPU_SYNTHETIC_ONLY',roles=role,journal=recovery,math=math,first_edit_W_bit_exact=True,H_resume_bit_exact=True,missing_or_insufficient_source_stops=True,real_VLM_admission='NOT_RUN',controller_adopts_live_and_skips_completed=True)

if __name__=='__main__':print(json.dumps(run()))
