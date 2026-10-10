"""Small real-autograd tests of all five interventions, exact resume and score reuse."""
import ast
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import torch
import diagnostic_math as dm
import propagation_math as sm
import journal as j
from protocol import specification


def recovery():
    torch.manual_seed(27);base=torch.randn(3,9,dtype=torch.float64);x=torch.randn(6,9,dtype=base.dtype);target=torch.randn(6,3,dtype=base.dtype)
    bases,_=sm.make_basis(base,3,1,27);q0=dm.row_basis(base)
    def trial(arm,interrupt):
        w=torch.nn.Parameter(base.clone());coef=tuple(torch.zeros(3,a.shape[0],dtype=base.dtype) for a in bases);p=tuple(torch.eye(a.shape[0],dtype=base.dtype) for a in bases);boundary=base.clone();losses=[]
        with tempfile.TemporaryDirectory() as tmp:
            for step in range(1,13):
                if step%4==1:boundary=w.detach().clone()
                q=dm.row_basis(boundary) if arm=='D_FULL_REFRESH' else q0
                w.grad=None;loss=(x@w.T-target).square().mean();loss.backward();g=w.grad
                d=dm.direction(g,arm,bases,p,q);delta,_=dm.step(d,arm,dm.full(g,q))
                if isinstance(delta,torch.Tensor):new=w.detach()+delta
                else:coef=tuple(c+v for c,v in zip(coef,delta));new=sm.merged(base,coef,bases)
                with torch.no_grad():w.copy_(new)
                if arm=='E_RLS' and step%4==0:p=tuple(sm.recurse(pp,a@x[0],.25) for pp,a in zip(p,bases))
                losses.append(float(loss.detach()));state=dict(weight=w.detach().clone(),coefficients=coef,inverses=p,boundary=boundary,last_event={'update':step,'loss':losses[-1]})
                torch.save(state,Path(tmp)/'state.pt')
                if interrupt and step==5:
                    restored=torch.load(Path(tmp)/'state.pt',weights_only=True);j.ensure_event(tmp,restored)
                    with torch.no_grad():w.fill_(123.);w.copy_(restored['weight'])
                    coef=restored['coefficients'];p=restored['inverses'];boundary=restored['boundary']
                j.ensure_event(tmp,state)
            assert len(list(Path(tmp).glob('*.json')))==12
        return w.detach(),losses
    for arm in dm.ARMS:
        a,b=trial(arm,False),trial(arm,True);assert torch.equal(a[0],b[0]) and a[1]==b[1]
        fresh=torch.nn.Linear(9,3,bias=False,dtype=base.dtype)
        with torch.no_grad():fresh.weight.copy_(b[0])
        assert torch.equal(fresh(x),x@b[0].T) and len(list(fresh.parameters()))==1
    return dict(status='PASS',all5_resume_bit_exact=True,all5_native_reload=True,independent_event_gap=True)


def inheritance():
    node=next(x for x in ast.parse(Path(__file__).with_name('score.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='inherited')
    with tempfile.TemporaryDirectory() as tmp:
        parent=Path(tmp);ddl='CREATE TABLE payload (key TEXT PRIMARY KEY,record TEXT,binding TEXT,status TEXT,batch TEXT,correct INTEGER)'
        records=[dict(opaque_query_id='old'+str(i),question='synthetic',gold_answer='A',raw_base_answer=str(i)) for i in range(2)]
        for index,(name,queue) in enumerate([('purew-scope-v1-20261009','judge_purew_astra_medium'),('purew-scope-v2-20261010','judge_scope_v2_astra_medium')]):
            folder=parent/name/'run/private'/queue;(folder/'evidence').mkdir(parents=True)
            db=sqlite3.connect(folder/'queue.sqlite');db.execute(ddl);status='FORMAT_VALID' if index==0 else 'MISSING'
            db.execute('INSERT INTO payload VALUES (?,?,?,?,?,?)',(str(index),json.dumps(records[index]),json.dumps({'input':index}),status,'batch',1 if index==0 else None));db.commit();db.close()
            batch={'records':[records[index]]};e=dict(input_binding=json.dumps(batch,sort_keys=True),status=status,actual_model='gpt-6-astra',reasoning_effort='medium',exit_code=0,errors=[],tool_event_types=[],isolation_checks={'all':True})
            j.write(folder/'evidence/batch.json',dict(batch=batch,evidence=e,response=[dict(opaque_query_id='old0',is_correct=True)] if index==0 else None))
        q=SimpleNamespace(read=lambda p:json.loads(p.read_text()),digest=lambda v:json.dumps(v,sort_keys=True),validate=lambda b,r:r)
        namespace={'json':json,'Path':lambda _:parent,'q':q};exec(compile(ast.Module(body=[node],type_ignores=[]),'score','exec'),namespace)
        db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row;db.execute(ddl)
        for i in range(2):db.execute('INSERT INTO payload VALUES (?,?,?,?,?,?)',(str(i),json.dumps(dict(records[i],opaque_query_id='new'+str(i))),json.dumps({'input':i}),'PENDING',None,None))
        namespace['inherited'](db);assert [r['status'] for r in db.execute('SELECT * FROM payload ORDER BY key')]==['FORMAT_VALID','MISSING']
        assert db.execute("SELECT correct FROM payload WHERE key='1'").fetchone()[0] is None
        db.execute("UPDATE payload SET binding='{}' WHERE key='0'")
        try:namespace['inherited'](db)
        except AssertionError:pass
        else:raise AssertionError('changed binding accepted')
    return dict(status='PASS',exact_valid_reused=True,old_failure_not_retried=True,changed_binding_rejected=True)


def checks(parent):
    torch.set_num_threads(1)
    results=dict(math=dm.selfcheck(),journal=j.selfcheck(),recovery=recovery(),score_inheritance=inheritance())
    protocol=specification(parent,results['math'],results['journal'])
    assert protocol==specification(protocol,results['math'],results['journal'])
    assert protocol['GPUS']==[5] and protocol['maximum_updates']==6400 and protocol['consumers']==2147
    for path in Path(__file__).parent.glob('*.py'):compile(path.read_text(),str(path),'exec')
    return dict(status='PASS',checks=results,protocol_idempotent=True),protocol


if __name__=='__main__':
    result,protocol=checks(json.loads(Path(os.environ['PARENT_PROTOCOL']).read_text()));print(json.dumps(result))
    if 'CHECK_OUTPUT' in os.environ:
        j.write(Path(os.environ['CHECK_OUTPUT'])/'CPU_VALIDATION.json',result);j.write(Path(os.environ['CHECK_OUTPUT'])/'PROTOCOL.json',protocol)
