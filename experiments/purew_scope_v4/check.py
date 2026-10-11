"""Small real-autograd tests of all five interventions, exact resume and score reuse."""
import ast
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import torch
import shared_math as dm
import journal as j
from protocol import specification


def recovery():
    torch.manual_seed(27)
    base=torch.randn(3,9,dtype=torch.float32)*.01
    x=torch.randn(6,9);target=torch.randn(6,3);q0=torch.linalg.qr(base.T,mode='reduced').Q
    c=base.clone();norms=[]
    for step in range(12):
        w=c.detach().requires_grad_();loss=(x@w.T-target).square().mean();g=torch.autograd.grad(loss,w)[0]
        delta=dm.reference_step(dm.direction(g,'C',q0));c,norm=dm.apply_step(c,delta,float(delta.norm()));norms.append(norm)
    def trial(arm,interrupt):
        w=torch.nn.Parameter(base.clone());losses=[];writes=[]
        with tempfile.TemporaryDirectory() as tmp:
            for step,norm in enumerate(norms,1):
                w.grad=None;loss=(x@w.T-target).square().mean()
                if arm=='H' and step>4:loss=loss+.5*(x[:2]@w.T-target[:2]).square().mean()
                loss.backward();new,actual=dm.apply_step(w.detach(),dm.shared_step(w.grad,norm),norm)
                with torch.no_grad():w.copy_(new)
                losses.append(float(loss.detach()));writes.append(w.detach().clone())
                state=dict(weight=w.detach().clone(),last_event=dict(update=step,loss=losses[-1],reference_step_norm=norm))
                torch.save(state,Path(tmp)/'state.pt')
                if interrupt and step==5:
                    restored=torch.load(Path(tmp)/'state.pt',weights_only=True);j.ensure_event(tmp,restored)
                    with torch.no_grad():w.fill_(123);w.copy_(restored['weight'])
                j.ensure_event(tmp,state)
            assert len(list(Path(tmp).glob('*.json')))==12
        return w.detach(),losses,writes
    traces={}
    for arm in ('U','H'):
        a,b=trial(arm,False),trial(arm,True);assert torch.equal(a[0],b[0]) and a[1]==b[1]
        traces[arm]=a[2]
        fresh=torch.nn.Linear(9,3,bias=False)
        with torch.no_grad():fresh.weight.copy_(b[0])
        assert torch.equal(fresh(x),x@b[0].T) and len(list(fresh.parameters()))==1
    assert all(torch.equal(a,b) for a,b in zip(traces['U'][:4],traces['H'][:4]))
    assert not torch.equal(traces['U'][-1],traces['H'][-1])
    return dict(status='PASS',both_resume_bit_exact=True,native_reload=True,first_edit_H_equals_U=True,history_changes_direction=True,shared_targets=True,checkpoint_log_gap_repaired=True)


def inheritance():
    node=next(x for x in ast.parse(Path(__file__).with_name('score.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='inherited')
    with tempfile.TemporaryDirectory() as tmp:
        parent=Path(tmp);ddl='CREATE TABLE payload (key TEXT PRIMARY KEY,record TEXT,binding TEXT,status TEXT,batch TEXT,correct INTEGER)'
        records=[dict(opaque_query_id='old'+str(i),question='synthetic',gold_answer='A',raw_base_answer=str(i)) for i in range(3)]
        for index,(name,queue) in enumerate([('purew-scope-v1-20261009','judge_purew_astra_medium'),('purew-scope-v2-20261010','judge_scope_v2_astra_medium'),('purew-scope-v3-20261010','judge_scope_v3_astra_medium')]):
            folder=parent/name/'run/private'/queue;(folder/'evidence').mkdir(parents=True)
            db=sqlite3.connect(folder/'queue.sqlite');db.execute(ddl);status='MISSING' if index==1 else 'FORMAT_VALID'
            db.execute('INSERT INTO payload VALUES (?,?,?,?,?,?)',(str(index),json.dumps(records[index]),json.dumps({'input':index}),status,'batch',None if index==1 else 1));db.commit();db.close()
            batch={'records':[records[index]]};e=dict(input_binding=json.dumps(batch,sort_keys=True),status=status,actual_model='gpt-6-astra',reasoning_effort='medium',exit_code=0,errors=[],tool_event_types=[],isolation_checks={'all':True})
            j.write(folder/'evidence/batch.json',dict(batch=batch,evidence=e,response=[dict(opaque_query_id='old'+str(index),is_correct=True)] if index!=1 else None))
        q=SimpleNamespace(read=lambda p:json.loads(p.read_text()),digest=lambda v:json.dumps(v,sort_keys=True),validate=lambda b,r:r)
        namespace={'json':json,'Path':lambda _:parent,'q':q};exec(compile(ast.Module(body=[node],type_ignores=[]),'score','exec'),namespace)
        db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row;db.execute(ddl)
        for i in range(3):db.execute('INSERT INTO payload VALUES (?,?,?,?,?,?)',(str(i),json.dumps(dict(records[i],opaque_query_id='new'+str(i))),json.dumps({'input':i}),'PENDING',None,None))
        namespace['inherited'](db);assert [r['status'] for r in db.execute('SELECT * FROM payload ORDER BY key')]==['FORMAT_VALID','MISSING','FORMAT_VALID']
        assert db.execute("SELECT correct FROM payload WHERE key='1'").fetchone()[0] is None
        db.execute("UPDATE payload SET binding='{}' WHERE key='0'")
        try:namespace['inherited'](db)
        except AssertionError:pass
        else:raise AssertionError('changed binding accepted')
    return dict(status='PASS',exact_valid_reused=True,old_failure_not_retried=True,changed_binding_rejected=True)


def checks(parent,reference):
    torch.set_num_threads(1)
    results=dict(math=dm.selfcheck(),journal=j.selfcheck(),recovery=recovery(),score_inheritance=inheritance())
    protocol=specification(parent,results['math'],results['journal'],reference)
    assert protocol==specification(protocol,results['math'],results['journal'],reference)
    assert protocol['GPUS']==[5] and protocol['maximum_updates']==2560 and protocol['consumers']==1395
    for path in Path(__file__).parent.glob('*.py'):compile(path.read_text(),str(path),'exec')
    return dict(status='PASS',checks=results,protocol_idempotent=True),protocol


if __name__=='__main__':
    result,protocol=checks(json.loads(Path(os.environ['PARENT_PROTOCOL']).read_text()),json.loads(Path(os.environ['NORM_REFERENCE']).read_text()));print(json.dumps(result))
    if 'CHECK_OUTPUT' in os.environ:
        j.write(Path(os.environ['CHECK_OUTPUT'])/'CPU_VALIDATION.json',result);j.write(Path(os.environ['CHECK_OUTPUT'])/'PROTOCOL.json',protocol)
