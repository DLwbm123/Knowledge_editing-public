"""CPU regression checks for numerical recovery and inherited Judge provenance."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import torch
import scope_math as sm
import journal as j
from protocol import specification


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def recovery():
    torch.manual_seed(71)
    layer=torch.nn.Linear(12,4,bias=False)
    schema={k:tuple(v.shape) for k,v in layer.state_dict().items()}
    base=layer.weight.detach().clone()
    bases,audit=sm.make_basis(base,3,2,seed=7)
    x=torch.randn(3,12);target=torch.randn(3,4)
    key=torch.randn(12)
    def trajectory(directory,interrupt=False):
        coefficients=tuple(torch.zeros(4,a.shape[0]) for a in bases)
        inverses=tuple(torch.eye(a.shape[0],dtype=torch.float64) for a in bases)
        state={'last_event':None};losses=[]
        with torch.no_grad():layer.weight.copy_(base)
        for step in range(1,10):
            layer.zero_grad();loss=(layer(x)-target).square().mean();loss.backward()
            parts=sm.coordinates(layer.weight.grad,bases,inverses,.4)
            delta,_=sm.clipped_step(parts)
            coefficients=tuple(c+d for c,d in zip(coefficients,delta))
            with torch.no_grad():layer.weight.copy_(sm.merged(base,coefficients,bases))
            if step in (3,6):
                inverses=tuple(sm.recurse(p,a@key,.4) for p,a in zip(inverses,bases))
            event={'update':step,'loss':float(loss.detach())};losses.append(event['loss'])
            state={'coefficients':coefficients,'inverses':inverses,'last_event':event}
            temp=Path(directory)/'state.tmp';torch.save(state,temp);temp.replace(Path(directory)/'state.pt')
            if interrupt and step==4:
                # Recover after numeric commit but before publishing the per-step diagnostic.
                restored=torch.load(Path(directory)/'state.pt',weights_only=True)
                j.ensure_event(Path(directory)/'events',restored)
                coefficients=restored['coefficients'];inverses=restored['inverses']
                with torch.no_grad():layer.weight.fill_(123.)
                with torch.no_grad():layer.weight.copy_(sm.merged(base,coefficients,bases))
            j.ensure_event(Path(directory)/'events',state)
        assert len(list((Path(directory)/'events').glob('*.json')))==9
        assert {k:tuple(v.shape) for k,v in layer.state_dict().items()}==schema
        assert len(list(layer.parameters()))==1 and not any(x.requires_grad for x in coefficients+inverses+bases)
        return layer.weight.detach().clone(),coefficients,inverses,losses
    with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
        full=trajectory(a);resumed=trajectory(b,True)
        assert torch.equal(full[0],resumed[0]) and full[3]==resumed[3]
        assert all(torch.equal(x,y) for xs,ys in zip(full[1:3],resumed[1:3]) for x,y in zip(xs,ys))
        fresh=torch.nn.Linear(12,4,bias=False)
        with torch.no_grad():fresh.weight.copy_(resumed[0])
        assert torch.equal(fresh(x),layer(x))
        assert not fresh._forward_hooks and not fresh._forward_pre_hooks
    return dict(status='PASS',interrupted_trajectory_bit_exact=True,loss_sequence_exact=True,
                native_reload_exact=True,no_added_parameters=True,event_gap_repaired=True,basis=audit)


def inherited_scores():
    source=Path(__file__).with_name('score.py')
    tree=ast.parse(source.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='inherited')
    with tempfile.TemporaryDirectory() as tmp:
        parent=Path(tmp);folder=parent/'private/judge_purew_astra_medium';folder.mkdir(parents=True)
        (folder/'evidence').mkdir()
        prior=sqlite3.connect(folder/'queue.sqlite')
        ddl='CREATE TABLE payload (key TEXT PRIMARY KEY, record TEXT,binding TEXT,status TEXT,batch TEXT,correct INTEGER)'
        prior.execute(ddl)
        record=dict(opaque_query_id='old-id',question='synthetic Q',gold_answer='A',raw_base_answer='A')
        binding={'tokens':[1,2],'runtime':'fixed','judge':'fixed'}
        prior.execute('INSERT INTO payload VALUES (?,?,?,?,?,?)',('key',json.dumps(record),json.dumps(binding),'FORMAT_VALID','batch',1));prior.commit();prior.close()
        batch={'records':[record]}
        evidence=dict(status='FORMAT_VALID',actual_model='gpt-6-astra',reasoning_effort='medium',exit_code=0,errors=[],tool_event_types=[],isolation_checks={'a':True},input_binding=digest(batch))
        path=folder/'evidence/batch.json'
        j.write(path,dict(batch=batch,evidence=evidence,response=[{'opaque_query_id':'old-id','is_correct':True}]))
        q=SimpleNamespace(read=lambda p:json.loads(p.read_text()),digest=digest,validate=lambda b,r:r)
        namespace={'PARENT':parent,'Path':Path,'json':json,'q':q}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
        def current(value):
            db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row;db.execute(ddl)
            db.execute('INSERT INTO payload VALUES (?,?,?,?,?,?)',('key',json.dumps(dict(record,opaque_query_id='new-id')),json.dumps(value),'PENDING',None,None));return db
        db=current(binding);namespace['inherited'](db)
        row=db.execute('SELECT * FROM payload').fetchone()
        assert row['status']=='FORMAT_VALID' and row['correct']==1 and json.loads(row['record'])==record
        db=current(dict(binding,tokens=[3]))
        try:namespace['inherited'](db)
        except AssertionError:pass
        else:raise AssertionError('changed full binding inherited')
        evidence['tool_event_types']=['unexpected-tool'];j.write(path,dict(batch=batch,evidence=evidence,response=[]))
        try:namespace['inherited'](current(binding))
        except AssertionError:pass
        else:raise AssertionError('unisolated Judge inherited')
    return dict(status='PASS',new_opaque_id_reuses_exact_score=True,changed_tokens_rejected=True,unisolated_evidence_rejected=True)


def checks(parent):
    results=dict(math=sm.selfcheck(),journal=j.selfcheck(),recovery=recovery(),scoring_inheritance=inherited_scores())
    results['completed_controller']=completed_controller()
    protocol=specification(parent,results['math'],results['journal'])
    assert protocol==specification(protocol,results['math'],results['journal'])
    assert protocol['GPUS']==[3,4] and protocol['maximum_updates']==2560
    assert protocol['maximum_generations']==754 and protocol['consumers']==1019
    assert protocol['maximum_Judge']==752 and protocol['round_one_read_only']
    for path in Path(__file__).parent.glob('*.py'):compile(path.read_text(),str(path),'exec')
    return dict(status='PASS',checks=results,protocol_idempotent=True),protocol


def completed_controller():
    import sys
    tree=ast.parse(Path(__file__).with_name('run.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    with tempfile.TemporaryDirectory() as tmp:
        # Executing the real controller function must skip all GPU dispatch on completed reentry.
        pipeline=SimpleNamespace(launch=lambda *a,**k:(_ for _ in ()).throw(AssertionError('completed stage launched')))
        prior=sys.modules.get('pipeline');sys.modules['pipeline']=pipeline
        try:
            namespace={'RUN':Path(tmp),'j':j,'is_done':lambda n:n=='CONTROLLER_COMPLETE.json'}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'<controller>','exec'),namespace)
            namespace['controller']();namespace['controller']()
        finally:
            if prior is None:sys.modules.pop('pipeline')
            else:sys.modules['pipeline']=prior
    return dict(status='PASS',reentry_launches_zero_workers=True)


if __name__=='__main__':
    parent=json.loads(Path(os.environ['PARENT_PROTOCOL']).read_text())
    result,protocol=checks(parent)
    print(json.dumps(result,indent=2))
    if 'CHECK_OUTPUT' in os.environ:
        j.write(Path(os.environ['CHECK_OUTPUT'])/'CPU_VALIDATION.json',result)
        j.write(Path(os.environ['CHECK_OUTPUT'])/'PROTOCOL.json',protocol)
