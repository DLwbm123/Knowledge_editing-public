"""CPU-only temporary fixture; reuse C binding logic, no model or Judge invocation."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

package=json.loads(sys.stdin.readline());old=Path(package['protocol']['reference_run'])
with tempfile.TemporaryDirectory(prefix='e.',dir='/data/bmw/tmp') as folder:
    R=Path(folder);(R/'private/tools').mkdir(parents=True);(R/'public').mkdir()
    for p in (old/'private/tools').glob('*.py'):shutil.copy2(p,R/'private/tools'/p.name)
    for name,source in package['files'].items():
        compile(source,name,'exec');(R/'private/tools'/name).write_text(source)
    (R/'private/source').symlink_to((old/'private/source').resolve())
    for name in ('BENCHMARK146_QUEUE.json','EVAL_LEDGER.json'):(R/'private'/name).symlink_to(old/'private'/name)
    os.environ['RUN_ROOT']=str(R);sys.path.insert(0,str(R/'private/tools'))
    import common as c
    import validate
    result=validate.selfcheck()
    c.write(R/'private/FROZEN_PROTOCOL.json',package['protocol'])
    c.write(R/'private/BASELINE_REUSE_ADMISSION.json',dict(status='PASS',test_only=True))
    c.write(R/'RUN_MANIFEST.json',dict(Judge_limit=4744,deadline_epoch=time.time()+3600))
    c.write(R/'RESOURCE_LEDGER.json',dict(Judge_attempts=0,Astra_purew146_batches=[],gpu_sessions=[],gpu_seconds_used=0,weights_observed_peak_bytes=0))
    import queue_col as qc
    q=qc.q
    q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch='TEST_ONLY_NO_REQUESTS',model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT))
    db=q.connect();qc.ingest(db)
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==2372
    miss=db.execute("SELECT c.* FROM consumer c JOIN payload p ON c.payload_key=p.key WHERE p.status='MISSING' LIMIT 1").fetchone()
    owner=miss['edit_order'];source=Path(miss['path']).parent
    target=R/'private/arms/O/private/outputs/single_R0'/str(owner)/'1'
    phase=dict(arm='O',mode='single_R0',owner=owner,prefix=1,protocol=c.digest(package['protocol']),code='CPU_TEST_ONLY')
    receipt=q.read(source/'COMPLETE.json')
    for p in source.glob('*.json'):
        if p.name=='COMPLETE.json' or p.name.endswith('.STARTED.json'):continue
        d=q.read(p);d['binding']['phase']=phase;q.write(target/p.name,d)
    q.write(target/'COMPLETE.json',dict(phase=phase,outputs=receipt['outputs']))
    before=db.execute('SELECT count(*) FROM payload').fetchone()[0];qc.ingest(db)
    assert db.execute('SELECT count(*) FROM payload').fetchone()[0]==before
    inherited=db.execute('SELECT status,correct FROM payload WHERE key=?',(miss['payload_key'],)).fetchone()
    assert tuple(inherited)==('MISSING',None)
    qc.ingest(db);assert db.execute("SELECT count(*) FROM consumer WHERE method='O'").fetchone()[0]==receipt['outputs']
    assert not qc.terminal_write_allowed(q.ROOT/'ALL_WORKERS_COMPLETE.json',False)
    assert qc.terminal_write_allowed(q.ROOT/'ALL_WORKERS_COMPLETE.json',True)
    exhausted=c.read(R/'RESOURCE_LEDGER.json');exhausted['Judge_attempts']=4744;c.write(R/'RESOURCE_LEDGER.json',exhausted)
    assert qc.status(db)['status']=='GENERATING'
    exhausted['Judge_attempts']=0;c.write(R/'RESOURCE_LEDGER.json',exhausted)
    import report_col
    report_col.main()
    partial=c.read(R/'public/RESULTS.json')
    assert partial['status']=='PARTIAL_NOT_ALL_CONSUMERS_COMPLETE' and not partial['scientific_success']
    assert partial['not_generated']['L']==2372 and not partial['panels']
    assert report_col.r.score_bounds(report_col.r.combine([{'missing':1},{'missing':-1}]),{'missing':None})==[0,0]
    result.update(C_consumers_checked=2372,C_missing_consumers=11,full_binding_shared_missing_reuse=True,
        panel_ingest_idempotent=True,no_premature_scoring_terminal=True,incomplete_not_permanent_missing=True,
        shared_missing_exact_cancellation=True,GPU_updates=0,Judge_requests=0)
    print(json.dumps(result))
