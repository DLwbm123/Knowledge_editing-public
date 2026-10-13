"""CPU mechanical check; never trains, calls a Judge, or charges the real ledger."""
import json
import os
from pathlib import Path
import tempfile
import time

import torch
import lora146 as stage
import lora146_queue as queue


def main():
    stage.retro.shard_test()
    ts = stage.tasks()
    entries = list(stage.outputs(ts))
    assert len(entries)==len({str(x[0]) for x in entries})==5722
    assert len(stage.retro.query_ids(ts))==1509
    expert = stage.lora.LoRA(17)
    assert sum(p.numel() for p in expert.parameters())==18432
    x = torch.ones(2,14336)
    assert expert.residual(x).count_nonzero()==0
    with torch.no_grad(): expert.B.fill_(.125)
    clone = stage.lora.clone(expert.state_dict(),18,'cpu')
    assert torch.equal(expert.residual(x),clone.residual(x))
    parents = queue.parent_rows()
    assert len(parents)==2137
    q = queue.q
    with tempfile.TemporaryDirectory(prefix='e.',dir=os.environ['TMPDIR']) as directory:
        root = Path(directory)
        q.RUN, q.ROOT = root, root/'queue'
        q.write(root/'RUN_MANIFEST.json',dict(Judge_limit=10**18,Judge_limit_enabled=False,deadline_epoch=time.time()+3600))
        q.write(root/'RESOURCE_LEDGER.json',dict(Judge_attempts=6001))
        db = q.connect()
        for i in range(1,206):
            db.execute('INSERT INTO payload VALUES (?,?,?,?,?,?)',(str(i),json.dumps({'test':i}),'{}','PENDING',None,None))
        db.execute("INSERT INTO done VALUES ('generation')")
        db.commit()
        got = [queue.a.request(db,dict(action='next',worker=w)) for w in range(4)]
        assert all(x['state']['status']=='GENERATION_COMPLETE' for x in got)
        assert [x['rows'][0]['key'] for x in got]==['1','51','101','151']
        queue.a.request(db,dict(action='reserve',worker=0,keys=['1'],batch_id='synthetic'))
        assert q.read(root/'RESOURCE_LEDGER.json')['Judge_attempts']==6002
        try:
            queue.a.request(db,dict(action='reserve',worker=1,keys=['2'],batch_id='wrong-worker'))
        except AssertionError:
            pass
        else:
            raise AssertionError('Cross-worker reservation accepted')
        assert q.read(root/'RESOURCE_LEDGER.json')['Judge_attempts']==6002
        db.close()
    result = dict(status='PASS',tests=['5722 unique consumers','1509 final queries','exhaustive disjoint shards',
        '18432 parameters','zero start','exact state reload','2137 immutable parent scores',
        'disabled Judge cap accepts count above6000','wrong-worker reservation rejected'])
    stage.common.write(stage.RUN/'public/LORA146_CPU_TEST.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
