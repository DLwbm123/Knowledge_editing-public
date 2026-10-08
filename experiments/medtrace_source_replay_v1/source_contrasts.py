"""Supplementary paired source-group uncertainty for the complete CHECK scores."""
import json
import os
import random
import sqlite3
from pathlib import Path


def paired(values):
    rng=random.Random(20261008)
    samples=sorted(sum(rng.choices(values,k=len(values)))/len(values)*100 for _ in range(2000))
    return dict(delta_pp=sum(values)/len(values)*100,source_paired_CI95=[samples[49],samples[1949]])


def main():
    assert paired([0.]*32)==dict(delta_pp=0.,source_paired_CI95=[0.,0.])
    assert paired([1.]*32)==dict(delta_pp=100.,source_paired_CI95=[100.,100.])
    run=Path(os.environ['RUN_ROOT']);root=run/'private/judge_replay_astra_medium'
    assert (run/'private/REPORT_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    scores={x['key']:x['correct'] for x in db.execute('SELECT * FROM payload')}
    mapping={(x['method'],x['query_id']):scores[x['payload_key']] for x in db.execute('SELECT * FROM consumer')}
    rows=json.loads((run/'private/REPLAY_CHECK.json').read_text());groups=sorted({x['source_group'] for x in rows})
    assert len(rows)==96 and len(groups)==32
    contrasts=[]
    for a,b in [('SOURCE_REPLAY192','W0'),('SOURCE_REPLAY192','CE192'),('SOURCE_REPLAY192','BASE')]:
        values=[]
        for group in groups:
            queries=[x['query_id'] for x in rows if x['source_group']==group];assert len(queries)==3
            assert all(mapping[arm,q] in (0,1) for q in queries for arm in (a,b))
            values.append(sum(mapping[a,q]-mapping[b,q] for q in queries)/3)
        contrasts.append(dict(a=a,b=b,source_groups=32,observations=96,**paired(values)))
    result=dict(bootstrap='paired source-group percentile,2000 resamples,seed20261008',contrasts=contrasts,
        scope='forced first32 experts on96 source CHECK queries; development only')
    (run/'public/SOURCE_PAIRED_CONTRASTS.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
