"""Anonymous old/new comparison; terminal scores and stored generations only."""
import contextlib
import io
import json
import os
from pathlib import Path
import sqlite3
import statistics
from . import report


def lengths(rows):
    values = [x['tokens'] for x in rows]
    return dict(count=len(rows),median=statistics.median(values) if values else None,
                maximum=max(values) if values else None,cap_hits=sum(x['tokens']>=x['cap'] for x in rows),
                cap_hits_wrong=sum(x['tokens']>=x['cap'] and x['correct']==0 for x in rows),
                cap_hits_correct=sum(x['tokens']>=x['cap'] and x['correct']==1 for x in rows))


def long_outputs(root):
    read = lambda p:json.loads(p.read_text())
    sixth = read(root/'private/BENCHMARK146_QUEUE.json')['tasks'][5]
    protection = set(read(root/'private/protection_ids.json'))
    ids = {sixth['edit_id']} | {q for e in sixth['events'] if e['task'] in ('T1G','T2G') for q in e['all_probe_query_ids']}
    db = sqlite3.connect('file:'+str(root/'private/judge_fixed8/queue.sqlite')+'?mode=ro',uri=True)
    db.row_factory = sqlite3.Row
    rows = db.execute('SELECT c.*,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key').fetchall()
    output = {}
    for arm in ('U','P'):
        output[arm] = {}
        for mode, prefix in [('base',0),('prefix',4),('prefix',8),('insertion',6)]:
            stage = {}
            for group, selected in [('protection',protection),('source6',ids)]:
                measurements = []
                for row in rows:
                    if (row['method'],row['mode'],row['prefix']) == (arm,mode,prefix) and row['query_id'] in selected:
                        d = read(Path(row['path']))
                        measurements.append(dict(tokens=len(d['R0']['raw_token_ids']),cap=d['binding']['judge_input']['generation']['max_new_tokens'],correct=row['correct']))
                stage[group] = lengths(measurements)
            output[arm][mode+str(prefix)] = stage
    db.close()
    return output


def main():
    current = Path(os.environ['RUN_ROOT'])
    old = current.parent.parent/'fixed-input-guard8-v1-20261016/run'
    assert current != old and current.name == 'run'
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):report.main()
    new_results = json.loads(stdout.getvalue())
    old_results = json.loads((old/'public/completion/RESULTS.json').read_text())
    print(json.dumps(dict(status='TERMINAL_COMPARISON',old=old_results,new=new_results,
                          long_outputs=dict(old=long_outputs(old),new=long_outputs(current)),
                          definitions=dict(long_output='raw token count reaches bound max_new_tokens; no EOS-cause claim',
                                           source6_forgetting='insertion versus final T0/T2G; T1G final only',
                                           missing='unknown; never replaced with zero or retried')),
                     indent=2))


if __name__=='__main__':
    if os.environ.get('SELF_CHECK'):
        assert lengths([dict(tokens=2,cap=4,correct=1),dict(tokens=4,cap=4,correct=0)]) == dict(count=2,median=3.,maximum=4,cap_hits=1,cap_hits_wrong=1,cap_hits_correct=0)
        assert lengths([])['median'] is None
        print('PASS: long-output denominators and empty groups')
    else:main()
