"""Read-only final checks of the frozen run; writes anonymous audit counts only."""
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import torch

RUN = Path(os.environ['RUN_ROOT'])
def read(path):
    return json.loads(path.read_text())
def alive(pid, ticks):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[21] == str(ticks)
    except FileNotFoundError:
        return False

def main():
    ledger = read(RUN/'RESOURCE_LEDGER.json')
    assert ledger['gpu_seconds_used'] <= 24*3600
    assert all(s.get('ended_epoch') and not alive(s['pid'], s['start_ticks']) for s in ledger['gpu_sessions'])
    for name in ('CONTROLLER_START', 'RECOVERY_CONTROLLER_START'):
        start = read(RUN/f'private/{name}.json')
        assert not alive(start['pid'], start['start_ticks'])
    assert read(RUN/'private/CONTROLLER_COMPLETE.json')['P2'] == 'P2B'
    expected = {f'TT88_{arm}_s{s}': 8 for arm in ('DIR', 'MATCH') for s in range(3)}
    expected.update({f'{shape}_{arm}_s0': 8 for shape in ('OUTER', 'MIDDLE') for arm in ('NO_H', 'H1')})
    training = list((RUN/'private/edits').glob('e*/*/TRAINING.json'))
    counts = Counter()
    for path in training:
        d = read(path); counts[d['binding']['arm']] += 1
        assert d['status'] == 'COMPLETE' and d['steps'] == 320 and len(d['curve']) == 320
        assert d['independent_AB_parameters'] == 0
        assert d['binding']['execution']['commit'] == '426ffe24f30c9a28c44fd3f4267430c2bf08ec0f'
        saved = torch.load(path.parent/'final.pt', map_location='cpu', weights_only=True)
        assert set(saved['expert']) == {'G1','G2','G3','G4'} and saved['step'] == 320
        assert saved['binding'] == d['binding']
        assert sum(t.numel() for t in saved['expert'].values()) == d['parameters']
        assert all(torch.isfinite(t).all() for t in saved['expert'].values())
    assert dict(counts) == expected
    assert not list((RUN/'private/edits').glob('e*/*/latest.pt'))
    embeddings = [read(p) for p in (RUN/'private/embedding').glob('*.json')]
    assert len(embeddings) == 16 and all(e['status']=='PASS' and e['generated_tokens_equal'] and all(g>0 for g in e['opening_gradient_norms']) for e in embeddings)
    db = sqlite3.connect('file:'+str(RUN/'private/judge_common/queue.sqlite')+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    payloads = {r['key']:dict(r) for r in db.execute('SELECT * FROM payload')}
    assert len(payloads)==459 and all(p['status']=='FORMAT_VALID' and p['correct'] in (0,1) for p in payloads.values())
    inherited = db.execute('SELECT count(*) FROM inherited').fetchone()[0]
    consumers = db.execute('SELECT count(*) FROM consumer').fetchone()[0]
    assert inherited==257 and ledger['Judge_attempts']==202 and consumers==6596
    old = sqlite3.connect('file:'+str(RUN/'private/recovery/receipt_provenance/queue_before.sqlite')+'?mode=ro', uri=True);old.row_factory=sqlite3.Row
    previous = list(old.execute('SELECT * FROM payload'))
    assert all(payloads[r['key']]==dict(r) for r in previous)
    retained = read(RUN/'private/PARENT_REUSE_INVENTORY.json')['items']
    parent_counts = Counter(Path(i['path']).name for i in retained)
    assert parent_counts == {'W0.pt':96,'final.pt':240,'ROUTER.pt':8}
    assert all(Path(i['path']).is_file() and Path(i['path']).stat().st_size==i['bytes'] for i in retained)
    own = list((RUN/'private/edits').glob('e*/*/final.pt'))
    size = sum(p.stat().st_size for p in own)
    assert size<2*1024**3 and shutil.disk_usage(RUN).free>=8*1024**3
    result = dict(status='PASS',epoch=time.time(),training_runs=len(training),attempts_per_run=320,arm_counts=dict(counts),TT_only_final_files=len(own),TT_final_bytes=size,active_latest=0,embedding_qualifications=len(embeddings),all_owned_processes_ended=True,GPU_hours=ledger['gpu_seconds_used']/3600,score_unique=459,score_inherited=inherited,score_new_requests=202,score_missing=0,score_consumers=consumers,pre_repair_payloads_unchanged=len(previous),parent_retained_counts=dict(parent_counts),parent_check='existence and known size only; no byte identity claim',P1='FAIL_DEV_ONLY',P2='P2B_COMPLETE',P3='BLOCKED_DATA',scientific_status='NOT_CONFIRMED',no_training_reruns=True,no_score_retries=True,no_clock_reset=True)
    (RUN/'public/FINAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':
    main()
