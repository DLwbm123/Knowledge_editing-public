"""Bound endpoints, all requested curves, missing evidence and owned cleanup."""
from collections import Counter
from pathlib import Path
import sqlite3
import torch
from common import RUN,read,write,state_hash,used
from train import clone

def main():
    ledger=read(RUN/'RESOURCE_LEDGER.json');assert all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    finals=list((RUN/'private/edits').glob('P*/*/CE*/step*.pt'));errors=[]
    for p in finals:
        s=torch.load(p,map_location='cpu',weights_only=True);assert set(s['expert'])=={'G1','G2','G3','G4'} and sum(v.numel() for v in s['expert'].values())==7168
        assert all(torch.isfinite(v).all() for v in s['expert'].values());e=clone(s['expert'],17,'cpu');assert state_hash(e)==s['state_hash']
    trains=list((RUN/'private/edits').glob('P*/*/CE*/TRAINING.json'))
    for p in trains:
        tr=read(p);assert tr['actual_updates']==tr['steps']==len(tr['curve'])
        for i,row in enumerate(tr['curve'],1):assert row['step']==i and all(x==i for x in row['Adam_steps'])
        assert ('U_fit' in tr['binding']['task']) and not any(x in tr['binding']['task'] for x in ('H','H_fit','H_eval'))
        assert bool(tr['binding']['task']['U_fit'])==any('U' in row['terms'] for row in tr['curve'])
        for n in tr['binding']['nodes']:assert (p.parent/('step'+str(n)+'.pt')).exists()
    latest=list((RUN/'private/edits').rglob('latest.pt'));assert not latest,'Unfinished active state: cannot call final audit complete'
    db=sqlite3.connect(RUN/'private/judge_common/queue.sqlite');db.row_factory=sqlite3.Row
    counts=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status').fetchall());assert all(k in ('FORMAT_VALID','MISSING') for k in counts)
    attempts=[k for b in ledger['Judge_batches'] for k in b['keys']];assert len(attempts)==len(set(attempts))==ledger['Judge_attempts']<=6000
    for c in db.execute('SELECT * FROM consumer'):
        from common import digest
        assert digest(read(c['path']))==c['output_binding']
    db.close();assert used()<=24*3600
    # All scheduled GPU consumers ended. Preserve raw teacher tokens/bindings for regeneration.
    paths=list((RUN/'private/teacher').glob('*.pt'))+list((RUN/'private/smoke').glob('*.pt'))
    for d in (RUN/'private/edits').glob('P2_*/warmup'):
        if (d/'W0.pt').exists():paths+=[p for p in [d/'native.pt',d/'A2.pt'] if p.exists()]
    items=[]
    for p in paths:
        assert not p.is_symlink() and p.resolve().is_relative_to(RUN.resolve());items.append(dict(path=str(p),bytes=p.stat().st_size,reason='Last registered training/diagnostic consumer ended; shared teacher regenerable from frozen Base; warmup intermediate rebuildable'))
        if p.parent.name=='teacher':
            s=torch.load(p,map_location='cpu',weights_only=True);write(p.with_suffix('.json'),{k:v for k,v in s.items() if k!='logp'})
    write(RUN/'private/FINAL_TEMPORARY_CLEANUP_MANIFEST.json',dict(items=items,consumers='P1/P2 admitted blocks and bank or explicit NOT_RUN; all owned GPU sessions ended',old_assets_touched=False))
    for item in items:Path(item['path']).unlink()
    write(RUN/'private/FINAL_TEMPORARY_CLEANUP_RECEIPT.json',dict(items=items,completed=True,reconstruction='Teacher from frozen Base; generated warmup/smoke from saved source and bindings; no final expert deleted'))
    audit=dict(status='PASS',continuations=len(trains),node_TT_checkpoints=len(finals),active_latest=0,Judge_counts=counts,Judge_attempts=ledger['Judge_attempts'],GPU_hours=used()/3600,owned_cleanup_files=len(items),owned_cleanup_bytes=sum(x['bytes'] for x in items),historical_assets_read_only=True,all_owned_GPU_sessions_ended=True,no_optional_payload_file=(not (RUN/'private/H_AVAILABLE.json').exists()),publication='PENDING_FINAL_SCIENTIFIC_REVIEW_AND_PUBLIC_DELIVERY')
    write(RUN/'public/FINAL_AUDIT.json',audit)
if __name__=='__main__':main()
