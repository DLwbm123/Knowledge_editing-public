"""CPU replay of stored training-support keys; retrospective, no online-selection claim."""
import sys,json
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def run(common,out,prior):
    import torch
    from scripts.medtrace.stage18_score import query_id
    from m3bench_repro.editors.routing import MemoryRouter,euclidean_distances
    common=Path(common);out=Path(out)
    read=lambda p:json.loads(p.read_text())
    stream=read(common/'private/STREAM.json');selection=read(common/'private/H_SELECTION_FREEZE.json');features=torch.load(common/'private/H_COVERAGE_FEATURES.pt',map_location='cpu',weights_only=True)['features'];bank=torch.load(Path(prior)/'private/BANKS.pt',map_location='cpu',weights_only=False)
    ids=[t['canonical_edit_id'] for t in stream['tasks']];position={e:i+1 for i,e in enumerate(ids)}
    routers={n:MemoryRouter.from_state(dict(distance='euclidean',entries=[bank['routes'][e] for e in ids[:n]]),device='cpu') for n in range(1,46)}
    rows=[];missing=[]
    for task,chosen in zip(stream['tasks'],selection['rows']):
        n=task['order'];edit=task['canonical_edit_id'];route=bank['routes'][edit]
        assert chosen['edit_id']==edit
        for arm,qs in [('S0',[query_id(r) for r in task['H_fit']]),('S1',chosen['S1']),('S2',chosen['S2'])]:
            for q in qs:
                if q not in features:missing.append(dict(position=n,support=q,arm=arm,reason='No stored key; no GPU reconstruction scheduled'));continue
                key=features[q]['key'];distance=float(euclidean_distances(route['key'],key)[0]);history={}
                for prefix in sorted({n,19,32,45}):
                    if prefix<n:continue
                    d=routers[prefix].route(key);history[str(prefix)]=dict(winner=position.get(d.logical_edit_id,0),nearest=position.get(d.nearest_logical_edit_id,0),active=d.activated,supervised_wins=d.logical_edit_id==edit)
                rows.append(dict(position=n,support=q,arm=arm,inside_own=distance<=route['radius'],history=history))
    pub=[]
    for arm in ('S0','S1','S2'):
        for n in (19,32,45):
            subset=[r for r in rows if r['arm']==arm and r['position']<=n];valid=[r for r in subset if str(n) in r['history']]
            pub.append(dict(arm=arm,bank_N=n,slots=len(valid),own_radius=sum(r['inside_own'] for r in valid),actual_supervised_wins=sum(r['history'][str(n)]['supervised_wins'] for r in valid),winner_changed_since_insertion=sum(r['history'][str(r['position'])]['winner']!=r['history'][str(n)]['winner'] for r in valid),actual_winner_distribution=dict(Counter(r['history'][str(n)]['winner'] for r in valid))))
    (out/'private/SUPPORT_OWNERSHIP_ROWS.json').write_text(json.dumps(dict(rows=rows,missing=missing),indent=2)+'\n')
    summary=dict(rows=pub,missing_key_slots=len(missing),computed_on='CPU from existing frozen GPU keys',scope='Retrospective bank comparison; future keys not used in any online training or selection',no_new_GPU_or_Judge=True)
    (out/'public/SUPPORT_OWNERSHIP_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':run(*sys.argv[1:])
