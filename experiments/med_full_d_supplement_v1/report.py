"""Publish aggregate MED-versus-MED+D results without source content."""
import json
import statistics
from pathlib import Path

ROLES=('T0','FIT','T1G','T2G','T1L','T2L')
FIELDS=('core_correct','full_supported')


def aggregate(items):
    by={}
    for owner,value in items:
        if value not in (True,False,None):raise ValueError('Invalid score')
        by.setdefault(owner,[]).append(value)
    unknown=sum(v is None for _,v in items)
    return dict(events=len(items),owners=len(by),correct=sum(v is True for _,v in items),unknown=unknown,
                macro_percent=None if not by or unknown else 100*statistics.mean(statistics.mean(v) for v in by.values()))


def run(root,worktree):
    import score_index, stream_extract as stream, prior_binding
    root,worktree=Path(root),Path(worktree)
    packet=json.loads((root/'PACKET_PRIVATE.json').read_text())
    decisions=json.loads((root/'DECISIONS.json').read_text())['decisions']
    if len(packet['consumers'])!=371:raise ValueError('Packet coverage')
    owners={x['owner'] for x in packet['consumers']}
    if len(owners)!=9:raise ValueError('Owner coverage')
    private=worktree/'.private/med_fixed_depth_v1_20261030'
    baseline=[]
    with (private/'BASE_M0_SCORE_BINDINGS.json').open('rb') as f:
        if stream.nonspace(f)!=b'{':raise ValueError('Expected baseline object')
        while True:
            token=stream.nonspace(f)
            if token==b'}':break
            name=stream.string(f,True)
            if stream.nonspace(f)!=b':':raise ValueError('Expected colon')
            token=stream.nonspace(f)
            if name!='rows':score_index.skip(f,token)
            else:
                if token!=b'[':raise ValueError('Expected array')
                while True:
                    token=stream.nonspace(f)
                    if token==b']':break
                    start=f.tell()-1; identity=prior_binding.select(f,token,{'owner':True}); end=f.tell()
                    if identity.get('owner') in owners:
                        f.seek(start);baseline.append(json.loads(f.read(end-start)));f.seek(end)
                    token=stream.nonspace(f)
                    if token==b']':break
                    if token!=b',':raise ValueError('Invalid array')
            token=stream.nonspace(f)
            if token==b'}':break
            if token!=b',':raise ValueError('Invalid object')
    base={(x['owner'],x['query_id']):x['MED'] for x in baseline}
    candidate={(x['owner'],x['query_id']):decisions.get(x['key']) for x in packet['consumers']}
    if len(base)!=371 or set(base)!=set(candidate):raise ValueError('Baseline coverage mismatch')
    roles=json.loads((private/'EVALUATION_ROLES.json').read_text())
    panels={}
    for role in ROLES:
        selected=[(int(o),q) for o,groups in roles.items() if int(o) in owners for q in groups[role]]
        if any(x not in base for x in selected):raise ValueError('Role outside fixed panel')
        panels[role]={}
        for field in FIELDS:
            def value(d,key):return None if d[key] is None else d[key].get(field)
            med=[(o,value(base,(o,q))) for o,q in selected]
            cand=[(o,value(candidate,(o,q))) for o,q in selected]
            pairs=[(m[1],d[1]) for m,d in zip(med,cand)]
            panels[role][field]=dict(MED=aggregate(med),MED_D=aggregate(cand),
                gains=sum(a is False and b is True for a,b in pairs),losses=sum(a is True and b is False for a,b in pairs),
                unknown_pairs=sum(a is None or b is None for a,b in pairs))
    gates={}
    for role in ('T0','FIT','T1G','T1L','T2L'):
        panel=panels[role]['core_correct'];a=panel['MED']['macro_percent'];b=panel['MED_D']['macro_percent']
        gates[role]=None if a is None or b is None else b>=a-(5 if role=='T2L' else 0)-1e-10
    tg=panels['T2G']['core_correct']; med_tg=tg['MED']['macro_percent']; candidate_tg=tg['MED_D']['macro_percent']
    gates['T2G_improved']=None if med_tg is None or candidate_tg is None else candidate_tg > med_tg + 1e-10
    status='DEV_FAIL' if any(v is False for v in gates.values()) else ('INCONCLUSIVE' if any(v is None for v in gates.values()) else 'DEV_SIGNAL')
    out=dict(status=status,events=371,panels=panels,gates=gates,scoring=json.loads((root/'COMPLETE.json').read_text()),
             scope='nine-editor exposed development panel; protected-derived curvature authorized; T1L NA')
    dest=worktree/'reports/med_full_d_supplement_v1_20261102';dest.mkdir(exist_ok=True,parents=True)
    (dest/'RESULTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    def fmt(x):return 'NA/UNKNOWN' if x is None else f'{x:.4f}%'
    lines=['# 完整 MED＋D：九编辑开发结果',f'状态：{status}。','','|角色|MED CORE|MED＋D CORE|未知配对|','|---|---:|---:|---:|']
    for role in ROLES:
        v=panels[role]['core_correct'];lines.append(f"|{role}|{fmt(v['MED']['macro_percent'])}|{fmt(v['MED_D']['macro_percent'])}|{v['unknown_pairs']}|")
    lines += ['', '弱医学词面是训练代理；该面板已曝光。原保护曲率使用含用户授权的 V95/V106 衍生来源，不能将其视为本实验的隔离保留。T1L空支持为NA。', '按用户授权对历史状态无法核实的固定载荷最多补评一次；历史评分冲突保留UNKNOWN。']
    (dest/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    (root/'REPORT_COMPLETE.json').write_text(json.dumps(dict(status=status)))
