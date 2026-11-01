"""Aggregate nine matched editors with no public identities or source text."""
import json,statistics,sys
from pathlib import Path
ROLES=('T0','FIT','T1G','T2G','T1L','T2L')
FIELDS=('core_correct','full_supported')
ARMS=('BASE','MED','A','B','C','D')

def aggregate(rows):
    by={}
    for owner,value in rows:
        if value not in (True,False,None):raise ValueError('Invalid score')
        by.setdefault(owner,[]).append(value)
    unknown=sum(v is None for _,v in rows)
    macro=None if not by or unknown else 100*statistics.mean(statistics.mean(v) for v in by.values())
    return dict(events=len(rows),owners=len(by),correct=sum(v is True for _,v in rows),unknown=unknown,macro_percent=macro)

def contrast(pairs):
    ok=[(a,b) for a,b in pairs if a is not None and b is not None]
    return dict(events=len(pairs),unknown=len(pairs)-len(ok),gains=sum(a is False and b is True for a,b in ok),losses=sum(a is True and b is False for a,b in ok))

def evaluate(packet,decisions,baseline,roles,oldpacket,olddecisions):
    if len(packet['consumers'])!=1113:raise ValueError('Expected all candidate events')
    owners={r['owner'] for r in packet['consumers']}
    if len(owners)!=9:raise ValueError('Owner count mismatch')
    values={}
    for row in baseline:
        if row['owner'] in owners:
            for arm in ('BASE','MED'):values[(row['owner'],row['query_id'],arm)]=row[arm]
    for row in oldpacket['consumers']:
        if row['owner'] in owners and row['arm']=='C21':
            values[(row['owner'],row['query_id'],'A')]=olddecisions.get(row['key'])
    for row in packet['consumers']:
        key=(row['owner'],row['query_id'],row['arm'])
        if key in values:raise ValueError('Duplicate event')
        values[key]=decisions.get(row['key'])
    events={(o,q) for (o,q,a) in values if a=='A'}
    if len(events)!=371 or any((o,q,a) not in values for o,q in events for a in ARMS):
        raise ValueError('Paired event coverage mismatch')
    def score(o,q,a,f):
        x=values[(o,q,a)]
        return None if x is None else x.get(f)
    panels={}
    for role in ROLES:
        selected=[(int(o),q) for o,groups in roles.items() if int(o) in owners for q in groups[role]]
        if any(e not in events for e in selected):raise ValueError('Role event outside frozen panel')
        panels[role]={}
        for field in FIELDS:
            row={a:aggregate([(o,score(o,q,a,field)) for o,q in selected]) for a in ARMS}
            row['D_vs_A']=contrast([(score(o,q,'A',field),score(o,q,'D',field)) for o,q in selected])
            row['D_vs_B']=contrast([(score(o,q,'B',field),score(o,q,'D',field)) for o,q in selected])
            row['D_vs_C']=contrast([(score(o,q,'C',field),score(o,q,'D',field)) for o,q in selected])
            panels[role][field]=row
    gates={}
    for role in ('T0','FIT','T1G','T2L'):
        x=panels[role]['core_correct'];d=x['D']['macro_percent'];m=x['MED']['macro_percent']
        gates[role]=None if d is None or m is None else d>=m-(5 if role=='T2L' else 0)-1e-10
    tg=panels['T2G']['core_correct'];d=tg['D']['macro_percent']
    gates['D_vs_B_C_T2G']=None if d is None or any(tg[x]['macro_percent'] is None for x in ('B','C')) else all(d>tg[x]['macro_percent'] for x in ('B','C'))
    return dict(status='INCONCLUSIVE' if any(x is None for x in gates.values()) else ('DEV_SIGNAL' if all(gates.values()) else 'DEV_FAIL'),
                panels=panels,gates=gates,t1l='NA',scope='nine-editor exposed development panel',
                evaluation_events=371,scorer_snapshot='unknown; user accepted comparison')

def publish(root,worktree):
    root,worktree=Path(root),Path(worktree)
    if any(json.loads((root/name).read_text())['code']!=0 for name in ('DRIVER_EXIT.json','GUARD_EXIT.json')):
        raise ValueError('Scoring chain incomplete')
    packet=json.loads((root/'PACKET_PRIVATE.json').read_text())
    judgments=json.loads((root/'DECISIONS.json').read_text())['decisions']
    oldroot=Path('/private/tmp/.q.fxdscore')
    sys.path[:0]=[str(worktree/'experiments/med_fixed_depth_v1'),str(worktree/'experiments/med150_source_review_v1')]
    import stream_extract as stream
    import score_index
    import prior_binding
    import ledger
    owners={r['owner'] for r in packet['consumers']}
    def selected_array(path,field,wanted,select):
        out=[]
        with path.open('rb') as f:
            if stream.nonspace(f)!=b'{':raise ValueError('Expected object')
            while True:
                token=stream.nonspace(f)
                if token==b'}':break
                name=stream.string(f,True)
                if stream.nonspace(f)!=b':':raise ValueError('Expected colon')
                token=stream.nonspace(f)
                if name==field:
                    if token!=b'[':raise ValueError('Expected array')
                    while True:
                        token=stream.nonspace(f)
                        if token==b']':break
                        start=f.tell()-1
                        ident=prior_binding.select(f,token,wanted);end=f.tell()
                        if select(ident):
                            f.seek(start);out.append(json.loads(f.read(end-start)));f.seek(end)
                        token=stream.nonspace(f)
                        if token==b']':break
                        if token!=b',':raise ValueError('Invalid delimiter')
                else:score_index.skip(f,token)
                token=stream.nonspace(f)
                if token==b'}':break
                if token!=b',':raise ValueError('Invalid object delimiter')
        return out
    consumers=selected_array(oldroot/'PACKET_PRIVATE.json','consumers',{'owner':True,'arm':True},
                             lambda x:x.get('owner') in owners and x.get('arm')=='C21')
    oldpacket={'consumers':consumers}
    oldjudgments=ledger.selected_decisions(oldroot/'DECISIONS.json',{x['key'] for x in consumers})
    private=worktree/'.private/med_fixed_depth_v1_20261030'
    baseline=selected_array(private/'BASE_M0_SCORE_BINDINGS.json','rows',{'owner':True},lambda x:x.get('owner') in owners)
    roles=json.loads((private/'EVALUATION_ROLES.json').read_text())
    result=evaluate(packet,judgments,baseline,roles,oldpacket,oldjudgments)
    result['scoring']=json.loads((root/'COMPLETE.json').read_text())
    dest=worktree/'reports/med_proposition_generalization_v1_20261101'
    (dest/'NINE_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 九编辑同支持对照',f"状态：{result['status']}。",'','| 角色 | MED | A | B | C | D |','|---|---:|---:|---:|---:|---:|']
    for role in ROLES:
        x=result['panels'][role]['core_correct']
        def fmt(v):return 'NA/UNKNOWN' if v is None else f'{v:.4f}%'
        lines.append('| '+role+' | '+' | '.join(fmt(x[a]['macro_percent']) for a in ('MED','A','B','C','D'))+' |')
    lines+=['','数值为编辑宏平均；分母、FULL及配对增损见结果文件。T1L无支持；开发集已曝光。',
            '匹配项是弱词面代理，并非已确认的医学关系。原C21没有原MED保护曲率；评分历史不完整，未知历史载荷本轮最多一次，存在潜在历史重复。']
    (dest/'NINE_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    (root/'REPORT_COMPLETE.json').write_text(json.dumps(dict(status=result['status'])))
