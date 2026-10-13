"""Existing evidence only: preserve per-occurrence old/new score and route joins."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
from audit import RUN, OLD, read, write, payload, output, digest, sha


def main():
    l=read(OLD/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json');b=read(OLD/'private/legacy_stage17/BINDINGS.json')
    by={t['edit_id']:t for t in l['tasks']};tasks=[by[e] for e in l['main_T0']];H=set(read(OLD/'private/H_AVAILABLE.json'))
    old_bind=read(RUN/'private/history/judge/BINDINGS.json')
    old_scores={r['opaque_query_id']:r['is_correct'] for r in map(json.loads,(RUN/'private/history/judge/VERDICTS_ASTRA.jsonl').read_text().splitlines())}
    old_index=defaultdict(list)
    for key,v in old_bind.items():
        assert digest(v)==key and type(old_scores[key]) is bool
        old_index[digest(payload(v,v,v['output']))].append((key,v))
    db=sqlite3.connect('file:'+str(OLD/'private/judge_common/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    current={r['key']:dict(r) for r in db.execute('SELECT * FROM payload')}
    consumers={(r['mode'],r['prefix'],r['edit_order'],r['folder'],r['query_id']):dict(r) for r in db.execute('SELECT * FROM consumer WHERE method="medtrace"')}
    old_raw={}
    for mode,root in [('single',RUN/'private/history/single'),('sequential',RUN/'private/history/sequential')]:
        for f in root.glob('e*/query_*.json') if mode=='single' else root.glob('e*/panel/*.json'):
            d=read(f);q=d['binding']['input'];parent=next(p for p in f.parents if p.name.startswith('e') and p.name[1:].isdigit());order=int(parent.name[1:])
            old_raw[(mode,1 if mode=='single' else order,order,q['query_id'])]=(f,d)
    rows=[];summary=Counter();transfers={};diagnostics=[]
    for mode,prefix in [('single',1),('sequential',1),('sequential',50),('sequential',100),('sequential',146)]:
        selected=tasks if mode=='single' else tasks[:prefix]
        for t in selected:
            for event in t['events']:
                for qid in event['all_probe_query_ids']:
                    q=l['queries'][qid];key=(mode,prefix,t['order'] if mode=='single' else prefix,'panel',qid)
                    n=consumers.get(key);o=old_raw.get((mode,prefix,t['order'] if mode=='single' else prefix,qid));row=dict(edit_id=t['edit_id'],order=t['order'],query_id=qid,event_id=event['event_id'],task=event['task'],source_group=q['source_group'],H8_owner=t['edit_id'] in H,mode=mode,prefix=prefix,Base_correct=l['Base_correctness'][qid])
                    if n is None or o is None:
                        row.update(category='ORIGINAL_EVIDENCE_UNAVAILABLE',old_raw_available=o is not None,new_raw_available=n is not None);rows.append(row);summary[row['category']]+=1;continue
                    oldpath,od=o;nd=read(n['path']);oo=output(od.get('modes',od)['R0']);no=output(nd['R0']);base=b[q['opaque_Base_id']]
                    oldp=payload(q,base,oo);newp=payload(q,base,no)
                    oldc=[(k,v) for k,v in old_index[digest(oldp)] if v['training']==od['binding']]
                    oldvals={old_scores[k] for k,v in oldc};oc=next(iter(oldvals)) if len(oldvals)==1 else None
                    nc=current[n['payload_key']]['correct']
                    assert json.loads(current[n['payload_key']]['binding'])==newp
                    if 'route' in od: oroute=od['route']
                    else:
                        on=od['route_on']['R0'];assert type(on) is bool and on==(od['distance']<=od['radius_R0'])
                        oroute=dict(logical_edit_id=t['edit_id'] if on else None,nearest_logical_edit_id=t['edit_id'],nearest_distance=od['distance'],radius=od['radius_R0'],activated=on,distance='euclidean')
                    nr=nd['route']; same_input=od['binding']['input']==nd['binding']['input'] and od['binding']['generation']==nd['binding']['generation']
                    same_selection=(oroute['logical_edit_id'],oroute['activated'])==(nr['logical_edit_id'],nr['activated'])
                    category=('OTHER_INPUT_OR_PROTOCOL_BINDING' if not same_input else 'SELECTION_OR_ACTIVATION_CHANGED' if not same_selection else 'SAME_SELECTION_OUTPUT_CHANGED' if oo!=no else 'IDENTICAL_FULL_PAYLOAD_VERDICT_CHANGED' if oc is not None and nc is not None and oc!=bool(nc) else 'IDENTICAL_PAYLOAD_UNCHANGED_OR_MISSING')
                    transition='missing' if oc is None or nc is None else 'correct_to_wrong' if oc and not nc else 'wrong_to_correct' if not oc and nc else 'unchanged'
                    row.update(category=category,transition=transition,old_correct=oc,new_correct=None if nc is None else bool(nc),old_Judge_objects=[k for k,v in oldc],new_Judge_object=n['payload_key'],new_actual_opaque_id=json.loads(current[n['payload_key']]['record'])['opaque_query_id'],old_payload_hash=digest(oldp),new_payload_hash=digest(newp),old_route=oroute,new_route=nr,old_output=oo,new_output=no,old_raw_sha256=sha(oldpath),new_raw_sha256=sha(n['path']),selected_H8=nr['logical_edit_id'] in H)
                    rows.append(row);summary[category]+=1
    with (RUN/'private/HISTORICAL_ATTRIBUTION.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    for mode in ['single','sequential']:
        panel=[r for r in rows if r['mode']==mode and r['prefix']==(1 if mode=='single' else 146) and r['task']=='T2G' and not r['Base_correct']]
        losses=[r for r in panel if r.get('transition')=='correct_to_wrong'];fixes=[r for r in panel if r.get('transition')=='wrong_to_correct']
        transfers[mode]=dict(probes=len(panel),old_correct=sum(r.get('old_correct') is True for r in panel),new_correct=sum(r.get('new_correct') is True for r in panel),losses=len(losses),fixes=len(fixes),missing=sum(r.get('old_correct') is None or r.get('new_correct') is None for r in panel))
        transfers[mode+'_loss_ids']={(r['edit_id'],r['query_id'],r['event_id']) for r in losses}
    overlap=len(transfers.pop('single_loss_ids') & transfers.pop('sequential_loss_ids'))
    single={(r['edit_id'],r['event_id'],r['query_id']):r for r in rows if r['mode']=='single' and r['task']=='T1G' and not r['Base_correct']}
    final=[r for r in rows if r['mode']=='sequential' and r['prefix']==146 and r['task']=='T1G' and not r['Base_correct']]
    damaged=[r for r in final if single[(r['edit_id'],r['event_id'],r['query_id'])].get('new_correct') is True and r.get('new_correct') is False]
    for r in damaged:diagnostics.append(dict(edit_id=r['edit_id'],query_id=r['query_id'],event_id=r['event_id'],reason='PR22_T1G_single_correct_final_wrong'))
    for r in rows:
        if r['task']=='T2G' and r['H8_owner'] and r.get('transition') in ('correct_to_wrong','wrong_to_correct'):diagnostics.append(dict(edit_id=r['edit_id'],query_id=r['query_id'],event_id=r['event_id'],reason='historical_H8_T2G_changed'))
    write(RUN/'private/FROZEN_ORACLE_PRIORITY.json',dict(items=diagnostics,selected_from='existing historical evidence before new training',historical_explanation_requires_reproduction=True))
    t2l=[r for r in rows if r['task']=='T2L' and r['Base_correct'] and r['mode']=='sequential' and r['prefix']==146]
    per=defaultdict(lambda:dict(probes=0,old_correct=0,new_correct=0))
    for r in t2l:
        per[r['edit_id']]['probes']+=1;per[r['edit_id']]['old_correct']+=r.get('old_correct') is True;per[r['edit_id']]['new_correct']+=r.get('new_correct') is True
    write(RUN/'private/HISTORICAL_T2L_PER_EDIT.json',per)
    s=dict(status='COMPLETE_EXISTING_EVIDENCE_ONLY',occurrences=len(rows),categories=dict(summary),T2G=transfers,T2G_loss_occurrence_intersection=overlap,T1G=dict(damage_occurrences=len(damaged),unique_queries=len({r['query_id'] for r in damaged}),edits=len({r['edit_id'] for r in damaged})),T2L=dict(probes=len(t2l),old_correct=sum(v['old_correct'] for v in per.values()),new_correct=sum(v['new_correct'] for v in per.values()),old_macro=sum(v['old_correct']/v['probes'] for v in per.values())/len(per),new_macro=sum(v['new_correct']/v['probes'] for v in per.values())/len(per)),new_Judge_attempts=0,old_scores_read_only=True)
    write(RUN/'public/PHASE_A_ATTRIBUTION.json',s);print(json.dumps(s),flush=True)


if __name__=='__main__':main()
