"""Terminal paired146 report; full masks and missing values retain denominators."""
from collections import Counter
import json,os,sqlite3,time
from pathlib import Path
from configuration import ARMS,PREFIXES,ELIGIBLE,source_gate
from panels import protection_ids


def main():
    import astra_report as ar
    from report_fs import audit_requests
    from logic import transitions
    R,c,r=ar.RUN,ar.common,ar.r
    assert c.read(R/'private/GENERATION_COMPLETE.json')['new_consumers']==8246
    assert (R/'private/judge_purew146/ALL_WORKERS_COMPLETE.json').exists()
    assert (R/'private/FINAL_CONSUMED.json').exists()
    cost=c.read(R/'RESOURCE_LEDGER.json');assert all(s.get('ended_epoch') for s in cost['gpu_sessions'])
    db=sqlite3.connect('file:'+str(R/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    rows=list(db.execute('SELECT * FROM consumer'));assert len(rows)==8246
    assert Counter(x['method'] for x in rows)==dict(HM=4123,HE=4123)
    payloads={x['key']:x for x in db.execute('SELECT * FROM payload')};assert all(x['status'] in ('FORMAT_VALID','MISSING') for x in payloads.values())
    scores={k:x['correct'] for k,x in payloads.items()}
    audit_requests(payloads,cost)
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(R/'private/EVAL_LEDGER.json')
    records=[];lookup={};coverage={a:Counter() for a in ARMS}
    protocol=c.digest(c.read(R/'private/FROZEN_PROTOCOL.json'))
    for row in rows:
        d=c.read(row['path']);bb=d['binding'];phase=bb['phase']
        assert c.digest(d)==row['output_binding'] and phase['protocol']==protocol
        assert phase['code']==c.read(R/'private/GPU_SOURCE_VERSION.json')['commit']
        query=ledger['queries'][row['query_id']];assert bb['input']==query
        assert json.loads(payloads[row['payload_key']]['binding'])['output']==d['R0']
        active={(t['native']['image_sha256'],t['native']['question']) for t in tasks[:row['prefix']]} if row['mode']=='bank_R0' else set()
        records.append((dict(binding=dict(phase=dict(arm=row['method'],node=0,prefix=row['prefix']),input=query,
            mode=row['mode'],owner_order=row['edit_order']),active_target=(query['image_sha256'],query['question']) in active,effective_expert=None),row['payload_key']))
        lookup[row['method'],row['mode'],row['prefix'],row['edit_order'],row['query_id']]=row['payload_key']
        coverage[row['method']][payloads[row['payload_key']]['status']]+=1
    panels=[];coef={};groups={}
    for arm in ARMS:
        for prefix in PREFIXES:
            for task in ('T0','T1G','T2G','T1L','T2L'):
                m,co,g=r.panel(records,scores,'RETRO146',arm,0,'bank_R0',task,tasks[:prefix],prefix)
                m.pop('route_ON',None);m.update(ar.bootstrap(co,scores,g));panels.append(m);coef[arm,prefix,task]=co;groups.update(g)
    stratified=[]
    for label,selected in [('E_eligible',[t for t in tasks if t['order'] in ELIGIBLE]),('H_only',[t for t in tasks if t['order'] not in ELIGIBLE])]:
        for arm in ARMS:
            for task in ('T0','T1G','T2G'):
                m,co,g=r.panel(records,scores,'RETRO146',arm,0,'bank_R0',task,selected,146)
                m.pop('route_ON',None);stratified.append(dict(stratum=label,**m))
    contrasts=[]
    for prefix in PREFIXES:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            he,hm=coef['HE',prefix,task],coef['HM',prefix,task];assert set(he)==set(hm)
            delta={e:r.combine([he[e],{k:-v for k,v in hm[e].items()}]) for e in he}
            limits=r.score_bounds(r.combine([{k:v/len(delta) for k,v in x.items()} for x in delta.values()]),scores) if delta else [None,None]
            contrasts.append(dict(prefix=prefix,task=task,delta_bounds_pp=[None if v is None else 100*v for v in limits],
                negative_edit_bound_count=sum(r.score_bounds(x,scores)[0]<0 for x in delta.values()),**ar.bootstrap(delta,scores,groups)))
    endpoints=[];protection=[];source=[];gates={}
    for arm in ARMS:
        unknown=coverage[arm].get('MISSING',0)
        assert sum(coverage[arm].values())==4123
        for task in ('T0','T2G'):
            pairs=[]
            for t in tasks:
                ids=[t['edit_id']] if task=='T0' else list(dict.fromkeys(q for e in t['events'] if e['task']=='T2G' for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]))
                for q in ids:pairs.append((lookup[arm,'insertion_R0',t['order'],t['order'],q],lookup[arm,'bank_R0',146,0,q]))
            endpoints.append(dict(arm=arm,task=task,**transitions(pairs,scores)))
        for prefix in PREFIXES:
            ids=protection_ids(tasks,ledger);keys=[lookup[arm,'protect146_R0',prefix,0,q] for q in ids]
            good=sum(scores[k]==1 for k in keys);missing=sum(scores[k] is None for k in keys)
            protection.append(dict(arm=arm,prefix=prefix,count=54,damage_bounds=[54-good-missing,54-good]))
        own=[]
        for owner in range(1,147):
            s=c.read(R/'public'/('SOURCE_'+arm+'_'+str(owner)+'.json'));detail=s['evidence_final']
            gate=source_gate(s['final']['weighted_all_FIT_CE'],[x['margin'] for x in detail['values']] if detail else [],owner in ELIGIBLE)
            assert gate==s['gate']
            own.append(dict(arm=arm,owner=owner,CE=s['final']['weighted_all_FIT_CE'],gate=gate,
                initial=s.get('initial'),final=s['final'],evidence_initial=s['evidence_initial'],evidence_final=detail,accepted=s['accepted'],slots=s['slots']))
        source.extend(own)
        t0=next(x for x in panels if x['arm']==arm and x['prefix']==146 and x['task']=='T0')
        t1=next(x for x in panels if x['arm']==arm and x['prefix']==146 and x['task']=='T1G')
        insertion=all(scores[lookup[arm,'insertion_R0',t['order'],t['order'],t['edit_id']]]==1 for t in tasks)
        gates[arm]=dict(full_known_scoring=unknown==0,all_source_CE=all(x['gate']['CE_pass'] for x in own),
            all_eligible_margins=all(x['gate']['evidence_pass'] for x in own if x['owner'] in ELIGIBLE),
            T0_insertion_and_final=insertion and t0['known_correct']==146,
            T1G_all=t1['missing']==0 and t1['known_correct']==t1['observations'],
            zero_individual_damage=all(x['damage_bounds'][1]==0 for x in protection if x['arm']==arm))
    protection_pairs=[]
    for prefix in PREFIXES:
        pairs=[(lookup['HM','protect146_R0',prefix,0,q],lookup['HE','protect146_R0',prefix,0,q]) for q in protection_ids(tasks,ledger)]
        protection_pairs.append(dict(prefix=prefix,**transitions(pairs,scores)))
    final=[x for x in contrasts if x['prefix']==146 and x['task'] in ('T0','T1G','T2G','T2L')]
    comparison=all(x['delta_bounds_pp'][0] is not None and x['delta_bounds_pp'][0]>=0 and x['negative_edit_bound_count']==0 for x in final)
    success=all(gates['HE'].values()) and gates['HM']['full_known_scoring'] and comparison
    result=dict(status='COMPLETE' if all(g['full_known_scoring'] for g in gates.values()) else 'COMPLETE_WITH_UNKNOWN',scientific_success=success,
        gates=gates,paired_no_degradation=comparison,coverage=coverage,panels=panels,stratified_secondary=stratified,contrasts=contrasts,endpoints=endpoints,
        fixed_protection=protection,protection_transitions_HM_to_HE=protection_pairs,source=source,
        new_Judge=cost['Judge_attempts'],GPU_process_hours=cost['gpu_seconds_used']/3600,
        scope='146-edit single-order DEV, evidence active on9;125 source groups; patient independence unknown',automatic_next=False)
    c.write(R/'public/RESULTS.json',result)
    (R/'public/REPORT_ZH.md').write_text('# HE–HM 146连续编辑对照\n\n工程状态：'+result['status']+'；科学门：'+str(success)+'。\n\n完整前缀、配对缺失界、逐源失败、证据两侧与逐题损伤见RESULTS.json。所有146条均保留，E仅覆盖预注册9条；不能宣称146条均有证据验证。单顺序开发结果，不是独立临床确认。\n')
    c.write(R/'private/REPORT_COMPLETE.json',dict(status=result['status'],epoch=time.time()))

if __name__=='__main__':main()
