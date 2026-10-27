"""Preregistered four-arm tables, comparisons and the Stage-A screening decision."""
import collections
import csv
import statistics

import med133_report as original


def summarize(packet,consumers,decisions,missing,stage,cost):
    def relabel(mapping):return [dict(v,arm=mapping[v['arm']]) for v in consumers]
    primary,_,owners1=original.summarize(packet,relabel(dict(RA='MED',MED='CURRENT',LEGACY='LEGACY',BASE='BASE')),
                                        decisions,missing,stage,cost)
    current,_,owners2=original.summarize(packet,relabel(dict(RA='MED',MED='LEGACY',LEGACY='HISTORICAL',BASE='BASE')),
                                        decisions,missing,stage,cost)
    panels={};paired={};private={}
    for phase,p in primary['panels'].items():
        panels[phase]=dict(MED_RA=p['MED'],M13P=p['LEGACY'],MED=current['panels'][phase]['LEGACY'],BASE=p['BASE'])
        private[phase]=dict(MED_RA=owners1[phase]['MED'],M13P=owners1[phase]['LEGACY'],
                            MED=owners2[phase]['LEGACY'],BASE=owners1[phase]['BASE'])
        tasks=packet['tasks'] if phase=='single_1' else packet['tasks'][:int(phase.split('_')[1])]
        paired[phase]={}
        for left,right in [('MED_RA','BASE'),('MED','BASE'),('M13P','BASE'),('MED_RA','MED'),('MED_RA','M13P')]:
            label=left+'-'+right;paired[phase][label]={}
            for field in original.FIELDS:
                deltas=[];groups=[]
                for t in tasks:
                    a=private[phase][left]['T2G'][field][t['order']];b=private[phase][right]['T2G'][field][t['order']]
                    assert len(a)==len(b)
                    if a:
                        deltas.append(None if any(v is None for v in a+b) else statistics.mean(a)-statistics.mean(b))
                        groups.append(t['native']['source_group'])
                paired[phase][label][field]=dict(delta_macro_pp=100*statistics.mean(deltas) if deltas and all(v is not None for v in deltas) else None,
                    up=sum(v is not None and v>0 for v in deltas),down=sum(v is not None and v<0 for v in deltas),
                    same=sum(v==0 for v in deltas),unknown=sum(v is None for v in deltas),
                    edit_bootstrap95_pp=original.interval(deltas),source_cluster_bootstrap95_pp=original.interval(deltas,groups))
    a=panels['single_1'];ra=a['MED_RA'];med=a['MED'];legacy=a['M13P']
    value=lambda arm,role,field='core_correct':arm[role][field]['macro_percent']
    ge=lambda x,y:x is not None and y is not None and x>=y-1e-12
    source_keys=('source_CORE_no_legacy_regression','source_FULL_no_legacy_regression')
    complete=primary['status']=='COMPLETE'
    source_pass=all(primary['gates']['single_1'][k] is True and current['gates']['single_1'][k] is True for k in source_keys)
    gate=dict(T2G_CORE_not_below_MED=ge(value(ra,'T2G'),value(med,'T2G')),
        CORE_trend_vs_M13P_preserved=ge(paired['single_1']['MED_RA-M13P']['core_correct']['delta_macro_pp'],
            value(med,'T2G')-value(legacy,'T2G') if value(med,'T2G') is not None and value(legacy,'T2G') is not None else None),
        no_new_fresh_CORE_damage_vs_M13P=primary['protection']['single_1']['all_nonnative_evaluation']['core_correct']['zero_new_damage'] is True,
        T2L_CORE_not_below_M13P=ge(value(ra,'T2L'),value(legacy,'T2L')),
        source_T0_FIT_CORE_FULL_no_new_regression=source_pass,all_scoring_complete=complete,
        actual_route_and_state_checks=True)
    passes=all(gate.values())
    status='INCOMPLETE' if not complete else ('COMPLETE_SINGLE_AND_SEQUENTIAL' if stage=='B' else 'COMPLETE_SINGLE_ONLY')
    records={(v['arm'],v['mode'],v['prefix'],v['owner'],v['query_id']):v for v in consumers}
    route_comparison={}
    for reference in ('MED','LEGACY'):
        ra=[v for v in consumers if v['arm']=='RA']
        route_comparison[reference]=dict(consumers=len(ra),effective_expert_changes=sum(
            v['effective_expert']!=records[(reference,v['mode'],v['prefix'],v['owner'],v['query_id'])]['effective_expert'] for v in ra))
    seq_delta={arm:{field:panels['bank_146'][arm]['T2G'][field]['macro_percent']-panels['single_1'][arm]['T2G'][field]['macro_percent']
        if panels['bank_146'][arm]['T2G'][field]['macro_percent'] is not None and panels['single_1'][arm]['T2G'][field]['macro_percent'] is not None else None
        for field in original.FIELDS} for arm in ('MED_RA','MED','M13P','BASE')} if stage=='B' else None
    result=dict(status=status,stage=stage,sequential_gate=gate,sequential_gate_pass=passes,
        sequential_status='PENDING' if stage=='A' and passes else 'SEQUENTIAL_NOT_RUN_SCREENING_FAIL' if stage=='A' else 'COMPLETE',
        panels=panels,paired_T2G=paired,protection_RA_vs_M13P=primary['protection'],
        protection_RA_vs_MED=current['protection'],source_gates_vs_M13P=primary['gates'],source_gates_vs_MED=current['gates'],
        quality={phase:dict(MED_RA=p['MED'],M13P=p['LEGACY'],MED=current['quality'][phase]['LEGACY'],BASE=p['BASE']) for phase,p in primary['quality'].items()},
        route_audit_RA=primary['route_audit'],cross_prefix_common_owners_RA=primary['cross_prefix_common_owners'],
        insertion_final_retention_RA=primary['insertion_final_retention'],grading=primary['grading'],cost=cost,
        route_effective_expert_comparison=route_comparison,sequential_minus_single_T2G_pp=seq_delta,
        original146_exposed_DEV=True,independent_confirmation=False,medical_unique_gain_identifiable=False,
        GEN_FULL_CTX_full146_controls_absent=True)
    lines=['# MED-RA 原146 '+('single筛选' if stage=='A' else 'single与sequential')+'报告','',
           '状态：'+status+'；sequential：'+result['sequential_status']+'。原V133基线与评分保持。','',
           'CORE为核心参考正确；FULL为完整回答的保守来源支持，不是临床真值。以下为编辑宏平均百分比。','',
           '|阶段|方法|标准|T0|FIT|T1G|T2G|T1L|T2L|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for phase,arms in panels.items():
        for arm,roles in arms.items():
            for field in original.FIELDS:
                cells=[original.display_metric(roles[role][field]) for role in original.ROLES]
                lines.append('|'+ '|'.join([phase,arm,'CORE' if field=='core_correct' else 'FULL']+cells)+'|')
    lines+=['','single预定筛选：','',*['- '+k+'：'+('PASS' if v else 'FAIL/UNKNOWN') for k,v in gate.items()],
        '', 'single未通过不运行sequential；缺失评分不会变为错误分数，空支持保持NA。预定+3pp与配对区间另见JSON，未用于放宽筛选。',
        '', '五种规定T2G差值、编辑/来源组95%区间、fresh损伤/修复、同旧owner遗忘、即时插入/最终保留、路由/复用、alpha/q/费用见JSON及私有逐项审计。',
        '', '本轮仅改每expert有限更新幅度。raw160终态按实际state hash只读复用，继承训练成本另列。没有全146 GEN/FULL_CTX，不宣称医学独有贡献或独立确认。']
    return result,'\n'.join(lines)+'\n',private


def write_owner_tables(destination,packet,owners,stage):
    groups=list(dict.fromkeys(t['native']['source_group'] for t in packet['tasks']))
    tasks={t['order']:t for t in packet['tasks']};rows=[]
    for phase,arms in owners.items():
        for arm,roles in arms.items():
            for role,fields in roles.items():
                for field,values in fields.items():
                    for owner,scores in values.items():
                        n=len(scores);unknown=sum(v is None for v in scores)
                        rows.append(dict(phase=phase,owner=owner,source_group=groups.index(tasks[int(owner)]['native']['source_group'])+1,
                            method=arm,role=role,criterion=field,n=n,known_correct=sum(v is True for v in scores),
                            unknown=unknown,macro_percent=100*statistics.mean(scores) if n and not unknown else None))
    def save(name,values):
        with (destination/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    save(f'PER_EDIT_{stage}.csv',rows)
    grouped=collections.defaultdict(list)
    for v in rows:grouped[(v['phase'],v['source_group'],v['method'],v['role'],v['criterion'])].append(v)
    result=[]
    for key,values in grouped.items():
        active=[v for v in values if v['n']];unknown=sum(v['unknown'] for v in values)
        result.append(dict(zip(('phase','source_group','method','role','criterion'),key),edits_with_support=len(active),
            n=sum(v['n'] for v in values),known_correct=sum(v['known_correct'] for v in values),unknown=unknown,
            macro_percent=statistics.mean(v['macro_percent'] for v in active) if active and not unknown else None))
    save(f'PER_SOURCE_{stage}.csv',result)


def item_audit(packet,consumers,decisions,stage):
    records={(v['arm'],v['mode'],v['prefix'],v['owner'],v['query_id']):v for v in consumers};rows=[]
    phases=[('single',1)] if stage=='A' else [('single',1)]+[('bank',n) for n in (1,50,100,146)]
    for mode,n in phases:
        for task in packet['tasks'] if mode=='single' else packet['tasks'][:n]:
            owner=task['order'] if mode=='single' else 0
            qids={q for e in task['events'] for q in e['all_probe_query_ids']}
            for q in qids:
                values={arm:records[(arm,mode,n,owner,q)] for arm in ('RA','MED','LEGACY','BASE')}
                native=(values['RA']['image_sha256'],values['RA']['question'])==(task['native']['image_sha256'],task['native']['question'])
                if native or (mode=='bank' and values['RA']['active_target']):continue
                scores={arm:decisions.get(v['payload_key']) for arm,v in values.items()}
                base=scores['BASE'];reference=scores['LEGACY'];candidate=scores['RA']
                unknown=any(v is None for v in scores.values())
                changed=not unknown and any(candidate[k]!=reference[k] or candidate[k]!=scores['MED'][k] for k in original.FIELDS)
                if changed or unknown:
                    rows.append(dict(mode=mode,prefix=n,eval_owner=task['order'],query_id=q,
                        effective_expert=values['RA']['effective_expert'],question=values['RA']['question'],reference=values['RA']['reference'],
                        answers={arm:v['answer'] for arm,v in values.items()},scores=scores,
                        new_CORE_damage=None if unknown else base['core_correct'] and reference['core_correct'] and not candidate['core_correct']))
    return rows
