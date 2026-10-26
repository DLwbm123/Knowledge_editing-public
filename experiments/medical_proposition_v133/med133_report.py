"""Original support masks, common CORE/FULL epoch, paired changes and fresh audits."""
import collections
import json
import random
import statistics

FIELDS=('core_correct','full_supported')
ROLES=('T0','FIT','T1G','T2G','T1L','T2L')


def summary(byowner):
    values=[v for rows in byowner.values() for v in rows]
    usable={o:rows for o,rows in byowner.items() if rows}
    complete=bool(values) and all(v is not None for v in values)
    return dict(n=len(values),support_edits=len(usable),known_correct=sum(v is True for v in values),
        unknown=sum(v is None for v in values),
        macro_percent=100*statistics.mean([statistics.mean(rows) for rows in usable.values()]) if complete else None,
        macro_bounds=[100*statistics.mean([sum(v is True for v in rows)/len(rows) for rows in usable.values()]),
                      100*statistics.mean([sum(v is not False for v in rows)/len(rows) for rows in usable.values()])] if values else None)


def interval(delta,groups=None):
    if not delta or any(v is None for v in delta):return None
    if len(set(delta))==1:return [100*delta[0],100*delta[0]]
    groups=groups or list(range(len(delta)));cluster=collections.defaultdict(list)
    for d,g in zip(delta,groups):cluster[g].append(d)
    rng=random.Random(20261026);names=list(cluster);samples=[]
    for _ in range(10000):
        values=[v for g in rng.choices(names,k=len(names)) for v in cluster[g]]
        samples.append(100*statistics.mean(values))
    samples.sort();return [samples[249],samples[9749]]


def summarize(packet,consumers,decisions,missing,stage,cost):
    tasks=packet['tasks'];mask=packet['base_mask']
    records={(v['arm'],v['mode'],v['prefix'],v['owner'],v['query_id']):v for v in consumers}
    assert len(records)==len(consumers)
    def score(k,field):
        v=records[k];d=decisions.get(v['payload_key']);return d[field] if d else None
    def roleids(t,role):
        if role=='FIT':return [f"edit_{t['order']}_{i}" for i in range(1,5)]
        ids=list(dict.fromkeys(q for e in t['events'] if e['task']==role for q in e['all_probe_query_ids']))
        if role!='T0':ids=[q for q in ids if mask[q]==role.endswith('L')]
        return ids
    def keys(arm,mode,n,t,role):
        out=[(arm,mode,n,t['order'] if mode=='single' else 0,q) for q in roleids(t,role)]
        return [k for k in out if not (mode=='bank' and role.endswith('L') and records[k]['active_target'])]
    phases=[('single',1)] if stage=='A' else [('single',1)]+[('bank',n) for n in (1,50,100,146)]
    panels={};owner_panels={};paired={};protection={};quality={}
    def audit(pairs,field):
        counts=collections.Counter()
        for b,l,m in pairs:
            if b is None or l is None or m is None:counts['unknown']+=1;continue
            counts['known']+=1
            if b:
                counts['Base_correct']+=1
                if not m:counts['total_Base_damage']+=1
                if l:
                    counts['Base_and_legacy_correct']+=1
                    if not m:counts['newly_damaged']+=1
                elif m:counts['old_damage_repaired']+=1
        den=counts['Base_and_legacy_correct']
        return dict(counts,qualified_support=den,zero_new_damage=(counts['newly_damaged']==0) if den and not counts['unknown'] else None,
            reference_scope='Current common epoch; FULL conservative source entailment, not clinical truth')
    for mode,n in phases:
        label=f'{mode}_{n}';ts=tasks if mode=='single' else tasks[:n]
        panels[label]={};owner_panels[label]={}
        for arm in ('MED','LEGACY','BASE'):
            panels[label][arm]={};owner_panels[label][arm]={}
            for role in ROLES:
                panels[label][arm][role]={};owner_panels[label][arm][role]={}
                for field in FIELDS:
                    values={t['order']:[score(k,field) for k in keys(arm,mode,n,t,role)] for t in ts}
                    owner_panels[label][arm][role][field]=values
                    panels[label][arm][role][field]=summary(values)
        paired[label]={}
        for field in FIELDS:
            med=owner_panels[label]['MED']['T2G'][field];legacy=owner_panels[label]['LEGACY']['T2G'][field]
            delta=[];groups=[]
            for t in ts:
                a,b=med[t['order']],legacy[t['order']];assert len(a)==len(b)
                if a:
                    delta.append(None if any(v is None for v in a+b) else statistics.mean(a)-statistics.mean(b))
                    groups.append(t['native']['source_group'])
            paired[label][field]=dict(delta_macro_pp=100*statistics.mean(delta) if delta and all(v is not None for v in delta) else None,
                up=sum(v is not None and v>0 for v in delta),down=sum(v is not None and v<0 for v in delta),
                same=sum(v==0 for v in delta),unknown=sum(v is None for v in delta),
                edit_bootstrap95_pp=interval(delta),source_cluster_bootstrap95_pp=interval(delta,groups))
        protection[label]={}
        for registry in ('all_nonnative_evaluation','locality_all_probes'):
            protection[label][registry]={}
            for field in FIELDS:
                pairs=[]
                for t in ts:
                    qids={q for e in t['events'] if registry=='all_nonnative_evaluation' or e['task'] in ('T1L','T2L') for q in e['all_probe_query_ids']}
                    active={(v['native']['image_sha256'],v['native']['question']) for v in (tasks[:n] if mode=='bank' else [t])}
                    for qid in qids:
                        ks=[(arm,mode,n,t['order'] if mode=='single' else 0,qid) for arm in ('BASE','LEGACY','MED')]
                        v=records[ks[-1]]
                        if (v['image_sha256'],v['question']) in active:continue
                        pairs.append(tuple(score(k,field) for k in ks))
                protection[label][registry][field]=audit(pairs,field)
        quality[label]={arm:dict(n=sum(v['arm']==arm and v['mode']==mode and v['prefix']==n for v in consumers),
            at_cap=sum(v['arm']==arm and v['mode']==mode and v['prefix']==n and v['at_cap'] for v in consumers),
            EOS=sum(v['arm']==arm and v['mode']==mode and v['prefix']==n and v['EOS'] for v in consumers)) for arm in ('MED','LEGACY','BASE')}
    transition=[];insertion={}
    if stage=='B':
        for left,right in ((1,50),(50,100),(100,146)):
            for field in FIELDS:
                rows=[]
                for t in tasks[:left]:
                    for role in ('FIT','T2G'):
                        for q in roleids(t,role):
                            a=score(('MED','bank',left,0,q),field);b=score(('MED','bank',right,0,q),field)
                            rows.append((a,b))
                transition.append(dict(from_prefix=left,to_prefix=right,field=field,same_old_owners=left,n=len(rows),
                    forgotten=sum(a is True and b is False for a,b in rows),repaired=sum(a is False and b is True for a,b in rows),
                    unknown=sum(a is None or b is None for a,b in rows)))
        for field in FIELDS:
            rows=[]
            for t in tasks:
                inserted=[v for v in consumers if v['arm']=='MED' and v['mode']=='insertion' and v['owner']==t['order']]
                assert len(inserted)==1;v=inserted[0]
                final=[x for x in consumers if x['arm']=='MED' and x['mode']=='bank' and x['prefix']==146 and x['image_sha256']==v['image_sha256'] and x['question']==v['question']]
                assert final
                a=decisions.get(v['payload_key'],{}).get(field);b=decisions.get(final[0]['payload_key'],{}).get(field)
                rows.append((a,b))
            insertion[field]=dict(n=146,inserted_correct=sum(a is True for a,b in rows),final_correct=sum(b is True for a,b in rows),
                forgotten=sum(a is True and b is False for a,b in rows),unknown=sum(a is None or b is None for a,b in rows))
    complete=not any(v['payload_key'] not in decisions for v in consumers)
    gates={}
    for label in panels:
        a=panels[label]['MED']['T2G']['core_correct']['macro_percent'];b=panels[label]['LEGACY']['T2G']['core_correct']['macro_percent']
        fit=[(a,b) for t in (tasks if label=='single_1' else tasks[:int(label.split('_')[1])]) for role in ('T0','FIT')
             for a,b in zip(owner_panels[label]['MED'][role]['core_correct'][t['order']],owner_panels[label]['LEGACY'][role]['core_correct'][t['order']])]
        full_fit=[(a,b) for t in (tasks if label=='single_1' else tasks[:int(label.split('_')[1])]) for role in ('T0','FIT')
             for a,b in zip(owner_panels[label]['MED'][role]['full_supported'][t['order']],owner_panels[label]['LEGACY'][role]['full_supported'][t['order']])]
        medfull=panels[label]['MED']['T2G']['full_supported']['macro_percent'];legacyfull=panels[label]['LEGACY']['T2G']['full_supported']['macro_percent']
        gates[label]=dict(T2G_plus3pp=a-b>=3 if a is not None and b is not None else None,
            source_CORE_no_legacy_regression=all(a is not None and b is not None and (not b or a) for a,b in fit),
            source_FULL_no_legacy_regression=all(a is not None and b is not None and (not b or a) for a,b in full_fit),
            T2G_FULL_no_legacy_regression=medfull>=legacyfull if medfull is not None and legacyfull is not None else None,
            CORE_no_new_damage=protection[label]['all_nonnative_evaluation']['core_correct']['zero_new_damage'],
            FULL_no_new_damage=protection[label]['all_nonnative_evaluation']['full_supported']['zero_new_damage'])
    result=dict(status='COMPLETE' if complete else 'INCOMPLETE_SCORING',stage=stage,
        original146_exposed_DEV=True,independent_confirmation=False,medical_unique_gain_identifiable=False,
        no_146_GEN_or_FULL_CTX_control=True,panels=panels,paired_T2G=paired,protection=protection,
        quality=quality,route_audit=dict(
            MED_consumers=sum(v['arm']=='MED' for v in consumers),
            Base_fallbacks=sum(v['arm']=='MED' and v.get('effective_expert') is None for v in consumers),
            strict_answer_reuse_consumers=sum(v['arm']=='MED' and v.get('prediction_reused',False) for v in consumers),
            all_membership_and_state_input_checks_enforced=True),cross_prefix_common_owners=transition,insertion_final_retention=insertion,gates=gates,
        grading=dict(protocol='MEDPROP_SMALL4_CORE_FULL_SOURCE_V1',model='gpt-6-astra',effort='medium',consumers=len(consumers),
                     unique=len({v['payload_key'] for v in consumers}),unknown_consumers=sum(v['payload_key'] not in decisions for v in consumers)),
        cost=cost,scientific_negative_does_not_cancel_sequential=True)
    lines=['# MED 原146 '+('单次编辑' if stage=='A' else '完整顺序编辑')+'报告','',
        '状态：'+result['status']+'。原队列、资格掩码与R0固定；本轮为已暴露开发集。',
        'MED 与历史 M13P 回答在同一 CORE/FULL 来源支持协议下评分；历史 accepted 标签不改写。',
        '没有全146 GEN/FULL_CTX对照，不能把差异归因为医学命题特异增益。','',
        '|阶段|方法|T2G CORE宏平均|T2G FULL宏平均|T2L CORE支持|', '|---|---|---:|---:|---:|']
    for label,v in panels.items():
        for arm in ('MED','LEGACY','BASE'):
            a=v[arm]['T2G']['core_correct']['macro_percent'];b=v[arm]['T2G']['full_supported']['macro_percent'];c=v[arm]['T2L']['core_correct']
            lines.append(f"|{label}|{arm}|{a if a is not None else 'NA'}|{b if b is not None else 'NA'}|{c['known_correct']}/{c['n']}|")
    lines+=['','百分比为编辑宏平均；空支持为NA，UNKNOWN与错误分列。逐编辑配对区间、原资格掩码、fresh CORE/FULL损伤与旧伤修复、生成完整性和费用见JSON。',
        '顺序阶段真实计算未编辑Base的R0选择；相同专家状态、完整输入及确定性解码身份允许复用同一回答，每个消费者均保留独立实际路由绑定。',
        '所有146个MED终态专家与复现配置保留。科学负结果不取消已登记顺序阶段。','']
    return result,'\n'.join(lines),owner_panels
