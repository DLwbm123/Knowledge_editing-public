"""Offline comparison of both frozen expert banks after authorized score recovery."""
import json
from pathlib import Path

from med133_report import FIELDS, ROLES


def compare(result,packet,consumers,decisions):
    assert result['status']=='COMPLETE' and result['grading']['unknown_consumers']==0
    records={(v['arm'],v['mode'],v['prefix'],v['owner'],v['query_id']):v for v in consumers}
    assert len(records)==len(consumers)==22482
    assert all(v['payload_key'] in decisions for v in consumers)
    tasks=packet['tasks'];mask=packet['base_mask']

    def roleids(t,role):
        if role=='FIT':return [f"edit_{t['order']}_{i}" for i in range(1,5)]
        ids=list(dict.fromkeys(q for e in t['events'] if e['task']==role for q in e['all_probe_query_ids']))
        return ids if role=='T0' else [q for q in ids if mask[q]==role.endswith('L')]

    def score(arm,prefix,q,field):
        return decisions[records[(arm,'bank',prefix,0,q)]['payload_key']][field]

    insertion={};transitions={}
    for arm in ('MED','LEGACY'):
        insertion[arm]={};transitions[arm]=[]
        final=[v for v in consumers if v['arm']==arm and v['mode']=='bank' and v['prefix']==146]
        for field in FIELDS:
            pairs=[]
            for t in tasks:
                inserted=[v for v in consumers if v['arm']==arm and v['mode']=='insertion' and v['owner']==t['order']]
                assert len(inserted)==1
                v=inserted[0]
                terminal=[x for x in final if (x['image_sha256'],x['question'])==(v['image_sha256'],v['question'])]
                assert terminal
                pairs.append((decisions[v['payload_key']][field],decisions[terminal[0]['payload_key']][field]))
            insertion[arm][field]=dict(n=len(pairs),inserted_correct=sum(a for a,b in pairs),
                final_correct=sum(b for a,b in pairs),forgotten=sum(a and not b for a,b in pairs),unknown=0)
        for left,right in ((1,50),(50,100),(100,146)):
            for field in FIELDS:
                pairs=[(score(arm,left,q,field),score(arm,right,q,field))
                       for t in tasks[:left] for role in ('FIT','T2G') for q in roleids(t,role)]
                transitions[arm].append(dict(from_prefix=left,to_prefix=right,field=field,
                    same_old_owners=left,n=len(pairs),forgotten=sum(a and not b for a,b in pairs),
                    repaired=sum(not a and b for a,b in pairs),unknown=0))
    assert insertion['MED']==result['insertion_final_retention']
    assert transitions['MED']==result['cross_prefix_common_owners']
    fresh={}
    for phase,registries in result['protection'].items():
        fresh[phase]={}
        for registry,fields in registries.items():
            fresh[phase][registry]={}
            for field,v in fields.items():
                assert not v.get('unknown',0)
                fresh[phase][registry][field]=dict(event_observations=v['known'],Base_correct=v.get('Base_correct',0),
                    M13P_damage_vs_Base=v.get('Base_correct',0)-v['qualified_support'],
                    MED_damage_vs_Base=v.get('total_Base_damage',0),new_MED_damage_vs_M13P=v.get('newly_damaged',0),
                    MED_repairs_old_M13P_damage=v.get('old_damage_repaired',0),
                    shared_qualified_support=v['qualified_support'],zero_new_damage=v['zero_new_damage'])
    return dict(status='COMPLETE',grading=result['grading'],panels=result['panels'],paired_T2G=result['paired_T2G'],
        fresh=fresh,insertion_final_retention=insertion,cross_prefix_common_owners=transitions,
        gates=result['gates'],quality=result['quality'],route_audit=result['route_audit'],cost=result['cost'],
        independent_confirmation=False,medical_unique_gain_identifiable=False)


def markdown(value):
    lines=['# MED 与 M13P-160 完整 single / sequential 对比（补评后）','',
        '**评分状态：COMPLETE。** 原80个失败载荷已全部补齐。两个方法各7494个消费者、连同Base共22482个消费者均有有效评分，UNKNOWN/MISSING为0。只新增80次评分尝试，没有训练、编辑、回答生成或GPU重建。原失败、原有效标签和补齐前报告保持不变。','',
        '统一使用原`MEDPROP_SMALL4_CORE_FULL_SOURCE_V1`、gpt-6-astra medium。CORE判断参考核心意思是否正确；FULL还要求完整回答没有参考之外的实质扩展、额外矛盾、截断或循环。FULL是保守来源支持，不是临床真值。历史二元评分没有混入本表。','',
        'single为146次独立单专家R0评测；sequential为同一冻结专家库按原顺序逐步插入、实际R0路由后的prefix1/50/100/146评测。MED训练23360步在所有生成前完成；B阶段新增训练0。M13P使用先前冻结的原146/160步回答。本轮不因其原权重缺失而重建。','',
        '## Single 与最终全库','',
        '百分比为编辑宏平均。括号为微正确数/探针数；两种平均不可互换。T1L仅1个编辑/2探针，T2L仅31编辑/54探针。','',
        '|阶段|方法|判定|T0|FIT|T1G|T2G|T1L|T2L|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for phase in ('single_1','bank_146'):
        for arm in ('MED','LEGACY'):
            for field in FIELDS:
                cells=[]
                for role in ROLES:
                    v=value['panels'][phase][arm][role][field]
                    cells.append(f"{v['macro_percent']:.4f} ({v['known_correct']}/{v['n']})")
                lines.append('|'+ '|'.join([phase,'MED' if arm=='MED' else 'M13P-160','CORE' if field=='core_correct' else 'FULL']+cells)+'|')
    lines+=['','## 所有顺序前缀','',
        '|前缀|T2G探针数|MED CORE|M13P CORE|差值/pp|MED FULL|M13P FULL|差值/pp|','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in (1,50,100,146):
        phase=f'bank_{n}';a=value['panels'][phase]['MED']['T2G'];b=value['panels'][phase]['LEGACY']['T2G']
        core=value['paired_T2G'][phase]['core_correct']['delta_macro_pp'];full=value['paired_T2G'][phase]['full_supported']['delta_macro_pp']
        lines.append(f"|{n}|{a['core_correct']['n']}|{a['core_correct']['macro_percent']:.4f}|{b['core_correct']['macro_percent']:.4f}|{core:+.4f}|{a['full_supported']['macro_percent']:.4f}|{b['full_supported']['macro_percent']:.4f}|{full:+.4f}|")
    lines+=['','## 配对区间与预定门槛','',
        '|阶段|判定|MED−M13P/pp|编辑bootstrap 95%|来源组bootstrap 95%|提高/相同/降低|','|---|---|---:|---|---|---|']
    for phase,fields in value['paired_T2G'].items():
        for field,v in fields.items():
            lines.append(f"|{phase}|{'CORE' if field=='core_correct' else 'FULL'}|{v['delta_macro_pp']:+.4f}|{v['edit_bootstrap95_pp']}|{v['source_cluster_bootstrap95_pp']}|{v['up']}/{v['same']}/{v['down']}|")
    lines+=['',
        'single和最终全库CORE均仅提高0.8562pp，FULL均提高1.1986pp；两类配对95%区间均跨0。未达到预登记CORE +3pp。来源T0/FIT的CORE/FULL无退化条件通过，T2G FULL均值未退化；CORE零新增损伤条件未通过，FULL零新增损伤仅在其非空来源支持上通过。不能宣称MED已经全面优于M13P。','',
        '## Fresh保护审计','',
        '以下单位为冻结评测消费者事件，保留每任务的既定重复关系，不是独立患者数或独立医学确认。原Base资格掩码的T1L/T2L与当前fresh支持分列。','',
        '|阶段|判定|已审计事件|当前Base正确|M13P损伤|MED损伤|MED新增损伤/共同合格支持|MED修复旧伤|','|---|---|---:|---:|---:|---:|---:|---:|']
    for phase,registries in value['fresh'].items():
        for field,v in registries['all_nonnative_evaluation'].items():
            lines.append(f"|{phase}|{'CORE' if field=='core_correct' else 'FULL'}|{v['event_observations']}|{v['Base_correct']}|{v['M13P_damage_vs_Base']}|{v['MED_damage_vs_Base']}|{v['new_MED_damage_vs_M13P']}/{v['shared_qualified_support']}|{v['MED_repairs_old_M13P_damage']}|")
    lines+=['',
        'single和bank146的fresh CORE支持均为Base正确85项：M13P损伤2项，MED损伤3项，MED新增损伤1/83、旧伤修复0。fresh FULL中Base正确13项，M13P与MED均损伤1项，MED新增损伤0/12。仅locality事件的FULL共同合格支持为0，必须记NA，不能作为零伤证据。原T2L FULL两方法均为0/54，表示整段回答不满足严格参考完整支持，不能把它解释成核心回答全错。','',
        '## 插入保留与相同旧owner的遗忘','',
        '|方法|判定|即时native正确|最终native正确|遗忘|UNKNOWN|','|---|---|---:|---:|---:|---:|']
    for arm,fields in value['insertion_final_retention'].items():
        for field,v in fields.items():
            lines.append(f"|{'MED' if arm=='MED' else 'M13P-160'}|{'CORE' if field=='core_correct' else 'FULL'}|{v['inserted_correct']}/{v['n']}|{v['final_correct']}/{v['n']}|{v['forgotten']}|0|")
    lines+=['','后表只比较相同旧owner的同一FIT/T2G集合，避免将不同前缀均值差误称遗忘。','',
        '|方法|前缀变化|判定|旧owner数|配对数|遗忘|修复|','|---|---|---|---:|---:|---:|---:|']
    for arm,rows in value['cross_prefix_common_owners'].items():
        for v in rows:
            lines.append(f"|{'MED' if arm=='MED' else 'M13P-160'}|{v['from_prefix']}→{v['to_prefix']}|{'CORE' if v['field']=='core_correct' else 'FULL'}|{v['same_old_owners']}|{v['n']}|{v['forgotten']}|{v['repaired']}|")
    lines+=['','## 执行、费用与证据边界','',
        'MED共7494个生成消费者，实际生成3226次、严格身份复用4268次，真实Base R0特征2093项、回落Base22次。single的MED为2955项EOS/1项触顶，最终bank146为2091项EOS/2项触顶；M13P分别为2956/0与2092/1。生成完整性和原运行开销保持不变。','',
        '原评分尝试4614项（含原80项失败），本次追加80项，累计4694项；最终4754个唯一载荷全部有效，其中140个继承原小实验相同协议有效评分。两次尝试均计费，不清除原失败，不重复评分此前有效载荷。所有146个MED FINAL、canonical与独立小备份保留；原M13P权重缺失事实不变。','',
        '本轮仍是原146已暴露开发集，缺少GEN/FULL_CTX全146对照；完整评分不构成独立确认，也不能单独证明医学命题特异增益。没有开启新训练或后续研究；不重新创建监测。','',
        '完整数字：[COMPARISON_RECOVERY1.json](COMPARISON_RECOVERY1.json)。原面板：[single](RESULTS_A_RECOVERY1.json)、[sequential](RESULTS_B_RECOVERY1.json)。补评准入与账本：[SCORING_RECOVERY1_AUDIT.json](SCORING_RECOVERY1_AUDIT.json)。']
    return '\n'.join(lines)+'\n'


def write_comparison(result,packet,consumers,decisions,destination):
    value=compare(result,packet,consumers,decisions);destination=Path(destination)
    (destination/'COMPARISON_RECOVERY1.json').write_text(json.dumps(value,indent=2)+'\n')
    (destination/'COMPARISON_RECOVERY1_ZH.md').write_text(markdown(value))
    return value
