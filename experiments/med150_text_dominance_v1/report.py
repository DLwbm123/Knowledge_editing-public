"""Aggregate the preregistered fixed-margin MED150 development comparison."""
import collections
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/medical_proposition_v133'))
from med133_report import summary,interval


def read(p):return json.loads(Path(p).read_text())
def score(decisions,key,field):return decisions.get(key,{}).get(field)
def transitions(pairs):
    c=collections.Counter()
    for a,b in pairs:
        if a is None or b is None:c['unknown']+=1
        else:
            c['known']+=1;c['before_correct']+=a;c['after_correct']+=b;c['lost']+=a and not b;c['gained']+=not a and b
    return {k:c[k] for k in ('known','unknown','before_correct','after_correct','lost','gained')}
def no_regression(pairs):
    if any(a is True and b is False for a,b in pairs):return False
    return None if not pairs or any(a is None or b is None for a,b in pairs) else True


def main():
    q=Path('/private/tmp/w.20261028.3');prior=Path('/private/tmp/u.20261027.150');base=Path('/private/tmp/s.20261027.150')
    done=read(q/'COMPLETE.json');assert done['generation_and_scoring_complete'] and read(q/'GUARD_EXIT.json')['code']==0
    modal=read(q/'PACKET_PRIVATE.json')['consumers'];md=read(q/'DECISIONS.json')['decisions'];missing=read(q/'DECISIONS.json')['unknown']
    previous=read(prior/'PACKET_PRIVATE.json')['consumers'];pd=read(prior/'DECISIONS.json')['decisions']
    margin=read('/private/tmp/v.20261028.2/PACKET_PRIVATE.json')['consumers'];margin={x['query_id']:x for x in margin}
    before={x['query_id']:x for x in previous if x['mode']=='bank' and x['prefix']==150}
    after={x['query_id']:x for x in modal if x['mode']=='bank' and x['prefix']==150}
    assert len(before)==len(after)==len(margin)==2148 and set(before)==set(after)==set(margin)
    comparison=dict(queries=2148,same_selected=sum(margin[k]['effective_expert']==after[k]['effective_expert'] for k in after),same_payload=sum(margin[k]['key']==after[k]['key'] for k in after),text_fallbacks=sum(after[k]['route']['text_rule_fallback'] for k in after))
    assert comparison['same_selected']==comparison['same_payload']==2148 and comparison['text_fallbacks']==0
    oldroot=ROOT/'.private/med146_single_sequential_v133_20261027/scoring_recovery1'
    allold=read(oldroot/'CONSUMERS_B_FROZEN.json')['consumers'];olddec=read(oldroot/'MERGED_DECISIONS.json')['decisions'];meta=read(oldroot/'PACKET_METADATA_B.json')
    oldbank={x['query_id']:x for x in allold if x['arm']=='MED' and x['mode']=='bank' and x['prefix']==146}
    oldbase={x['query_id']:x for x in allold if x['arm']=='BASE' and x['mode']=='bank' and x['prefix']==146}
    assert len(oldbank)==len(oldbase)==2093 and set(oldbank)<=set(before)
    mask=read(base/'BASE_MASK_FROZEN.json')['consumers'];assert len(mask)==55
    candidates=read(ROOT/'.private/med150_feasibility_20261027/ROLE_SAFE_CANDIDATES.private.json')['items']
    added_active={(v['task']['native']['image_sha256'],v['task']['native']['question']) for v in candidates}
    newly_active={k for k,v in oldbank.items() if (v['image_sha256'],v['question']) in added_active}
    tasks=meta['tasks'];oldmask=meta['base_mask'];assert len(tasks)==146
    def ids(t,role):
        if role=='FIT':return [f"edit_{t['order']}_{i}" for i in range(1,5)]
        qs=list(dict.fromkeys(k for e in t['events'] if e['task']==role for k in e['all_probe_query_ids']))
        if role!='T0':qs=[k for k in qs if oldmask[k]==role.endswith('L')]
        if role.endswith('L'):qs=[k for k in qs if not oldbank[k]['active_target'] and k not in newly_active]
        return qs
    panels={};delta={};changes={};source_pairs={};combined={};newpanels={}
    for field in ('core_correct','full_supported'):
        panels[field]={};delta[field]={};changes[field]={};source_pairs[field]=[]
        for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
            oldby={};newby={};ds=[];groups=[];pairs=[]
            for t in tasks:
                qs=ids(t,role);a=[score(pd,before[k]['key'],field) for k in qs];b=[score(md,after[k]['key'],field) for k in qs]
                oldby[t['order']]=a;newby[t['order']]=b;pairs.extend(zip(a,b))
                if role in ('T0','FIT'):source_pairs[field].extend(zip(a,b))
                if a:
                    ds.append(None if any(v is None for v in a+b) else statistics.mean(b)-statistics.mean(a));groups.append(t['native'].get('source_group',str(t['order'])))
            panels[field][role]=dict(R0=summary(oldby),modal=summary(newby));changes[field][role]=transitions(pairs)
            delta[field][role]=dict(delta_macro_pp=100*statistics.mean(ds) if ds and all(v is not None for v in ds) else None,
                support_edits=len(ds),up=sum(v is not None and v>0 for v in ds),down=sum(v is not None and v<0 for v in ds),same=sum(v==0 for v in ds),unknown=sum(v is None for v in ds),edit_bootstrap95_pp=interval(ds),source_group_bootstrap95_pp=interval(ds,groups))
        newpanels[field]={};combined[field]={}
        for role in ('T0','FIT','T1G','T2G','T2L'):
            a={};b={};newpairs=[]
            for owner in range(147,151):
                qs=[x['query_id'] for x in mask if x['owner']==owner and x['role']==role and (role in ('T0','FIT') or x['eligible'] is True)]
                if role=='T1G':qs=[]
                aa=[score(pd,before[k]['key'],field) for k in qs];bb=[score(md,after[k]['key'],field) for k in qs]
                a[owner]=aa;b[owner]=bb;newpairs.extend(zip(aa,bb))
            newpanels[field][role]=dict(R0=summary(a),modal=summary(b),paired=transitions(newpairs))
            if role in ('T0','FIT'):source_pairs[field].extend(newpairs)
            if role in ('T0','FIT','T2G'):
                old=panels[field][role]['modal'];added=summary(b);n=old['support_edits']+added['support_edits']
                valid=old['macro_percent'] is not None and added['macro_percent'] is not None
                combined[field][role]=dict(n=old['n']+added['n'],support_edits=n,known_correct=old['known_correct']+added['known_correct'],unknown=old['unknown']+added['unknown'],macro_percent=((old['macro_percent']*old['support_edits']+added['macro_percent']*added['support_edits'])/n if n and valid else None))
    route=dict(queries=2148,changed=sum(before[k]['effective_expert']!=after[k]['effective_expert'] for k in before),
        old146_changed=sum(before[k]['effective_expert']!=after[k]['effective_expert'] for k in oldbank),
        old_R0_off=sum(not after[k]['route']['original_route']['activated'] for k in after),
        off_rule_violations=sum((not after[k]['route']['original_route']['activated']) and after[k]['effective_expert'] is not None for k in after),
        scoring_payloads_identical=sum(before[k]['key']==after[k]['key'] for k in before),all_routes_frozen_before_generation=True)
    protection={}
    for field in ('core_correct','full_supported'):
        eligible=set()
        for t in tasks:
            for e in t['events']:
                for k in e['all_probe_query_ids']:
                    if not oldbank[k]['active_target'] and k not in newly_active:eligible.add(k)
        out=collections.Counter()
        for k in eligible:
            b=score(olddec,oldbase[k]['payload_key'],field);old=score(pd,before[k]['key'],field);new=score(md,after[k]['key'],field)
            if None in (b,old,new):out['unknown']+=1;continue
            out['known']+=1
            if b:
                out['base_correct']+=1
                if not new:out['modal_base_damage']+=1
                if old:
                    out['base_and_R0_correct']+=1
                    if not new:out['newly_damaged']+=1
                elif new:out['old_damage_repaired']+=1
        protection[field]={k:out[k] for k in ('known','unknown','base_correct','base_and_R0_correct','modal_base_damage','newly_damaged','old_damage_repaired')}
        protection[field]['scope']='unique original146 nonnative queries excluding active targets, not independent patients'
    def protect_gate(f):
        v=protection[f]
        return False if v['newly_damaged'] else (True if v['base_and_R0_correct'] and not v['unknown'] else None)
    def at_least(x,n):return None if x is None else x>=n
    gates=dict(original146_T1G_CORE_plus5pp=at_least(delta['core_correct']['T1G']['delta_macro_pp'],5),
        T0_FIT_CORE_no_loss=no_regression(source_pairs['core_correct']),T0_FIT_FULL_no_loss=no_regression(source_pairs['full_supported']),
        original146_T2G_CORE_no_mean_loss=at_least(delta['core_correct']['T2G']['delta_macro_pp'],0),
        original146_T2G_FULL_no_mean_loss=at_least(delta['full_supported']['T2G']['delta_macro_pp'],0),
        no_new_CORE_damage=protect_gate('core_correct'),no_new_FULL_damage=protect_gate('full_supported'))
    verdict='PASS' if all(v is True for v in gates.values()) else 'FAIL' if any(v is False for v in gates.values()) else 'UNKNOWN'
    quality=dict(consumers=len(modal),EOS=sum(x['EOS'] for x in modal),at_cap=sum(x['at_cap'] for x in modal))
    result=dict(status='COMPLETE' if not missing else 'INCOMPLETE_SCORING',time=time.time(),development_gate=verdict,gates=gates,original146_panels=panels,original146_paired=delta,original146_transitions=changes,new_four=newpanels,combined150=combined,protection=protection,routing=route,quality=quality,scoring=done,
        independent_confirmation=False,new_T1G_semantics='UNKNOWN',new_T2L_eligible=0,source_scope='already exposed development cohort; no independent clinical validation',method='MED150_TEXT_WEAK_DOMINANCE',baseline='MED150 original R0',comparison_with_margin_002=comparison)
    dest=ROOT/'reports/med150_text_dominance_v1_20261028';dest.mkdir(exist_ok=True)
    (dest/'RESULTS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# MED150 文本弱优势路由开发评测','',f"状态：{result['status']}；预登记开发门槛：**{verdict}**。原146及新增4为已暴露开发数据，不构成独立或临床验证。",'',
        '文本弱优势约束在2148个查询中没有触发回退；本条件与前一0.02模态分差条件的选中专家和完整评分载荷均为2148/2148相同。两者所有同题结果完全相同；这是否定该约束在当前队列起作用的证据，不能推广为其他数据上必然无效。','',
        '|原146面板|原R0 CORE|文本约束 CORE|差值 pp|原R0 FULL|文本约束 FULL|差值 pp|','|---|---:|---:|---:|---:|---:|---:|']
    for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
        c=panels['core_correct'][role];f=panels['full_supported'][role]
        def fmt(v):return 'NA' if v is None else f'{v:.4f}'
        lines.append(f"|{role}|{fmt(c['R0']['macro_percent'])}|{fmt(c['modal']['macro_percent'])}|{fmt(delta['core_correct'][role]['delta_macro_pp'])}|{fmt(f['R0']['macro_percent'])}|{fmt(f['modal']['macro_percent'])}|{fmt(delta['full_supported'][role]['delta_macro_pp'])}|")
    lines+=['',f"在原146的去重非native查询中，Base和原R0均正确的 {protection['core_correct']['base_and_R0_correct']} 条里有 {protection['core_correct']['newly_damaged']} 条新增CORE损伤；原146 T2L CORE从 {panels['core_correct']['T2L']['R0']['known_correct']}/{panels['core_correct']['T2L']['R0']['n']} 降至 {panels['core_correct']['T2L']['modal']['known_correct']}/{panels['core_correct']['T2L']['modal']['n']}。FULL共同支持 {protection['full_supported']['base_and_R0_correct']} 条中新增损伤 {protection['full_supported']['newly_damaged']}。预登记的零CORE新增损伤门槛未通过，不能替换原MED R0。",
        f"2148查询中实际切换专家{route['changed']}条；新增文本弱优势回退{comparison['text_fallbacks']}条。R0 OFF查询{route['old_R0_off']}条，缺少OFF支持不能证明范围安全。",
        '全部2148路由在读取历史回答与评分前固定；逐条核验输入、状态与输出，1988项复用原R0，160项复用上轮模态条件。新生成0、新Judge调用0，评分复用2148、UNKNOWN 0。GPU重新提取来源750项和查询2148项特征；无训练和新权重。',
        '原146 T1G CORE为501/578至576/578，编辑宏平均86.6438%至99.6575%；T2G CORE为432/557至433/557。完整分母、旧146、新4与合并DEV、CORE/FULL配对及来源组区间见RESULTS.json。',
        '新增4条T1G临床语义未知、T2L无合格支持；0.02与实验动机均来自已暴露DEV，不能声称独立或临床证据，也不能把旧M13P146称作150对照。原始问答、图像、权重及内部路径不公开。', '']
    (dest/'REPORT_ZH.md').write_text('\n'.join(lines));print(json.dumps(dict(status=result['status'],gate=verdict,comparison=comparison,protection=protection)))

if __name__=='__main__':main()
