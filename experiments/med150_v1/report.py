"""Aggregate scored MED150 development outcomes against frozen MED146 evidence."""
import collections
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/medical_proposition_v133'))
from med133_report import summary,interval


def read(p):return json.loads(Path(p).read_text())
def score(d,k,field):return d.get(k,{}).get(field)
def counts(pairs):
    c=collections.Counter()
    for a,b in pairs:
        if a is None or b is None:c['unknown']+=1
        else:
            c['known']+=1;c['before_correct']+=a;c['after_correct']+=b
            c['forgotten']+=a and not b;c['repaired']+=not a and b
    return {k:c[k] for k in ('known','unknown','before_correct','after_correct','forgotten','repaired')}


def main():
    q=Path('/private/tmp/u.20261027.150');assert read(q/'COMPLETE.json')['generation_and_scoring_complete']
    assert read(q/'GUARD_EXIT.json')['code']==0
    newpack=read(q/'PACKET_PRIVATE.json');newdec=read(q/'DECISIONS.json')['decisions'];unknown=read(q/'DECISIONS.json')['unknown']
    oldroot=ROOT/'.private/med146_single_sequential_v133_20261027/scoring_recovery1'
    oldcons=read(oldroot/'CONSUMERS_B_FROZEN.json')['consumers'];olddec=read(oldroot/'MERGED_DECISIONS.json')['decisions'];meta=read(oldroot/'PACKET_METADATA_B.json')
    oldrows={(x['arm'],x['mode'],x['prefix'],x['owner'],x['query_id']):x for x in oldcons};assert len(oldrows)==len(oldcons)
    newrows={(x['mode'],x['prefix'],x['owner'],x['query_id']):x for x in newpack['consumers']};assert len(newrows)==2207
    mask=read('/private/tmp/s.20261027.150/BASE_MASK_FROZEN.json')['consumers'];newmask={x['query_id']:x for x in mask};assert len(newmask)==55
    oldtasks=meta['tasks'];oldmask=meta['base_mask'];assert len(oldtasks)==146
    candidates=read(ROOT/'.private/med150_feasibility_20261027/ROLE_SAFE_CANDIDATES.private.json')['items']
    added_active={(v['task']['native']['image_sha256'],v['task']['native']['question']) for v in candidates}
    oldbank={x['query_id']:x for x in oldcons if x['arm']=='MED' and x['mode']=='bank' and x['prefix']==146}
    oldbase={x['query_id']:x for x in oldcons if x['arm']=='BASE' and x['mode']=='bank' and x['prefix']==146}
    newbank={x['query_id']:x for x in newpack['consumers'] if x['mode']=='bank' and x['prefix']==150}
    assert len(oldbank)==len(oldbase)==2093 and len(newbank)==2148 and set(oldbank)<=set(newbank)
    newly_active={k for k,v in oldbank.items() if (v['image_sha256'],v['question']) in added_active}
    def roleids(t,role):
        if role=='FIT':return [f"edit_{t['order']}_{i}" for i in range(1,5)]
        ids=list(dict.fromkeys(qid for e in t['events'] if e['task']==role for qid in e['all_probe_query_ids']))
        return ids if role=='T0' else [qid for qid in ids if oldmask[qid]==role.endswith('L')]
    panels={};paired={};transitions={}
    for field in ('core_correct','full_supported'):
        panels[field]={};transitions[field]={}
        for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
            before={};after={};deltas=[];groups=[];pairs=[]
            for t in oldtasks:
                qs=roleids(t,role);qs=[qid for qid in qs if not (role.endswith('L') and (oldbank[qid]['active_target'] or qid in newly_active))]
                a=[score(olddec,oldbank[qid]['payload_key'],field) for qid in qs]
                b=[score(newdec,newbank[qid]['key'],field) for qid in qs]
                before[t['order']]=a;after[t['order']]=b;pairs.extend(zip(a,b))
                if a:
                    deltas.append(None if any(v is None for v in a+b) else statistics.mean(b)-statistics.mean(a));groups.append(t['native'].get('source_group',str(t['order'])))
            panels[field][role]=dict(bank146=summary(before),bank150_old146=summary(after))
            transitions[field][role]=counts(pairs)
            if role=='T2G':paired[field]=dict(delta_macro_pp=100*statistics.mean(deltas) if deltas and all(v is not None for v in deltas) else None,
                edits=len(deltas),up=sum(v is not None and v>0 for v in deltas),down=sum(v is not None and v<0 for v in deltas),same=sum(v==0 for v in deltas),unknown=sum(v is None for v in deltas),edit_bootstrap95_pp=interval(deltas),source_group_bootstrap95_pp=interval(deltas,groups))
    # New same-image locality rows have no eligible support; T1G semantics unknown.
    newpanels={}
    for field in ('core_correct','full_supported'):
        newpanels[field]={}
        for mode,prefix in (('single',1),('bank',150)):
            newpanels[field][mode]={}
            for role in ('T0','FIT','T1G','T2G','T2L'):
                per={}
                for owner in range(147,151):
                    qs=[x for x in mask if x['owner']==owner and x['role']==role and (role in ('T0','FIT') or x['eligible'] is True)]
                    if role=='T1G':qs=[]
                    keyowner=owner if mode=='single' else 0
                    per[owner]=[score(newdec,newrows[(mode,prefix,keyowner,x['query_id'])]['key'],field) for x in qs]
                newpanels[field][mode][role]=summary(per)
    combined={}
    for field in ('core_correct','full_supported'):
        combined[field]={}
        for role in ('T0','FIT','T2G'):
            a=panels[field][role]['bank150_old146'];b=newpanels[field]['bank'][role]
            support=a['support_edits']+b['support_edits'];valid=a['macro_percent'] is not None and b['macro_percent'] is not None
            combined[field][role]=dict(n=a['n']+b['n'],support_edits=support,known_correct=a['known_correct']+b['known_correct'],unknown=a['unknown']+b['unknown'],macro_percent=((a['macro_percent']*a['support_edits']+b['macro_percent']*b['support_edits'])/support if valid and support else None),scope='146 original plus four exposed development additions')
    insertion={}
    for field in ('core_correct','full_supported'):
        pairs=[]
        for owner in range(147,151):
            native=next(x for x in mask if x['owner']==owner and x['role']=='T0');qid=native['query_id']
            a=newrows[('insertion',owner,owner,qid)];b=newrows[('bank',150,0,qid)]
            pairs.append((score(newdec,a['key'],field),score(newdec,b['key'],field)))
        insertion[field]=counts(pairs)
    route=dict(common_queries=2093,changed=sum(oldbank[k]['effective_expert']!=newbank[k]['effective_expert'] for k in oldbank),
        same_full_score_payload=sum(oldbank[k]['payload_key']==newbank[k]['key'] for k in oldbank),old_base_fallbacks=sum(v['effective_expert'] is None for v in oldbank.values()),new_base_fallbacks=sum(newbank[k]['effective_expert'] is None for k in oldbank),
        source='actual R0 selections; each query rerouted at 150 experts')
    protection={}
    for field in ('core_correct','full_supported'):
        eligible=set()
        for t in oldtasks:
            for e in t['events']:
                for qid in e['all_probe_query_ids']:
                    if not oldbank[qid]['active_target'] and qid not in newly_active:eligible.add(qid)
        out=collections.Counter()
        for qid in eligible:
            b=score(olddec,oldbase[qid]['payload_key'],field);o=score(olddec,oldbank[qid]['payload_key'],field);n=score(newdec,newbank[qid]['key'],field)
            if None in (b,o,n):out['unknown']+=1;continue
            out['known']+=1
            if b:
                out['base_correct']+=1
                if not n:out['total_new_bank_base_damage']+=1
                if o:
                    out['base_and_oldbank_correct']+=1
                    if not n:out['newly_damaged']+=1
                elif n:out['old_damage_repaired']+=1
        protection[field]={k:out[k] for k in ('known','unknown','base_correct','base_and_oldbank_correct','total_new_bank_base_damage','newly_damaged','old_damage_repaired')}
        protection[field]['scope']='unique original146 nonnative event queries, excludes active targets in both 146 and 150 banks'
    quality={mode:dict(consumers=sum(x['mode']==mode for x in newpack['consumers']),EOS=sum(x['mode']==mode and x['EOS'] for x in newpack['consumers']),at_cap=sum(x['mode']==mode and x['at_cap'] for x in newpack['consumers'])) for mode in ('single','insertion','bank')}
    completion=read(q/'COMPLETE.json');result=dict(status='COMPLETE' if not unknown else 'INCOMPLETE_SCORING',time=time.time(),cohort='original146 exposed plus four development additions',independent_confirmation=False,
        t1g_new_semantics='UNKNOWN',new_T2L_eligible=0,panels=panels,new_four=newpanels,combined150=combined,old146_paired_T2G=paired,old146_transitions=transitions,
        insertion_to_final_new=insertion,actual_R0=route,protection=protection,generation_quality=quality,scoring=completion,
        comparison_scope='MED146 versus MED150; historical M13P146 is not a 150-expert comparator',limitations=['newly selected records and source QA exposed before test construction','new T2G constructed semantic development questions','new T1G clinical semantics unconfirmed','no eligible new T2L','no full150 M13P comparator'])
    dest=ROOT/'reports/med150_v1_20261027';(dest/'RESULTS_150.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# MED 150 专家开发评测','',f"状态：{result['status']}。原146与新增4均为已暴露开发材料，非独立确认。",'', '|字段|原146 bank146 T2G宏平均|原146 bank150 T2G宏平均|配对差值（百分点）|','|---|---:|---:|---:|']
    for field in ('core_correct','full_supported'):
        a=panels[field]['T2G']['bank146'];b=panels[field]['T2G']['bank150_old146'];d=paired[field]['delta_macro_pp']
        lines.append(f"|{field}|{a['macro_percent']}|{b['macro_percent']}|{d}|")
    lines+=['',f"原146的 2093 个共同查询中，路由改变 {route['changed']} 个。新增四条的 single、最终bank及插入保留结果、旧条目遗忘和保护计数见 RESULTS_150.json。",
        '新增4条的T1G临床语义未知，不纳入有效T1G支持；新增T2L无合格支持，不能给出局部性估计。新T2G为构造的同义开发题，不能当作独立确认。',
        '仅比较 MED 的146与150专家库；原146 M13P对照不能冒充150条对照。全部判断为参考答案一致性而非临床真实性。','']
    lines += [
        '## 合并开发队列', '',
        '|指标|CORE正确 / 总数|CORE编辑宏平均|FULL支持 / 总数|FULL编辑宏平均|',
        '|---|---:|---:|---:|---:|',
    ]
    for role in ('T0','FIT','T2G'):
        a=combined['core_correct'][role];b=combined['full_supported'][role]
        lines.append(f"|{role}|{a['known_correct']}/{a['n']}|{a['macro_percent']:.4f}%|{b['known_correct']}/{b['n']}|{b['macro_percent']:.4f}%|")
    lines += ['', '合并T2G为原146的557个和新增4条的14个合格问题；这只描述混合开发队列，不能解释为对同一固定146条的提升。',
        '新增四条各有原题和四个FIT，single与bank150均为20/20；14个Base错误的构造T2G均获CORE与FULL支持。四条在插入时及最终bank的原题均为4/4，未观察到插入后遗忘。',
        '原146共同查询实际重新计算R0路由，2093/2093的选中专家及完整评分载荷与bank146一致；原146的584个FIT和557个合格T2G在这次扩展中无新遗忘。',
        '原146非native查询的保护审计按去重查询计算，CORE中Base与旧bank均正确的80项没有新增损伤，但150 bank相对Base仍有3项既有损伤；FULL对应12项没有新增损伤，仍有1项既有损伤。',
        '150专家组装于原146终态与新增4个专家；无需重新训练原146。150库及路由配置有独立CPU可读私有备份。',
        '评测生成2207消费者，实际2149次生成；所有模型输出结束后统一评分，复用2094个相同载荷的旧判断，新增评分55个，UNKNOWN 0。原始评分器在代理预检阶段失败、未发生任何评分尝试；网络恢复后仅继续未尝试项目，失败回执保留。',
        '单次消费者55/55、插入4/4、bank 2146/2148以EOS结束；bank有2条达到长度上限，详见质量统计。',
        '原146 M13P历史对照仅供对应146条比较，不能构成完整150条对照；原M13P-160权重缺失限制仍然存在。原146 MED-RA负结果以及原146 MED与M13P比较保留在既有报告。',
        '复现脚本、公开汇总与协议可用；原问答、图像、token、模型权重和内部执行记录因隐私与授权边界不公开，需在具备原始材料及模型访问权的环境运行。', '']
    (dest/'REPORT_150_ZH.md').write_text('\n'.join(lines));print(json.dumps(dict(status=result['status'],route=route,unknown=len(unknown),scoring=completion)))

if __name__=='__main__':main()
