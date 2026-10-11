"""Aggregate anonymous probe metrics; no access to private questions or weights."""
import json
import statistics
import sys
from pathlib import Path


def distribution(values):
    assert values and all(v is not None for v in values)
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),minimum=min(values),maximum=max(values))


def summarize(rows):
    assert {(r['owner'],r['state']) for r in rows}=={(i,s) for i in range(1,9) for s in ('BASE','FINAL')}
    assert len(rows)==16 and all(len(r['gauge'])==3 and len(r['visual'])==7 for r in rows)
    assert all([v['role'] for v in r['visual']]==['NATIVE']+['FIT']*4+['GFIT']*2 for r in rows)
    result={}
    for state in ('BASE','FINAL'):
        selected=[r for r in rows if r['state']==state]
        geometry={}
        for mode in ('IDENTITY','PERMUTATION','DIAGONAL'):
            values=[next(v for v in r['gauge'] if v['mode']==mode) for r in selected]
            geometry[mode]={k:distribution([v['operator_update'][k] for v in values]) for k in ('norm','norm_ratio','cosine','difference_norm')}
            geometry[mode].update({k:distribution([v[k] for v in values]) for k in ('native_FIT_gain','GFIT_gain')})
        vision={}
        for role in ('ALL','NATIVE','FIT','GFIT'):
            vision[role]={}
            for mode in ('ORIGINAL','SWAPPED_VISUAL','ZERO_VISUAL'):
                entries=[v for r in selected for v in r['visual'] if role=='ALL' or v['role']==role]
                values=[v['conditions'][mode] for v in entries]
                # Every owner contributes the same number of questions within each role.
                vision[role][mode]=dict(questions=len(values),owners=len(selected),
                    first_argmax_changed=sum(v['first_argmax_changed'] for v in values),
                    first_target_top1=sum(v['first_target_top1'] for v in values),
                    all_target_top1=sum(v['all_target_top1'] for v in values),
                    any_answer_token_changed=sum(v['changed_tokens']>0 for v in values),
                    lost_original_all_target_top1=sum(v['conditions']['ORIGINAL']['all_target_top1'] and not v['conditions'][mode]['all_target_top1'] for v in entries),
                    gained_all_target_top1=sum(not v['conditions']['ORIGINAL']['all_target_top1'] and v['conditions'][mode]['all_target_top1'] for v in entries),
                    **{k:distribution([v[k] for v in values]) for k in ('first_KL','first_delta_NLL','delta_NLL','mean_KL','first_target_margin')})
            for mode,value in vision[role].items():
                assert vision[role]['ORIGINAL']['all_target_top1']-value['lost_original_all_target_top1']+value['gained_all_target_top1']==value['all_target_top1']
        per_owner=[]
        for row in sorted(selected,key=lambda r:r['owner']):
            per_owner.append(dict(owner=row['owner'],revised_owner8=row['owner']==8,
                gauge=[dict(mode=v['mode'],operator_update=v['operator_update'],native_FIT_gain=v['native_FIT_gain'],GFIT_gain=v['GFIT_gain']) for v in row['gauge']],
                visual={m:dict(changed=sum(v['conditions'][m]['first_argmax_changed'] for v in row['visual']),
                    any_answer_token_changed=sum(v['conditions'][m]['changed_tokens']>0 for v in row['visual']),
                    mean_delta_NLL=statistics.mean(v['conditions'][m]['delta_NLL'] for v in row['visual']),
                    mean_first_KL=statistics.mean(v['conditions'][m]['first_KL'] for v in row['visual'])) for m in ('SWAPPED_VISUAL','ZERO_VISUAL')}))
        result[state]=dict(geometry=geometry,vision=vision,per_owner=per_owner,
            repeat_max_logit_difference=max(r['repeat_noise']['maximum_logit_difference'] for r in selected),
            repeat_max_first_KL=max(r['repeat_noise']['first_KL'] for r in selected),
            gauge_parity_max_logit_difference=max(v['before_native_parity']['maximum_logit_difference'] for r in selected for v in r['gauge']))
    return result


def render(summary,complete):
    lines=['# 最小机制诊断：数值结果','',
        '固定 8 owner、56 条已暴露源题；BASE 为零 TT 初始态，FINAL 为保留的 C160 终态。每种表示使用全新优化器的一步探针，不是续跑。',
        '', '## 等价 TT 表示', '',
        '| 状态 | 表示 | 算子更新范数比，均值 [最小,最大] | 方向余弦，均值 [最小,最大] | native/FIT 平均 NLL 改善 | GFIT 平均 NLL 改善 |',
        '|---|---|---|---|---|---|']
    def fmt(d):return f"{d['mean']:.6g} [{d['minimum']:.6g}, {d['maximum']:.6g}]"
    for state,part in summary.items():
        for mode,g in part['geometry'].items():
            lines.append(f"| {state} | {mode} | {fmt(g['norm_ratio'])} | {fmt(g['cosine'])} | {g['native_FIT_gain']['mean']:.6g} | {g['GFIT_gain']['mean']:.6g} |")
    lines+=['','同一原坐标更新为范数比与方向参照。范数比与方向余弦分别反映尺度与方向；正 NLL 改善表示目标损失下降。初始态 G1=0 导致部分 core 梯度为零，不代表训练已更新全部 core。',
        '', '## 视觉敏感性', '',
        '| 状态 | 条件 | 首 token argmax 改变 / 56 | native 改变 / 8 | FIT 改变 / 32 | GFIT 改变 / 16 | 平均首 token KL | 平均首 token ΔNLL | 平均全答案 ΔNLL |',
        '|---|---|---|---|---|---|---|---|---|']
    for state,part in summary.items():
        for mode in ('SWAPPED_VISUAL','ZERO_VISUAL'):
            v=part['vision']; a=v['ALL'][mode]
            lines.append(f"| {state} | {mode} | {a['first_argmax_changed']} | {v['NATIVE'][mode]['first_argmax_changed']} | {v['FIT'][mode]['first_argmax_changed']} | {v['GFIT'][mode]['first_argmax_changed']} | {a['first_KL']['mean']:.6g} | {a['first_delta_NLL']['mean']:.6g} | {a['delta_NLL']['mean']:.6g} |")
    lines+=['','预先记录的完整目标 token 辅助指标：','','| 状态 | 条件 | 任一答案 token argmax 改变 / 56 | 所有目标 token 均为 top1，原图 → 干预 | 平均全答案 KL |','|---|---|---|---|---|']
    for state,part in summary.items():
        for mode in ('SWAPPED_VISUAL','ZERO_VISUAL'):
            v=part['vision']['ALL']; a=v[mode]
            lines.append(f"| {state} | {mode} | {a['any_answer_token_changed']} | {v['ORIGINAL']['all_target_top1']} → {a['all_target_top1']} | {a['mean_KL']['mean']:.6g} |")
    lines+=['','ΔNLL 为干预减原图，正值表示原目标变难。每个 owner 在各角色内等权；56 条并非 56 个独立任务。错配条件没有医学反事实标签，不能把目标 top1 当作该条件正确率。完整答案使用 teacher forcing，首 token 不含答案前缀；固定答案首词可能限制其语义分辨力。首词不变不能代表后续词或完整回答不变。BASE 原图的首目标 token top1 为 0/56，FINAL 为 55/56，应保留这一指标的解释边界。',
        '', '## 数值与预算', '']
    for state,part in summary.items():
        lines.append(f"- {state}：原图重复最大 logit 差 {part['repeat_max_logit_difference']:.6g}，首 token KL {part['repeat_max_first_KL']:.6g}；等价表示更新前最大 logit 差 {part['gauge_parity_max_logit_difference']:.6g}。")
    lines+=[f"- 实际 GPU {complete['GPU_hours']:.6f} 小时；计数 {complete['counts']}。另有固定 56 次视觉预处理。", '',
        '## 解释边界', '',
        '坐标敏感性只能支持开展函数空间更新研究，不能证明它是历史保护/泛化失败的主因，也不证明新优化器有效。视觉敏感性只度量当前输入条件下的预测变化；置零为分布外消融，固定错配图可能仍共享答案。应结合连续 KL、argmax、目标概率及完整 NLL，不能只凭 argmax 不变宣布模型不看图。', '',
        '没有新 seed、强度扫描、独立测试集、LoRA 公平比较、自由生成或 Judge；未自动启动后续算法。全部匿名逐题数值保存在 rows/，逐 owner 结果和 owner8 修订标志在 SUMMARY.json。']
    return '\n'.join(lines)+'\n'


def selfcheck():
    assert distribution([1.,2.,3.])==dict(n=3,mean=2.,median=2.,minimum=1.,maximum=3.)
    try:summarize([])
    except AssertionError:pass
    else:raise AssertionError('Missing owners must fail')


if __name__=='__main__':
    selfcheck(); root=Path(sys.argv[1]); complete=json.loads((root/'COMPLETE.json').read_text())
    assert complete['status']=='COMPLETE' and complete['counts']==dict(forwards=864,backwards=144,updates=48,generations=0,Judge=0)
    rows=[json.loads(p.read_text()) for p in sorted((root/'rows').glob('*.json'))]
    summary=summarize(rows)
    (root/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (root/'NUMERICAL_REPORT_ZH.md').write_text(render(summary,complete))
    print('REPORT PASS',len(rows),'owner-state cells')
