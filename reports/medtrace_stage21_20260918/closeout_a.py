"""Machine-derived A milestone. B remains pending; Stage21 is not complete."""
import json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from report import report
from scripts.medtrace.astra_judge_bundle import read
from scripts.medtrace.stage19_fasttrack_budget import write


def run(root,destination):
    root=Path(root);destination=Path(destination);pub=root/'public';report(root)
    status=read(pub/'REPORT_STATUS.json');assert status['A_highest_scored_N']==45 and status['A_all_generated_outputs_scored']
    generated=read(pub/'GENERATED.json');assert generated['N']==45 and read(pub/'EXIT_NO_H_HSIC.json')['exit_code']==0
    rows=read(pub/'COMPARISON_DERIVED.json')['results'];resource=read(pub/'RESOURCE_PROFILE.json');gpu=read(pub/'GPU_BUDGET_LEDGER.json');budget=read(pub/'BUDGET_LEDGER.json')
    assert all('seconds' in s for s in gpu['sessions'])
    used=sum(s['seconds'] for s in gpu['sessions']);assert used<=12600 and budget['new_judgment_items_dispatched']<=800
    def metric(arm,panel,name):return next(r for r in rows if r['arm']==arm and r['N']==45 and r['panel']==panel and r['metric']==name)
    def cell(arm,panel,name):
        r=metric(arm,panel,name);return f"{r['numerator']:g}/{r['denominator']} ({r['value']:.1%})" if r['value'] is not None else 'NA'
    F='Stage20_FACT_H';A='NO_H_HSIC';B='Stage20_BE'
    table='|方法|native|insertion|old H保持|new H保持|old U保持|new U保持|PairCorrect|异图正向泛化|存储/编辑|\n|---|---|---|---|---|---|---|---|---|---|\n'
    columns=[('native','accuracy'),('insertion','accuracy'),('old_H_eval','Retention'),('new_H_eval','Retention'),('old_U_eval','Retention'),('new_U_eval','Retention'),('old_anchors','PairCorrect'),('positive_positive_image','accuracy')]
    for arm,label,profile in [(F,'Stage20 FACT-HSIC','Stage20_FACT_H'),(A,'Stage21 NO_H-HSIC','NO_H_HSIC_LOGICAL_BANK'),(None,'Stage21 FACT固定层：待澄清',None),(B,'Stage20 BalancEdit（实际L31）','Stage20_BE')]:
        cells=[cell(arm,panel,m) for panel,m in columns] if arm else ['NA']*len(columns)
        storage=f"{resource['profiles'][profile]['mean_file_bytes']/1024**2:.3f} MiB" if profile else 'NA'
        table+='|'+label+'|'+'|'.join(cells+[storage])+'|\n'
    brief=f'''# Stage21 导师更新：A 完成，B 尚未执行

N=45 的 NO_H 全流及 11/19/32/45 四个真实前缀已生成并完整评分。前11个兼容 NO_H 专家复用，第12—45条新训练；每条都重新记录了原始 HSIC 选择。Stage20 正式文件与原标签未修改，未重新判分。以下为新的比较视图，非替换 Stage20 原表。

{table}
U“保持”为去除与已插入 native 同问题/同属性后的严格分母；固定面板保持另见 JSON。PairCorrect 是11锚点关联计权的加权量（可为小数分子），旧 H 实际只有6个QA/2来源，不能当11份独立保护证据。新增 H24QA/8来源、U14QA/8来源。患者/研究独立性 UNKNOWN，无临床签署。

## 机制判断

- **SUPPORTED（本固定 DEV 流描述性证据）**：H 对旧 H 的保持有所改善；相同自然路由下，不同专家输出确实不同。低秩 writer 的存储占用更小。
- **NOT SUPPORTED**：当前 HSIC 动态选层主张（旧审计与 A 新记录均45/45选L30）；H 对 native 的额外收益；H 对新 H/U 均有稳定正贡献；NO_H 全指标都不低于 FACT；长序列确认。
- **INCONCLUSIVE**：层位置与 writer 类型分别贡献多少。B 未运行，不能声称已完成同层控制。

新 H 保持 FACT {cell(F,'new_H_eval','Retention')}、NO_H {cell(A,'new_H_eval','Retention')}；新 U 严格保持均为 {cell(A,'new_U_eval','Retention')}。旧 H 收益伴随旧 U 取舍，不能宣布普遍保护优势。Wilson95% 仅描述性边际区间，来源等权和逐来源删除范围已输出；不假设同来源QA独立，不报告显著性。

累计新增 GPU {used:.1f}/12600秒（{used/60:.1f}分钟），新增判定 {budget['new_judgment_items_dispatched']}/800，精确缓存复用计数 {budget['exact_reuse_count']}。复用计数是各消费者批次内去重后的缓存命中累计，不是独立QA数量。Stage21 余额不转成未来新额度；后续所有阶段另共用28800秒/2000项。实际货币费用未由提供商暴露，记NA；tokens见账本。

B 阻塞原因是实际冻结 BE 写入 `model.layers.31.mlp.up_proj`，而原计划是 FACT-L21 `down_proj`，两者不是同层。等待用户选择保留 L21 位置对照，或另锁 L31 同 Transformer 层对照（仍保留 up/down 模块差异）。不改写 Stage20 工件来消除此差异。

存储列为现有序列化文件，FACT 文件还含 optimizer、RNG 和训练曲线。文件大小比 BE/FACT={resource['BE_to_FACT_file_storage_ratio']:.2f}；纯 writer 张量比={resource['BE_to_FACT_writer_tensor_ratio']:.2f}，不能混为同一指标，也不能当训练算力比。详细测量与缺失项见 RESOURCE_PROFILE.json。
'''
    (pub/'ADVISOR_UPDATE_ZH.md').write_text(brief)
    route=read(pub/'ROUTING_MECHANISM_ANALYSIS.json');counts={}
    for r in route['aggregate']:
        c=counts.setdefault(r['arm'],{})
        for k,v in r['transitions'].items():c[k]=c.get(k,0)+v
    switched={arm:dict(TF=counts[arm].get('1->0|switch=True',0),previously_correct=sum(counts[arm].get(k,0) for k in ['1->0|switch=True','1->1|switch=True'])) for arm in [F,A,B]}
    write(pub/'SWITCH_CONDITIONAL_DIAGNOSTIC.json',dict(rows=switched,common_switch_events=15,scope='Adjacent registered prefixes; repeated queries across transitions are not independent',denominators_differ_by_prior_correctness=True))
    q=[
    ('Q1 NO_H在45后是否仍≥FACT','NOT SUPPORTED',f"并非逐指标成立。native同为45/45；旧H保持 FACT {cell(F,'old_H_eval','Retention')} vs NO_H {cell(A,'old_H_eval','Retention')}，旧U则 FACT {cell(F,'old_U_eval','Retention')} vs NO_H {cell(A,'old_U_eval','Retention')}。"),
    ('Q2 H是否提高native编辑成功','NOT SUPPORTED','两者native和当次写入均45/45；本流没有可观察的增益，不能外推为H在任何场景都无效。'),
    ('Q3 H对新H/U是否稳定正贡献','NOT SUPPORTED',f"新H保持仅多1/18，严格新U相同；新H总体正确性 FACT {cell(F,'new_H_eval','accuracy')} vs NO_H {cell(A,'new_H_eval','accuracy')}。参考来源宏平均、LOO和描述性区间，不能称稳定共同增益。"),
    ('Q4 同L21下FACT是否优于BE','INCONCLUSIVE','B未执行，且实际BE为L31 up_proj，原问题的同L21前提不成立。'),
    ('Q5 FACT-L30与FACT-L21差多少','INCONCLUSIVE','未取得冻结层控制结果，保留NA。'),
    ('Q6 优势来自layer还是writer','INCONCLUSIVE','H有局部取舍证据；层/模块/容量/优化步数差异尚未被充分控制，不能单独归因。'),
    ('Q7 HSIC有input-dependent层选择吗','NOT SUPPORTED','CURRENT HSIC DYNAMIC-LAYER CLAIM NOT SUPPORTED。旧45与本轮重新记录45均L30，选层熵0；分数随输入变化不等于所选层具有输入差异。'),
    ('Q8 相同routing switch下谁更易True→False','SUPPORTED_DESCRIPTIVE',f"三种已完成方法四个端点的自然路由逐查询一致，共15个相邻前缀切换事件。TF/各方法切换前正确数：FACT {switched[F]['TF']}/{switched[F]['previously_correct']}，NO_H {switched[A]['TF']}/{switched[A]['previously_correct']}，BE {switched[B]['TF']}/{switched[B]['previously_correct']}。这是不同既有正确集合上的描述性结果，事件重复、样本很小，不做独立推断。差异出现在相同路由对应的专家载荷；不排除未控layer/module差异。"),
    ('Q9 低秩patch实际storage ratio','SUPPORTED',f"实测文件 BE/FACT {resource['BE_to_FACT_file_storage_ratio']:.2f}，纯张量 {resource['BE_to_FACT_writer_tensor_ratio']:.2f}。前者含不同resume包装，不是严格等价部署格式。BE张量结构仅CPU读取一个兼容样本，45个文件均量取实际文件大小。"),
    ('Q10 支持long-sequence claim吗','NOT SUPPORTED','仍是固定45条小规模DEV纯流、单一顺序和有限评价来源；不支持100/146长流、患者独立或临床确认。')]
    write(pub/'MECHANISM_QUESTIONS.json',dict(answers=[dict(question=a,status=b,answer=c) for a,b,c in q],Stage21_complete=False))
    analysis='# Stage21 A机制分析（B待澄清）\n\n'+''.join(f'## {a}\n\n**{b}** — {c}\n\n' for a,b,c in q)
    analysis+='EXTRA_FULL11_UNSUPPORTED 保持；没有改样本、标签、tokenization、路由阈值或强制专家。后续实验在B问题明确且本阶段收口后再预注册，不能按现有分数挑好前缀停止。\n'
    (pub/'STAGE21_ANALYSIS_ZH.md').write_text(analysis)
    write(pub/'CHECKPOINT_CONSUMERS.json',dict(A_final_bank=dict(experts=45,new=34,reused=11,status='RETAIN',pending=['Stage21 controlled-layer comparison and registered mechanism reproducibility','bounded follow-on design inventory']),router='RETAIN',W0='RETAIN for B or explicitly registered follow-on',Stage20_artifacts='IMMUTABLE; no cleanup',temporary_cleanup='None: sufficient space; no active resume state deleted'))
    write(pub/'DELIVERY_STATUS.json',dict(milestone='A_COMPLETE_SCORED',Stage21_complete=False,B='PENDING_LAYER_CLARIFICATION',GPU_seconds=used,new_judgments=budget['new_judgment_items_dispatched'],scored_prefixes=[11,19,32,45],Stage20_formal_files_unchanged=True))
    # Explicit allowlist: no paths, raw answers, tokens, identity mappings or private consumers.
    names=['NO_H_PURE45_RESULTS.json','NO_H_PURE45_RESULTS.csv','COMPARISON_DERIVED.json','RISK_DIFFERENCES.json','ROUTING_MECHANISM_ANALYSIS.json','RESOURCE_PROFILE.json','GPU_BUDGET_LEDGER.json','BUDGET_LEDGER.json','STAGE20_IMMUTABILITY_CHECK.json','HSIC_RECORDED_A.json','SWITCH_CONDITIONAL_DIAGNOSTIC.json','ADVISOR_UPDATE_ZH.md','STAGE21_ANALYSIS_ZH.md','MECHANISM_QUESTIONS.json','CHECKPOINT_CONSUMERS.json','DELIVERY_STATUS.json']
    for name in names:
        data=(pub/name).read_bytes()
        assert not any(x in data for x in [b'"raw_answer"',b'"raw_token_ids"',b'"image_path"',b'"canonical_edit_id"',b'/Users/',b'/root/',b'/remote-home/']),name
        (destination/name).write_bytes(data)
    return names


if __name__=='__main__':run(sys.argv[1],sys.argv[2])
