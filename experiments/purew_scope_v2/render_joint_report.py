"""Render the public, anonymous two-round report; never read clinical materials."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/purew_scope_v2_20261010'
read = lambda path: json.loads(path.read_text())
first = read(ROOT / 'reports/purew_scope_v1_20261009/RESULTS.json')
second = read(OUT / 'RESULTS.json')
joint = read(OUT / 'JOINT_ANALYSIS.json')
audit = read(OUT / 'FINAL_AUDIT.json')
arms = ('PROJECTED', 'UNPROJECTED', 'GATED', 'ALWAYS_ON')
panels = dict(first['panels'], **second['panels'])
roles = ('NATIVE', 'FIT', 'GFIT', 'T1G', 'T2G', 'T1L', 'T2L', 'HELDOUT', 'ORIGINAL63', 'FRESH_BASE_CORRECT_HELD', 'T1L_SAME_REFERENCE', 'T1L_DIFFERENT_REFERENCE')
parts = []


def paragraph(text):
    parts.append('<p>' + html.escape(text) + '</p>')


def table(headers, rows):
    parts.append('<div class="scroll"><table><thead><tr>' + ''.join('<th>' + html.escape(str(x)) + '</th>' for x in headers) + '</tr></thead><tbody>')
    for row in rows:
        parts.append('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>')
    parts.append('</tbody></table></div>')


def cell(panel):
    return f"{panel['candidate_correct']}/{panel['queries']}" + (f"（缺 {panel['missing']}）" if panel['missing'] else '')


# These checks catch denominator drift and accidental conversion of missing scores to errors.
for arm in arms:
    for role in roles:
        p = panels[arm][role]
        assert p['candidate_correct'] + p['new_damage'] + p['still_wrong'] + p['missing'] == p['queries']
        assert p['queries'] == panels['PROJECTED'][role]['queries']
assert audit['new_valid'] + audit['missing_payloads'] == audit['new_attempts']
assert audit['scored_consumers'] + audit['missing_consumers'] == audit['formal_consumers']
assert audit['pending_or_reserved'] == 0
for round_data in joint.values():
    for arm, histories in round_data['anonymous_query_histories'].items():
        assert len(histories) == 24
        assert all(len(h['correctness']) == 9-h['first_prefix'] and set(h['correctness']) <= {0, 1, None} for h in histories)
        assert sum(len(h['correctness']) for h in histories) == 108
        assert sum(h['correctness'][0] == 1 for h in histories) == round_data['insertion'][arm]['correct']

parts.append('<h1>纯 W 知识编辑：两轮联合结果</h1>')
paragraph('有限实验已终止。第二轮两臂各完成 1280 步和全部生成；评分终态存在网络失败缺失，不能称完整评分成功。冻结主臂 GATED 未达标，ScopeEdit 传播门控的行为改善未建立。首轮独立冻结的 PROJECTED DEV 成功保留。')
paragraph('这是一个暴露过的开发顺序与种子的机制适配研究。无独立临床或 SOTA 确认；没有完整复现 LOKI、DOW-KE 或 ScopeEdit 原论文与作者代码。全部四臂直接修改原有 W，模型参数数量不变，无推理编辑分支。')
parts.append('<h2>原 Base 与四臂：完整分母</h2>')
paragraph('表格给出已确认正确数/全部题数，缺评分单列；缺失既不记为正确，也不记为错误。对应可能正确范围为 [正确数, 正确数+缺失数]/分母，不是置信区间。GFIT 是训练锚点；FIT 也是训练材料。只有 T1G/T2G 属于这里的未训练泛化。保护题各子集相互重叠，不合并为独立总样本。')
table(('角色', 'Base', *arms), [(role, f"{panels['PROJECTED'][role]['Base_correct']}/{panels['PROJECTED'][role]['queries']}", *(cell(panels[a][role]) for a in arms)) for role in roles])
paragraph('96 题保护集：四臂均保留 Base 正确的 62/62，新增损伤 0/62；首轮总正确 63/96，第二轮两臂均 66/96。第二轮共同增加 3 题，门控臂与恒开臂相同，不能归因于门控。FIT 的 Base 正确题第二轮两臂各损伤 2/3；这些训练题不属于 96 题保护集，但应保留该退步。')
parts.append('<h2>第二轮：只用配对比较判断门控</h2>')
paragraph('下面仅在同题两臂均有评分时列胜/负/平；同时给出全分母正确数差 GATED−ALWAYS_ON 的最坏/最好界。完全相同且完整绑定的缺失回答共用同一未知评分，差值固定为零；其他缺失按所有可能评分取界。界不是统计显著性，也未假定缺失随机。')
table(('角色', '全部题', '双方有评分', '门控胜', '门控负', '平', '至少一方缺', '全分母差值界'), [(role, p['queries'], p['both_scored'], p['gate_or_first_win'], p['gate_or_first_loss'], p['tie'], p.get('at_least_one_missing', 0), p['first_minus_second_correct_count_bounds']) for role,p in joint['ROUND2']['paired_final'].items()])
paragraph('源题的全分母差值固定为 −1；GFIT 为 −3 至 −2；T2G 为 −6 至 −1。即使补齐缺评分，这三个面板的门控正确数也低于恒开。保护集差值为零。T1G 差值界跨零，不作完整排名。第二轮没有建立无逐编辑回退的改善，也不作显著性或总体优越性声明。')
parts.append('<h2>插入、逐编辑与连续保持</h2>')
table(('实验臂', '自身插入时正确/24', '确认错误', '缺评分'), [(a, f"{p['correct']}/{p['queries']}", p['wrong'], p['missing']) for rd in joint.values() for a,p in rd['insertion'].items()])
paragraph('首轮 PROJECTED 的 108 个源题/GFIT 前缀观察全部正确；UNPROJECTED 为 103/108，编辑 1 的一个 GFIT 在前缀 3–7 变错、前缀 8 恢复。投影对这条保持链有帮助，但其 T2G 比未投影少 3/32，没有全方位胜出。第二轮插入本身已有确认错误；不能把未插入成功算作“保持成功”，也不能用平均准确率补偿逐编辑失败。')
paragraph('第二轮编辑 1 的 GFIT1：GATED 为 0,0,1,1,0,0,0,0，ALWAYS_ON 为 0,0,1,0,0,0,0,0。门控把这条后续变错延后一轮，但两臂初次插入都错，最终也错，不能据此报门控成功。编辑 3、5 的三题全部插入并保持；编辑 2、4 有缺评分；编辑 6、7、8 保留完整错误与缺失，不选成功子集。')
for rd_name, rd in joint.items():
    parts.append('<h3>' + rd_name + ' 前缀完整分母</h3>')
    table(('臂','前缀','角色','正确/全题','错误','缺评分'), [(a,p['prefix'],p['role'],f"{p['correct']}/{p['queries']}",p['wrong'],p['missing']) for a, values in rd['prefix_role_panels'].items() for p in values])
    parts.append('<details><summary>24 条匿名逐题保持链/臂（1 正确，0 错误，? 缺评分；从自身插入前缀开始）</summary>')
    table(('臂','编辑','角色/匿名序号','起始前缀','随后各前缀评分'), [(a,h['owner'],h['query'],h['first_prefix'],','.join('?' if x is None else str(x) for x in h['correctness'])) for a,values in rd['anonymous_query_histories'].items() for h in values])
    parts.append('</details>')
parts.append('<h3>每个编辑的终态源题、训练锚点与未训练泛化</h3>')
table(('臂','编辑','源题','FIT','GFIT','T1G','T2G'), [(a,o['owner'],*(cell(o['panels'][role]) for role in ('NATIVE','FIT','GFIT','T1G','T2G'))) for source in (first,second) for a,owners in source['per_owner'].items() for o in owners])
parts.append('<h2>机制诊断与必要适配</h2>')
paragraph('第二轮采用初始 Base 单层 down_proj 的固定零空间 448/64 私有/共享正交坐标；真实原生回答 CE 与 BASIS 的 Base 分布 KL 共同更新原 W。图文特征一致性门控控制共享写入及共享历史 RLS；历史只在本次编辑之后加入。恒开臂 γ=1，其他协议相同。依据论文公式 14–24 的 sqrt(rho) 历史更新，未声称作者代码另一归一化版本的逐行复现。固定层、固定低维空间、医疗锚点监督以及训练时门控合并 W 都是显式适配。')
paragraph('门控确实生效：56 个来源 γ 的范围 0.1117–0.6104，恒开为 1。8 个编辑的终步加权 CE 均高于恒开；两臂的 loss 下降不能保证自由生成回答正确。该证据与本设置共享写入受限、拟合较弱一致，尚不能证明它是所有失败的唯一原因。')
table(('编辑','GATED CE 首→末','恒开 CE 首→末','GATED γ 均值','GATED KL 末','恒开 KL 末'), [(i+1, f"{g['weighted_CE_first']:.4f} → {g['weighted_CE_last']:.4f}",f"{u['weighted_CE_first']:.4f} → {u['weighted_CE_last']:.4f}",f"{g['gamma_mean']:.4f}",f"{g['BASIS_KL_last']:.6f}",f"{u['BASIS_KL_last']:.6f}") for i,(g,u) in enumerate(zip(joint['ROUND2']['training_summary']['GATED'],joint['ROUND2']['training_summary']['ALWAYS_ON']))])
paragraph('末态私有历史逆矩阵两臂完全相同：trace 433.9713、最小特征值 0.03876。共享逆矩阵 GATED 的 trace 63.4305、最小特征值 0.54082；恒开为 62.9833、0.28180。全部历史矩阵对称误差为 0 且正定。门控同时减小当前共享写入和共享历史累计，后者保留更大的逆矩阵响应；不可只用 γ 小就断言传播更安全。真实特征不变与梯度原 W 写入的机械准入精确通过。')
paragraph('跨轮同时改变了可写空间和历史几何，不能单独归因 Scope 门控。即使第二轮全部缺评分都正确，GATED 源题最多 6/8、GFIT 10/16、T1G 24/32、T2G 19/32；恒开最多 7/8、13/16、24/32、22/32，均低于首轮相应结果。固定低维空间/历史机制是有依据的待研究解释，现有对照不能区分各因素的独立贡献。')
parts.append('<h2>评分、工程失败、成本与交付边界</h2>')
paragraph('1019 个正式输出消费者全部生成并完成绑定准入；267 个 Base 输出复用，754 次新生成包含 2 个机械 Base 身份探针和 2 个原生重载探针。消费者去重后 526 个评分 payload；352 个完全绑定旧评分复用。174 个新 payload 只尝试一次，109 有效；两个批次（50、15）网络传输失败导致 65 payload / 145 消费者缺评分。874/1019 消费者有评分，0 待办或保留任务。四评分 worker 均结束；SCORING_COMPLETE 回执表示队列终止，不表示评分无缺失。')
paragraph('实际有效 Judge 为 gpt-6-astra / medium；新有效批次与继承批次的输入绑定、模型、隔离、无工具调用和响应格式已检查。原错误为 Connection failed: error sending request；不推断管理员、模型或鉴权故障。未重试、未换模型、未用缺失当错误。原自动聚合的差值和“regression”字段含缺失，联合报告以双方已评分比较和全分母差值界取代其归因用法。')
paragraph('保留 4 份远端工程失败回执：预算兼容文件缺失及两次旧/新 scope_math 同名模块隔离错误等。改名 propagation_math 的修复内容不变；两次评价失败均在模型加载前发生、产生 0 终态输出。训练、机械准入和已有生成未重复，科学步数/截止/数据/阈值未重置。另修复本机 scorer 续跑识别：Python exec 后解释器路径会变化，按进程出生时间和中性入口判定原 worker，避免误判活进程。')
table(('账本','首轮新增','第二轮新增','含继承历史累计'), [('GPU 进程小时',f"{first['resource']['new_GPU_process_hours']:.6f}",f"{second['resource']['new_GPU_process_hours']:.6f}",f"{second['resource']['cumulative_GPU_process_hours']:.6f}"),('Judge payload 尝试',first['resource']['new_Judge'],second['resource']['new_Judge'],second['resource']['cumulative_Judge'])])
paragraph('第二轮实际提交 2560 更新、10244 次 backward（含机械 4 次），新生成 754 次；forward 总计含逐 token 解码，不能当作非生成 forward 计数。自产权重/缓存峰值 1,644,945,101 bytes；消费完成后按本轮删除回执清理 122 文件、713,661,672 bytes，未保留部署矩阵或无限 checkpoint 归档。小型步日志、几何、失败和评分证据保留；历史首轮只读，其他 GPU/资产未操作。')
paragraph('冻结判据分别保留：首轮主臂 DEV_SUCCESS；第二轮两臂 FROZEN_SUCCESS_CRITERIA_NOT_MET，缺评分明确标记 INCOMPLETE_SCORING。已知插入/终态错误本身即可否定第二轮成功，不只是网络原因。有限两臂研究在联合匿名报告公开交付后结束，不自动补评分、调参或启动第三轮。公开代码、JSON 与本报告；私有图像、问题、答案、病人标识、Judge 输入和第三方原始快照不公开。')
parts.append('<p>数据：<a href="RESULTS.json">第二轮结果</a> · <a href="JOINT_ANALYSIS.json">逐编辑/配对数据</a> · <a href="FINAL_AUDIT.json">终态审计</a> · <a href="PROTOCOL.html">冻结协议</a> · <a href="../purew_scope_v1_20261009/RESULTS.json">首轮原结果</a></p>')
page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>纯 W 两轮联合结果</title><style>body{max-width:1150px;margin:35px auto;padding:0 20px;color:#17202c;font:16px/1.7 system-ui}h1{font-size:30px}h2{margin-top:36px;font-size:23px}table{border-collapse:collapse;font-size:14px;width:100%}th,td{padding:7px 10px;border:1px solid #d9dfe8;text-align:left;white-space:nowrap}th{background:#eaf0f6}.scroll{overflow:auto;margin:18px 0}details{margin:20px 0}a{color:#1765a5}</style><body>' + ''.join(parts) + '</body></html>'
(OUT / 'JOINT_REPORT.html').write_text(page)
print('PASS: denominators, missing values, insertions and 108-prefix histories; rendered JOINT_REPORT.html')
