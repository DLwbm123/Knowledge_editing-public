"""Anonymous aggregate report, including all five arms and registered contrasts."""
import html,json,os
from pathlib import Path

out=Path(os.environ['WORKSPACE'])/'reports/purew_scope_v3_20261010';r=json.loads((out/'RESULTS.json').read_text())
parts=['<h1>五臂纯 W 归因对照</h1><p>固定五臂、同一已暴露DEV顺序；缺评分不记为错误。无独立临床或SOTA确认，不选择赢家、不自动调参。</p>']
def table(head,rows):
    parts.append('<div style="overflow:auto"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr>')
    for row in rows:parts.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>')
    parts.append('</table></div>')
def cell(p):return f"{p['candidate_correct']}/{p['queries']} (missing {p['missing']})"
arms=list(r['panels']);roles=list(r['panels'][arms[0]])
for arm in arms:
    for p in r['panels'][arm].values():assert p['candidate_correct']+p['new_damage']+p['still_wrong']+p['missing']==p['queries']
parts.append('<p>终态：'+html.escape(r['status'])+'；各臂冻结DEV成功：'+html.escape(str(r['DEV_success_by_arm']))+'。</p>')
table(['角色','Base']+arms,[[role,str(r['panels'][arms[0]][role]['Base_correct'])+'/'+str(r['panels'][arms[0]][role]['queries'])]+[cell(r['panels'][a][role]) for a in arms] for role in roles])
parts.append('<h2>预登记配对对照</h2><p>B−A 检验更新幅度；C−B 检验幅度匹配后的方向限制；D−C 检验几何刷新；A−E 检验移除历史RLS。所有臂门控恒为1；不能据此分别归因历史门控和当前写入门控。差值界包含所有缺评分可能性，不是置信区间。</p>')
for name,p in r['registered_contrasts'].items():
    parts.append('<h3>'+html.escape(name+' : '+p['first']+' − '+p['second'])+'</h3>')
    table(['角色','全部题','双方有评分','胜','负','平','至少一方缺','正确数差界'],[[role,x['queries'],x['both_scored'],x['win'],x['loss'],x['tie'],x['at_least_one_missing'],x['correct_count_difference_bounds']] for role,x in p['panels'].items()])
parts.append('<h2>逐编辑及前缀</h2>')
table(['臂','编辑','源题','FIT','GFIT','T1G','T2G'],[[a,o['owner']]+[cell(o['panels'][k]) for k in ['NATIVE','FIT','GFIT','T1G','T2G']] for a,values in r['per_owner'].items() for o in values])
table(['臂','前缀','全题','正确','缺评分'],[[a,p['prefix'],p['queries'],p['correct'],p['missing']] for a,h in r['continual_history'].items() for p in h['prefixes']])
parts.append('<h2>训练诊断</h2>')
table(['臂','编辑','CE 首','CE 末','KL 末','最大实际更新范数'],[[a,e['owner'],e['weighted_CE_first'],e['weighted_CE_last'],e['BASIS_KL_last'],e['maximum_step_norm']] for a,x in r['training_diagnostics'].items() for e in x['by_edit']])
parts.append('<p>每编辑第1/40/80/160步的真实梯度能量、RLS衰减及更新对CE/KL的点积见 RESULTS.json；它们用于解释行为对照，不代替自由生成正确性。</p>')
parts.append('<h2>两轮历史参照</h2>')
history=dict(r['historical_round_one'],**r['historical_round_two'])
table(['角色']+list(history),[[role]+[cell(x[role]) for x in history.values()] for role in roles])
parts.append('<p>旧轮失败评分保持缺失，仅完全绑定有效分数复用。跨轮参照不是随机独立临床验证。调用与资源账本：'+html.escape(json.dumps({'payload_status':r['payload_status'],'inherited':r['inherited_payload_status'],'missing_consumers':r['missing_consumers'],'resource':r['resource'],'deletion':r['deletion']}))+'</p>')
(out/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pure W 五臂</title><style>body{max-width:1200px;margin:30px auto;padding:20px;font:16px/1.7 system-ui;color:#17202c}table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #ddd;padding:7px;white-space:nowrap}th{background:#eef2f7}</style><body>'+''.join(parts)+'</body></html>')
print('PASS: complete denominators; report rendered')
