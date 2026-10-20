from pathlib import Path
import json,math,collections
r=Path(__file__).resolve().parents[2];o=r/'reports/vigor_controllerN3_v1_20261020/completion';n=json.loads((o/'RESULT.json').read_text());e=json.loads((o/'ENDED.json').read_text());l=json.loads((o/'LIFECYCLE.json').read_text());sc=json.loads((o/'SCORES.json').read_text());b=json.loads((r/'reports/vigor_controllerBC3_v1_20261020/completion/RESULT.json').read_text());bsc=json.loads((r/'reports/vigor_controllerBC3_v1_20261020/completion/SCORES.json').read_text())['scores'];checks=[]
def check(v,s):
 checks.append(dict(field=s,passed=bool(v)))
 if not v:raise ValueError(s)
for q in n['checks']:check(q.get('passed',q.get('absolute_difference',float('inf'))<=q.get('tolerance',0)),q['field'])
check(n['status']=='COMPLETE' and e['status']=='COMPLETE','terminal');check(all(n['restoration'].values()),'all_restoration');check(l['main_process_exited'] and l['gpu_process_exited'],'processes_exited');check(e['model_forward_calls']<=2400 and e['resident_seconds']<=1800,'global_budget');check(not sc['unscored'] and not sc['generation_missing'],'scores_complete')
rows=[]
for c,cb in zip(n['cases'],b['cases']):
 a=c['arms'][0];ab=next(x for x in cb['arms'] if x['name']=='B');accepted=[]
 for st in a['steps']:
  ats=[t for t in st['trials'] if t['status']=='ACCEPTED'];check(len(ats)<=1,'one_accept_per_step');check(len(st['trials'])<=3,'bounded_trials')
  for t in ats:
   accepted.append(t);g=t['final_gate'];check(g['accepted'] and all(g['gates'].values()),'accepted_original_gates')
   for v in ['actual_response','final_response']:
    rr=t[v];check(rr['NLL_gain']>1e-5 and rr['rho_NLL']>=.1 and rr['passed'],'accepted_NLL_response')
   check(t['cost']>0 and t['cost']<=1.01*5.368434395105091e-13*t['scale']**2,'native_cost');check(t['linear'][0]<-1e-5 and abs(t['predicted_original_base_null'])<=.01,'native_predictive_admission')
 check(len(accepted)==a['accepted_steps'],'accepted_step_count');check(a['base_restored'],'case_base_restored');check(a['editing_forwards']<=512,'arm_edit_budget')
 fm=a['last_accepted'];rr=dict(sample=c['sample'],B=dict(steps=ab['accepted_steps'],editing_forward=ab['editing_forwards'],backward=ab['backward_calls'],secants=ab['secant_calls'],NLL=ab['last_accepted']['native_nll'][0],margin=ab['last_accepted']['margin']['margin'],base_null_drift=ab['last_accepted']['native_nll'][1]-cb['base']['native_nll'][1],semantic_success=bsc[c['sample']+'_B_final'],endpoint_F0=ab['last_accepted']['cumulative_F0_cost']),N=dict(steps=a['accepted_steps'],status=a['status'],editing_forward=a['editing_forwards'],backward=a['backward_calls'],secants=a['secant_calls'],NLL=fm['native_nll'][0],margin=fm['margin']['margin'],base_null_drift=fm['native_nll'][1]-c['base']['native_nll'][1],semantic_success=sc['scores'][c['sample']+'_N_final'],endpoint_F0=fm['cumulative_F0_cost'],accepted_margin_nonpositive_gain=sum(t['final_response']['margin_gain']<=1e-5 for t in accepted),accepted_margin_rho_below_old_floor=sum(t['final_response'].get('rho_margin') is None or t['final_response']['rho_margin']<.1 for t in accepted),all_accepted_original_gates=True,final_generation=a['generations']['final'],generation_forwards=a['generation_forwards_total']))
 rows.append(rr)
old_audit=[]
for c in b['cases']:
 for a in c['arms']:
  pm=am=0;counts=collections.Counter()
  for st in a['steps']:
   for t in st['trials']:
    counts[t['status']]+=1;linear=t.get('linear');cost=t.get('cost')
    if t['status']=='NATIVE_ADMISSION_REJECTED' and linear and cost:
     if 0<cost<=1.01*5.368434395105091e-13*t['scale']**2 and linear[0]<-1e-5 and abs(t['predicted_original_base_null'])<=.01 and not linear[1]>1e-5:pm+=1
    if t['status']=='ACTUAL_RESPONSE_REJECTED':
     x=t['actual_response'];g=t['main_gate']['gates']
     if x['NLL_gain']>1e-5 and x['rho_NLL'] is not None and x['rho_NLL']>=.1 and all(g.values()):am+=1
  old_audit.append(dict(sample=c['sample'],arm=a['name'],statuses=dict(counts),only_margin_failed_at_local_predictive_admission=pm,main_actual_NLL_response_and_all_original_gates_pass_but_margin_response_failed=am,unexecuted_actual_checks_unknown=True,offline_only=True))
analysis=dict(rows=rows,semantic=dict(B=dict(success=1,denominator=3,missing=0),N=dict(success=sum(x['N']['semantic_success'] for x in rows),denominator=3,missing=0)),total_editing_forward={k:sum(x[k]['editing_forward'] for x in rows) for k in ['B','N']},total_backward={k:sum(x[k]['backward'] for x in rows) for k in ['B','N']},total_secant={k:sum(x[k]['secants'] for x in rows) for k in ['B','N']},global_N=dict(seconds=e['resident_seconds'],forward=e['model_forward_calls'],editing=sum(x['N']['editing_forward'] for x in rows),generation=sum(x['N']['generation_forwards'] for x in rows),control_and_restore=e['model_forward_calls']-sum(x['N']['editing_forward']+x['N']['generation_forwards'] for x in rows),per_arm_wall_seconds=None),comparison='same exposed cases and budget limits, unequal realized cost and progress',locality_valid_support=0,locality_retention=None)
(o/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n');(o/'AUDIT.json').write_text(json.dumps(dict(status='PASS',checks=checks,count=len(checks)),indent=2)+'\n');(o/'PRIOR_REFUSAL_AUDIT.json').write_text(json.dumps(old_audit,indent=2)+'\n')
lines=['# 三例NLL-only控制器消融：N与原B对照','','## 结论','','N与原B均为1/3语义成功，仅E03正确，缺评分0。移除指定字面token余量的硬条件确实允许更多更新和更低平均NLL，但本轮没有增加语义修复。不能继续把余量门当成主要失败解释，也不能据更低NLL宣布任务成功。保留原B作为简单基线；不因本轮失败自动加倍预算、追调半径或重判答案。','','## 逐例结果','','|例|B/N语义|B/N编辑forward|B/N接受步|B/N最终NLL|B/N原基座null漂移|原三门|','|---|---|---|---|---|---|---|']
for x in rows:
 B,N=x['B'],x['N'];lines.append(f"|{x['sample']}|{int(B['semantic_success'])}/ {int(N['semantic_success'])}|{B['editing_forward']}/{N['editing_forward']}|{B['steps']}/{N['steps']}|{B['NLL']:.9f}/{N['NLL']:.9f}|{B['base_null_drift']:+.9f}/{N['base_null_drift']:+.9f}|最终全部通过|")
lines+=['','B共661编辑forward/237backward/30secant；N共1350编辑forward/515backward/60secant。N编辑调用约为B的2.04倍，所有接受点原visual/null/trust通过；null只是固定代理，真实locality仍零有效支持NA。原F0/方向/原生提交及secant函数不变，移除margin准入/实际收益比/半径更新是本轮完整控制器消融，不能称仍遵守旧双收益标准。实际端点成本不同、每臂墙钟未记录NA，不据forward假称墙钟加速或等成本优势。','','E01/E03触512编辑forward限，分别接受40/53步；E02接受36步后无合法提案。最终生成104/66/8token均自然完成，原1024配置未降低、128forward上限未触。E02输出与旧正式评测同问题/参考/完整回答严格相同，复用旧失败评分；E01/E03两个新输入各一次盲评分。最多3payload实际2，无重试/缺分。E02尽管NLL降到1.034856677，完整回答仍复现旧失败输出，进一步提醒源teacher概率与自由生成任务不等价。','','## 旧候选离线拒绝原因审计','','六臂旧日志逐项复核：B-E02末端3个提案，在成本、NLL预测、null预测条件通过时被margin预测条件挡住；它们未执行实际原门，实际安全与收益仍UNKNOWN，不能算“原保护门全过”。B-E01/E03各有1个已测候选通过NLL实际响应和全部主原门却未通过margin响应；C对应1/6/3个。只能说明旧日志存在字面门阻塞，不能回写接受状态或推断改轨迹最终成功。新N提供直接轨迹证据，但三例语义结果不变。计数和互斥阶段定义见PRIOR_REFUSAL_AUDIT.json。','','## 对分享计划的修订','','“任务门消融→事实监督→真实保留”的问题拆分成立。当前N已完成，复用这次结果，无需重跑九条旧训练。不得在N运行结束后补16/32步生成或新面板；旧checkpoint未保留，中间语义NA保留。后续事实监督F必须独立冻结：只从编辑源问题/目标参考预先定义事实组，不能利用T1G/T2G/locality答案或编辑错误选token；额外语义标注成本单列。原平均NLL下降守卫、原null/trust/F0/secant不变，固定权重不搜索；F仍为teacher代理，不保证语义。','','真实locality准备需要先定义传播与保留边界，从已有授权数据中只依据Base资格固定候选顺序和筛选规则，不能看编辑后选有利题。保留实际有效分母、损伤和缺失界，局部零损伤不称全局保护。原B/N没有权重checkpoint，新增保护面板需要另冻重建成本，不虚称直接复用旧端点可评。开发推进标准必须在新轮前固定，旧三例暴露不作独立确认。','','## 审计、成本与交付','',f"GPU{e['resident_seconds']:.6f}/1800秒，总{e['model_forward_calls']}/2400forward=1350编辑+178生成+17控制恢复，峰值{e['peak_allocated_bytes']}字节。原运行586检查与本次{len(checks)}条独立核验PASS；各臂/全轮最初基座权重/NLL/dtype/flags/hooks/生成配置恢复，主/GPU进程退出。checkpoint0，未调用模型补缺失，未新下载数据或权重。",'',f"远端收集时逻辑{l['logical_bytes']}字节/du {l['du_kib']}KiB，之后另保存私有评分小证据；每处128MiB保守预算内，远低每服务器50GB。不是精确所有新增字节证明。2新评分一批，7359输入/109输出tokens、工具0/隔离检查见JUDGE_EVIDENCE，无新增购买。代码/所有标量/拒绝审计/预算/缺失/中文报告匿名公开；私有QA/回答/目标token身份/路径/梯度/权重不公开。旧正式八例与BC3历史结果不改写。"]
(o/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
(r/'reports/vigor_controllerN3_v1_20261020/STATE_ZH.md').write_text('# COMPLETE：N与原B均1/3\n\nN仅E03正确，三条评分齐全（2新+1严格缓存），原接受门全部通过，全部恢复/进程退出。1350编辑forward vs B661，287.557462秒/1545总forward。报告与审计见completion；等待本次公开回执。下一步先冻结源事实组/保留面板资格准备，不重跑旧臂、不追调余量/半径、不回填中间生成。持续自主监测仍ACTIVE。\n')
print(json.dumps(analysis,indent=2));print('AUDIT',len(checks))
