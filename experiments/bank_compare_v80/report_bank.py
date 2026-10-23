"""Publish only anonymous aggregate panels; raw answers and query IDs remain private."""
import collections,copy,statistics
from score_bank import *

def report():
 r=summarize();pub=copy.deepcopy(r);qids=sorted({x['query'] for x in read(S/'MAPPING.json')});qid={q:f'Q{i:03}' for i,q in enumerate(qids,1)}
 for x in pub['retention']:
  for d in x['lost_details']:d['query']=qid[d['query']]
 for x in pub['cost']:
  for d in x['latency']:d['query']=qid[d['query']]
 tasks=read(R/'DATA_ADMISSION.json')['tasks'];audit=[];tt_seconds=[];be_seconds=[]
 for t in tasks:
  tt=read(R/'TT_V69'/t['key']/'TRAIN.json');be=read(R/'BALANCEDIT'/t['key']/'BALANCEDIT_TRAIN.json');assert tt['status']==('RECOVERED_SAVED_TERMINAL_STATE_PENDING_BANK_EXACT' if t['order']==1 else 'PASS') and be['status']=='FROZEN';assert len(tt['curve'])==160 and len(be['curve'])==50
  assert (tt['cache_misses'],tt['cache_hits'])==(3,317);assert (be['cache_misses'],be['cache_hits'])==(1,49);assert be['save_load_exact'];assert tt.get('save_load_exact') or tt['stored_state_to_loaded_module_exact']
  tt_seconds.append(tt.get('train_seconds'));be_seconds.append(be['train_seconds_with_anchors_and_save']);audit.append(dict(owner=t['order'],TT_steps=160,BE_steps=50,TT_save_load='STORED_STATE_TO_NEW_MODULE' if t['order']==1 else 'MEMORY_TO_SAVE_LOAD',BE_save_load=True,TT_training_seconds=tt.get('train_seconds'),BE_training_anchor_save_seconds=be['train_seconds_with_anchors_and_save'],TT_curve_provenance='PRE_FAILURE_ASSERT_PASSED_THEN_REFERENCE_COPY' if t['order']==1 else 'NEW_EXACT_TRAJECTORY'))
 rows=read(S/'MAPPING.json');by={(x['prefix'],x['condition'],x['arm'],x['owner'],x['query']):x for x in rows}
 for arm in ('TT_V69','BALANCEDIT'):
  for x in rows:
   if x['condition']=='DESIGNATED' and x['arm']==arm and x['prefix'] in (1,5):assert x['answer']==by[10,'DESIGNATED',arm,x['owner'],x['query']]['answer']
 pub['training_audit']=audit;pub['cost_limits']=dict(wall_budget_seconds=43200,actual_first_start_to_final_exit=read(R/'EXIT.json')['time']-read(R/'WINDOW.json')['start'],all_attempt_process_seconds=r['execution']['seconds'],prior_initialization_cost_not_zero=True,TT_first_edit_training_seconds=None,TT_other9_seconds=sum(x for x in tt_seconds if x is not None),BE10_seconds_including_anchors_save=sum(be_seconds))
 route=read(R/'ROUTES_FROZEN.json')['prefixes'];pub['routing']={n:dict(unique_queries=len(d),rejected=sum(x['logical_edit_id'] is None for x in d.values()),activated=sum(x['logical_edit_id'] is not None for x in d.values())) for n,d in route.items()}
 pub['retrieval_cost']=dict(additional_training_updates=0,additional_generations_for_activation=0,uses_Base_query_key=True,shares_BE_source_router_anchors=True,standalone_end_to_end_latency='NOT_MEASURED',key_seconds_sum=sum(read(R/'KEY_COST.json')['seconds'].values()),Base_fallback_possible=True,measured_rejections=pub['routing'])
 out=Path('reports/bank_compare_v80_20261023');write(out/'BANK_RESULTS.json',pub)
 lines=['# V80：10编辑真实 bank 结果','', '10编辑自然路由下，TT T2G为37/40，BalancEdit与同R0检索均为40/40；TT未建立性能优势。TT专家参数存储显著更小，但本轮端到端延迟没有优势证据。两种方法仍破坏Base唯一原正确局部性答案，原V69联合门槛FAIL保持。所有数据均为DEV。','', '## 完整角色结果','', '每格为完整回答正确数/消费者数；T1L/T2L与其他角色可能重叠，不应相加当独立样本。自然请求183个在10编辑下映射到219个owner消费者；复用同一真实回答，不伪造219次独立请求。','', '|库容量|模式|方法|T0|T1G|T2G|T1L|T2L|改写|Base原正确局部性丢失|','|---|---|---|---|---|---|---|---|---|---|']
 for p in pub['panels']:
  cells=[f"{p['table'][role]['correct']}/{p['table'][role]['total']}" for role in ('T0','T1G','T2G','T1L','T2L','REPHRASE_DIAGNOSTIC')];lines.append('|'+ '|'.join(map(str,[p['prefix'],p['condition'],p['arm'],*cells,p['Base_correct_locality_losses']]))+'|')
 lines += ['', '## 库扩展与路由','', '固定1→5、1→10、5→10三个已观察节点：所有方法此前正确的T0/T1G/T2G和改写均保留。TT的T2G分别保留4/4、4/4、19/19，BE与检索为4/4、4/4、20/20。1→10和5→10均有5条旧正确T1L丢失（三方法相同），对应E001→E008路由变化；T1L分别保留14/19和16/21。1→5的T1L保留19/19。不能把这些端点结果解释为逐条插入瞬间的观测。','', '10编辑T1G自然38/40，而指定40/40；这两条损失属于新加入编辑的自然选专家问题，不与“旧T1G保留”矛盾。两方法自然路由相同，219消费者中166次选择指定owner、53次选择其他owner；prefix1有1/50请求被R0拒绝并回退Base；prefix5与10均无拒绝。因此这些有限观测不支持可靠拒绝能力或临床保护。不同owner选择不自动等同回答错误，完整答案另评分。','', '指定专家在共同输入跨库规模的答案保持，全部历史指定token逐一精确检查通过；BE新补3条T1G也正确。两种方法各自有独立冻结专家，没有共享参数连续覆盖；指定专家保持并不是强抗遗忘证据。','', '同R0直接返回所选源答案的非oracle检索与BE在本轮所有自然角色正确数相同，不代表逐字输出或所有C/I/U/A相同。它没有读评测owner或参考答案来路由。该对照提示本开发集的部分成绩可由源答案检索获得，不能单凭这组正确率证明参数编辑必要性。','', '## 完整答案与source scope','', '2298个消费者绑定280个完整(question, reference, answer)三元组：257精确复用/完整参考相等，23个全文混盲新评后冻结再汇总。没有关键词命中式计分、外部Judge调用或临床复核。C/I/U/A完整分布及逐编辑/逐角色配对在BANK_RESULTS.json。scope共875消费者、77四元组，28精确旧标签、49全文新评。','', '10编辑指定source未支持断言TT36/89、BE41/89；自然TT9/89、BE12/89、检索12/89。此项以该消费者原指定源为参照：错误路由引入别的疾病也可能不计入此源复制指标，所以自然scope数下降不是医学安全改善。参考未支持也不等同临床假阳性；需联合完整答案与Base原正确损伤阅读。','', '## 资源与运行证据','', '|方法|容量|专家FP32 bytes|常驻 allocated GiB|峰值 allocated GiB|峰值 reserved GiB|三请求秒数（排除warmup）|','|---|---|---|---|---|---|---|']
 for c in pub['cost']:
  lat=', '.join(f"{x['seconds']:.4f} ({x['tokens']} tokens)" for x in c['latency'] if not x['warmup']);lines.append(f"|{c['arm']}|{c['prefix']}|{c['resident_parameter_bytes']}|{c['resident_allocated']/2**30:.3f}|{c['peak_allocated']/2**30:.3f}|{c['peak_reserved']/2**30:.3f}|{lat}|")
 lines += ['', 'TT每编辑7168参数/28672bytes，BE58720256参数/234881024bytes，专家参数比8192倍；10专家TT0.273MiB，BE2.188GiB。系统仍加载约28.28GiB Base，不能把专家比值当系统显存或速度比。每容量仅3个预先固定请求、1次warmup；计时包含Base key、R0、视觉预处理、greedy及写盘，输出长度有差异。不能据此声称稳定生产延迟。', '', f"本轮实际2100更新、3880反向、17441语言forward、836生成、1155视觉forward；无新初始化反向。两次worker累计{r['execution']['seconds']:.3f}秒，失败尝试已计入。TT前一编辑单独训练耗时缺失，不能补造；其余9个TT耗时合计{pub['cost_limits']['TT_other9_seconds']:.3f}秒，BE10个含锚点与保存合计{sum(be_seconds):.3f}秒，计时范围不同不作严格训练加速比。旧初始化、历史开发与本次第一阶段统一评分的成本均保留，不宣称端到端零初始化。", '', '检索激活分支无新增训练或生成，但仍使用Base查询key与共享源锚点；prefix1存在1次Base回退，prefix5/10均激活；独立检索系统端到端延迟未测。BANK_RESULTS.json保留实际key提取成本与边界，不将无生成等同零系统成本。', '', '首个TT保存后发生CUDA/CPU状态比较故障：原160步曲线检查已通过，保存终态复用而未重训；参考曲线复制的来源明确标注，单次耗时NA。随后state到新模块精确装载及全部bank历史token检查通过。其余9个TT与10个BE曲线精确复现；cache分别3miss/317hit和1miss/49hit。原失败与266.544秒账保留，原12小时时钟未重置。详见RECOVERY1_ZH.md。', '', '所有6个bank消费者、全部延迟与评分已完成；EXIT0，guard确认全部登记进程退出。完整代码、匿名汇总公开；医学原始问答、token与私有路径不公开。checkpoint按明确清单在所有依赖完成后退休，回执另列。无新增LoRA、146扩展、阈值调参或后继实验；此次有限授权链完成后关闭监测。']
 if (out/'RETIREMENT.json').exists():
  retired=read(out/'RETIREMENT.json');lines += ['', f"生命周期已执行：20份本轮正式权重及ROUTE_KEYS，共{retired['count']}文件、{retired['bytes']:,} bytes已退休；无保留副本，需要重训才能重建。原始初始化、问答/token、冻结标签、配置与故障账保留。见RETIREMENT.json。"]
 (out/'BANK_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
 return pub
if __name__=='__main__':report()
