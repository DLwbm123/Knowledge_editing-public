import json,os,sys,time
from pathlib import Path
w=Path(os.environ['WORKTREE']);a=Path('/private/tmp/.r.20261102.05');b=Path('/private/tmp/.r.20261102.07');c=Path('/private/tmp/.r.20261102.08')
for p in (b,c):
 if json.loads((p/'GUARD_EXIT.json').read_text())['code']!=0 or json.loads((p/'WORKER_EXIT.json').read_text())['code']!=0:raise ValueError('guard or worker not complete')
old=json.loads((a/'DECISIONS.json').read_text());new=json.loads((c/'DECISIONS.json').read_text());lock=json.loads((c/'SUPPLEMENT_INPUT.private.json').read_text())
assert set(lock['payloads'])==set(new['decisions'])|set(new['unknown'])
assert not (set(old['decisions'])&set(new['decisions']))
packet=json.loads((a/'PACKET_PRIVATE.json').read_text());roles=json.loads((w/'.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json').read_text())
sys.path.insert(0,str(w/'experiments/med_full_d150_v1'));import report
combined={**old['decisions'],**new['decisions']};unknown=set(old['unknown'])-set(new['decisions'])
score={'original_reused':len(old['decisions']),'first_pre_call_reservations':42,'first_actual_requests':0,'first_unsent_preflight':14,'recovery_reservations':json.loads((c/'COMPLETE.json').read_text())['new_attempt_items'],'recovery_valid':len(new['decisions']),'remaining_unknown':len(unknown),'history_coverage_complete':False}
r=report.run(packet,combined,roles,score);dest=w/'reports/med_d_u101_t2g_supplement_v1_20261102'
with (dest/'RESULTS.json').open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2)
with (dest/'EXECUTION_AUDIT.json').open('x') as f:json.dump({'scoring':score,'actual_request_caveat':'first 42 reservations failed with Python TypeError before entering run_batch, so no request sent; preserved as unknown and not retried','first_error':'incompatible run_batch signature; two batches proxy preflight failed','recovery':'only 14 preflight-unreserved payloads, same deadline and corrected historical scorer','no_training_generation':True,'model':'gpt-6-astra','reasoning':'medium','snapshot':'unknown'},f,ensure_ascii=False,indent=2)
x=r['panels']['all']['T2G']['core_correct'];y=r['panels']['all']['T2L']['core_correct'];
with (dest/'REPORT_ZH.md').open('x') as f:
 f.write('# D-U101 T2G 定向补评\n\n')
 f.write(f"固定 571 个 T2G 事件，原 MED CORE 边界 {x['MED']['bounds'][0]:.4f}–{x['MED']['bounds'][1]:.4f}%，D-U 边界 {x['MED_D']['bounds'][0]:.4f}–{x['MED_D']['bounds'][1]:.4f}%，D-U 未知 {x['MED_D']['unknown']}；已知修复 {x['gained']}、损伤 {x['lost']}、未知配对 {x['unknown_pairs']}。\n\n")
 f.write(f"T2L CORE 原 MED {y['MED']['bounds'][0]:.4f}%，D-U 边界 {y['MED_D']['bounds'][0]:.4f}–{y['MED_D']['bounds'][1]:.4f}%：超过允许的下降 5 个百分点，原 DEV_FAIL 不改变。\n\n")
 f.write(f"56 个 T2G 唯一未知载荷中，42 项首次已登记但接口不兼容而在真正调用前失败，保留为未知、不重试；14 项代理预检未发出，修正接口后单次尝试，取得 {len(new['decisions'])} 个有效评分。其余未知未改；一项历史冲突保持未知。既有 1413 个有效载荷不重评。历史账本不完整，不能排除未留存的历史重复。\n\n")
 f.write('结果为已曝光开发集；101 个 D-U 专家、49 个原 MED 回退。完整角色分母、未知边界及评分审计见随附文件；原 D 对照及最终比较按原依赖继续。\n')
(c/'REPORT_COMPLETE.json').write_text(json.dumps({'status':r['status'],'unknown':len(unknown)}))
print(json.dumps({'status':r['status'],'t2g':x,'recovery':score}))
