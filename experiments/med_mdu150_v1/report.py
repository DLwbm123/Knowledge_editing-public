"""Publish the frozen 150-expert paired DEV result after all guards succeed."""
import json,os,subprocess
from pathlib import Path
w=Path(os.environ['WORKTREE']);local=Path(os.environ['LOCAL_SCORE_ROOT']);remote=os.environ['REMOTE_ROOT'];neutral=os.environ['NEUTRAL_ROOT']
def read(p):return json.loads(Path(p).read_text())
for n in ('DRIVER_EXIT.json','GUARD_EXIT.json'):
 if read(local/n).get('code')!=0:raise ValueError('local scoring failed')
complete=read(local/'COMPLETE.json');result=read(local/'RESULTS.json')
if complete.get('status')!='SCORING_REUSE_COMPLETE' or complete.get('events')!=2148 or complete.get('new_judge_attempts')!=0:raise ValueError('scoring incomplete')
script="""import json\nfrom pathlib import Path\nr=Path(%r);q=Path(%r);read=lambda p:json.loads(p.read_text())\nfor n in ('TRAIN_GUARD_EXIT.json','EVAL_GUARD_EXIT.json'):\n if read(r/n).get('code')!=0:raise ValueError('GPU guard failed')\nfor n in ('CPU_EXIT.json','CPU_GUARD_EXIT.json'):\n if read(q/n).get('code')!=0:raise ValueError('CPU guard failed')\nt=[read(r/f'TRAIN_COMPLETE_{i}.json') for i in range(2)];e=[read(r/f'EVAL_COMPLETE_{i}.json') for i in range(2)];tf=read(r/'TRAIN_FROZEN.json');ef=read(r/'PREDICTIONS_FROZEN.json');clock=read(q/'CLOCK.json')\nprint(json.dumps({'train_counts':[x['counts'] for x in t],'eval_counts':[x['counts'] for x in e],'train_frozen':tf,'eval_frozen':ef,'clock':clock,'cpu_guard_exit':0,'train_guard_exit':0,'eval_guard_exit':0}))\n""" % (remote,neutral)
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','pro5000','python3','-'],input=script,text=True,capture_output=True,timeout=60)
if p.returncode:raise RuntimeError('remote execution audit failed')
audit=json.loads(p.stdout);t=audit['train_counts'];e=audit['eval_counts']
if sum(x['updates'] for x in t)!=7840 or sum(x['generations'] for x in e)!=635 or sum(x['reused'] for x in e)!=1513:raise ValueError('execution counts differ')
audit.update(status='COMPLETE_WITH_REUSED_SCORING',events=2148,new_judge_attempts=0,scoring_partition=result['scoring'],score_driver_exit=0,score_guard_exit=0,history_coverage_complete=False,resource_limit_note='30 GB polling stop and 1 GiB per-file limit, not an aggregate filesystem quota',test_scope='32 historical sealed questions exact normalized text and original146 DEV input check; not semantic deduplication or full historical test coverage')
out=w/'reports/med_mdu150_v1_20261105';out.mkdir(parents=True,exist_ok=True)
for name,obj in [('RESULTS.json',result),('EXECUTION_AUDIT.json',audit)]:
 with (out/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
lines=['# 完整150专家 MDU开发评估','','49位按原MDU从MED状态继续训练，各160步；原101位预测复用，49位实际路由事件重新生成。固定原R0，共2148事件；下表为编辑宏平均，未知区间为赋值边界，不是置信区间。','','| 角色 / 指标 | MED | MDU 101+49 | MDU 150 | BalancEdit |','|---|---:|---:|---:|---:|']
for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
 for field in ('core_correct','full_supported'):
  arms=result['groups']['all'][role][field]['arms']
  def fmt(k):
   x=arms[k];b=x['bounds'];return 'NA' if b is None else f"{b[0]:.4f}–{b[1]:.4f}%（未知{x['unknown']}，{x['events']}事件/{x['edits']}编辑）"
  lines.append('| '+role+' '+('CORE' if field=='core_correct' else 'FULL')+' | '+' | '.join(fmt(a) for a in ('MED','MDU101','MDU150','BE'))+' |')
lines+=['','配对修复/损失、路由及owner分层、未知和历史冲突敏感性见 RESULTS.json。评分只复用可核验的历史判定，未发出新Judge请求；历史账本不完整，未判定项保留UNKNOWN。','', '原面板为已曝光开发集；T0/FIT是训练拟合检查，T1L仅2项。有限测试检查和预登记范围见启动及准备报告；不同措辞的语义重叠及未列入的历史测试未知。BalancEdit的运行精度与MED系列不同。结果不自动替换原MED。']
with (out/'REPORT_ZH.md').open('x') as f:f.write('\n'.join(lines)+'\n')
(local/'REPORT_COMPLETE.json').write_text(json.dumps({'status':'REPORT_COMPLETE','events':2148}))
print('REPORT_COMPLETE')
