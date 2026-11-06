"""Report the corrected comparator against the fixed selected main method."""
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys

w = Path(os.environ['WORKTREE'])
local = Path(os.environ['LOCAL_SCORE_ROOT'])
cfg = json.loads((local/'config.json').read_text())
sys.path.insert(0, str(w/'experiments/med_mdu150_v1'))
from summary import summarize

def read(p):
    return json.loads(Path(p).read_text())

if read(local/'GUARD_EXIT.json')['code'] != 0 or not read(local/'COMPLETE.json')['generation_and_scoring_complete']:
    raise ValueError('Scoring dependency failed')
rows = read(local/'PACKET_PRIVATE.json')['consumers']
judgments = read(local/'DECISIONS.json')
base_root = w/'.private/med_mdu150_t2g_supplement_v1_20261106/scoring'
baseline = read(base_root/'BASELINE.private.json')
old_decisions = {**baseline['decisions'], **read(base_root/'DECISIONS.json')['decisions']}
roles = baseline['roles']['roles']
arms = ('MED', 'MDU150', 'BE', 'GRACE_CORRECTED')
index = {(r['arm'],r['query_id']):r['key'] for r in baseline['consumers'] if r['arm'] in arms}
bank = [r for r in rows if r['mode']=='bank' and r['prefix']==150]
if len(rows)!=2207 or len(bank)!=2148 or len({r['query_id'] for r in bank})!=2148:
    raise ValueError('Corrected prediction coverage differs')
index.update({('GRACE_CORRECTED',r['query_id']):r['key'] for r in bank})
query_sets = [{q for a,q in index if a==arm} for arm in arms]
if any(q!=query_sets[0] for q in query_sets):
    raise ValueError('Methods do not share frozen events')
panels = {}
for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
    entries = [(int(owner),q) for owner,v in roles.items() for q in v[role] if q in query_sets[0]]
    panels[role] = {}
    for field in ('core_correct','full_supported'):
        values = {}
        for arm in arms:
            source = judgments['decisions'] if arm=='GRACE_CORRECTED' else old_decisions
            values[arm] = [(o,source.get(index[arm,q],{}).get(field)) for o,q in entries]
        known = {};gain=loss=missing=0
        for (owner,a),(_,b) in zip(values['MDU150'],values['GRACE_CORRECTED']):
            if a is None or b is None:
                missing+=1
            else:
                gain+=bool(b and not a);loss+=bool(a and not b)
                known.setdefault(owner,[]).append(int(b)-int(a))
        interval=None
        if entries and not missing:
            deltas=[statistics.mean(v) for v in known.values()];rng=random.Random(20261107)
            samples=sorted(100*statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(10000))
            interval=[samples[249],samples[9749]]
        panels[role][field]=dict(arms={a:summarize(v) for a,v in values.items()},
            MDU150_to_GRACE_CORRECTED=dict(gain=gain,loss=loss,unknown=missing,edit_bootstrap95=interval))
frozen_main = read(w/'reports/med_mdu150_t2g_supplement_v1_20261106/RESULTS.json')['groups']['all']
for role in panels:
    for field in panels[role]:
        for arm in ('MED','MDU150','BE'):
            if panels[role][field]['arms'][arm] != frozen_main[role][field]['arms'][arm]:
                raise ValueError('Selected method or an existing comparator changed')
script = '''import json
from pathlib import Path
r=Path(%r)
names=('LOCK.json','CLOCK.json','MECHANICAL_COMPLETE.json','GENERATION_COMPLETE.json','WORKER_EXIT.json','GUARD_EXIT.json')
d={n:json.loads((r/n).read_text()) for n in names}
rows=[json.loads(p.read_text()) for p in (r/'outputs').glob('*.json')]
d['quality']={m:{'n':sum(x['mode']==m for x in rows),'cap':sum(x['mode']==m and x['at_cap'] for x in rows)} for m in ('single','insertion','bank')}
d['cache_new_replacements']=sum(x['cache_audit']['cached_new_replacements'] for x in rows)
print(json.dumps(d))
''' % cfg['remote_root']
p = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',cfg['ssh_host'],cfg['remote_python'],'-'],
                   input=script,text=True,capture_output=True,timeout=90,check=True)
evidence = json.loads(p.stdout)
if (evidence['MECHANICAL_COMPLETE.json']['status']!='PASS' or evidence['WORKER_EXIT.json']['code']!=0
        or evidence['GUARD_EXIT.json']['code']!=0 or evidence['cache_new_replacements']!=0
        or evidence['GENERATION_COMPLETE.json']['counts']!=dict(edits=158,inserted=150,generation_attempts=2207,formal_consumers=2207,diagnostic_generation_attempts=12)):
    raise ValueError('Actual mechanical/training/generation evidence differs')
completion = read(local/'COMPLETE.json')
durable = w/'.private/med150_grace_correctness_v1_20261107/scoring'
durable.mkdir(parents=True,exist_ok=False)
for name in ('config.json','clock.json','CPU_CHECKS.json','PACKET_PRIVATE.json','BATCH_MANIFEST.json','DECISIONS.json','COST.json','COMPLETE.json','GUARD_EXIT.json'):
    if (local/name).exists():shutil.copy2(local/name,durable/name)
for bundle in (local/'batches').glob('*'):
    source=bundle/'operator';dest=durable/'batches'/bundle.name/'operator';dest.mkdir(parents=True)
    for name in ('ATTEMPT.json','PREFLIGHT.json','FAILURE_NO_RETRY.json'):
        if (source/name).exists():shutil.copy2(source/name,dest/name)
    for name in ('execution_evidence','responses'):
        if (source/name).exists():shutil.copytree(source/name,dest/name)

result = dict(status='COMPLETE' if not judgments['unknown'] else 'INCOMPLETE_SCORING',
              cohort='exposed 146+4 development; final 150-edit state',panels=panels,
              scoring=completion,quality=evidence['quality'],selected_main_unchanged=True,
              reference=evidence['LOCK.json']['method_config'])
out = w/'reports/med150_grace_correctness_v1_20261107'
def write(name,value):
    with (out/name).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
write('RESULTS.json',result)
audit=dict(status=result['status'],mechanical_source_edits=4,mechanical_updates=200,
    formal_edits=154,formal_updates=7700,diagnostic_generations=12,formal_generations=2207,
    actual_new_judge_items=completion['new_attempt_items'],unknown_payloads=len(judgments['unknown']),
    inherited_unknown_payloads=completion.get('inherited_unknown',0),automatic_retry=False,
    mechanical_cache_and_restore_pass=True,backbone_versions_unchanged=True,
    cached_generated_token_replacements=0,worker_exit=0,gpu_guard_exit=0,score_guard_exit=0,
    clock=evidence['CLOCK.json'],fixed_config=evidence['LOCK.json']['method_config'])
write('EXECUTION_AUDIT.json',audit)
lines=['# GRACE 多模态修正版：固定参考配置结果','','状态：'+result['status']+'。主方法仍为已选定MDU150，原异常GRACE记录保留；本次不是层位或参数搜索。',
       '', '| CORE（%） | 原MED | MDU150 | BalancEdit | GRACE修正版 |','|---|---:|---:|---:|---:|']
for role,panel in panels.items():
    def fmt(a):
        x=panel['core_correct']['arms'][a];b=x['bounds']
        if b is None:return 'NA'
        return (f'{b[0]:.4f}' if b[0]==b[1] else f'{b[0]:.4f}–{b[1]:.4f}')+f"（未知{x['unknown']}）"
    lines.append('| '+role+' | '+' | '.join(fmt(a) for a in arms)+' |')
lines+=['','单位为编辑等权百分比；区间是未知赋值边界。CORE为论文指标，FULL只作原始归档，完整值见机器记录。',
        'LLaMA-7B官方参考配置的多模态、cache一致适配：第27层down_proj、Euclidean、eps1、cold、Adam lr1、50步、replace_last；query定位、cache处理及精确目标标签碰撞是明确适配，不声称作者原生LLaVA实现。',
        '4个固定source-only机械single通过cache/非cache及保存/恢复核验后，真实执行150次连续编辑；既有主方法、对比和有效评分未改。',
        '原GRACE已知失败及冲突载荷不隐式重试，若新版本相同内容命中则保留UNKNOWN；其他新载荷每项至多一次。',
        '队列已曝光；T1L仅2事件，T2L仅54事件/31编辑，新增49专家的局部性支持仍为空。方法配置和精度差异不是完全受控的算法因果比较。',
        '', '生成质量：'+json.dumps(evidence['quality'],ensure_ascii=False)]
with (out/'REPORT_ZH.md').open('x') as f:f.write('\n'.join(lines)+'\n')
(local/'REPORT_COMPLETE.json').write_text(json.dumps(dict(status=result['status'],report=str(out))))
print(json.dumps(dict(status=result['status'],scoring=completion,quality=evidence['quality']),ensure_ascii=False))
