"""Sanitized fixed-prefix scoring; no medical text or private identifiers exported."""
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from freshstart.runtime import ROOT,read,write
from freshstart.pipeline import input_key
from freshstart.metrics import validate_rows,summarize
from scripts.medtrace.stage17_prepare import digest


def verdict(key):
    path=ROOT/'private/judge/scores'/f'{key}.json'
    return read(path)['is_correct'] if path.exists() else None


def generate(label='STATUS'):
    public=ROOT/'public';public.mkdir(exist_ok=True)
    base_path=ROOT/'private/preparation/BASE_OUTPUTS.json'
    bases=read(base_path) if base_path.exists() else {}
    table={};process=[];training=[]
    for path in ROOT.glob('formal*/private/consumers/*.json'):
        c=read(path);row=c['row'];run=path.parents[2].name
        base=bases.get(digest([input_key(row),row['reference']]),{})
        source=digest(['public-source',row['source_group']])[:20]
        identity=digest([input_key(row),row['reference'],row['role']])[:24]
        value=dict(arm=run+':'+c['arm'],prefix=c['prefix'],input_id=identity,source_group=source,role=row['role'],
            current_correct=verdict(c['judge_key']),base_correct=verdict(base['judge_key']) if base else None,
            strict_unrelated=None,teacher_agreement=verdict(c['teacher_judge_key']) if c.get('teacher_judge_key') else None)
        if c['event'].startswith('panel/'):
            key=(value['arm'],value['prefix'],value['input_id'])
            if key in table and table[key]!=value:raise ValueError('Conflicting fixed-prefix consumer')
            table[key]=value
        else:
            value['consumer_id']=c['consumer_id']
            (training if c['event'].startswith('training_diagnostic/') else process).append(value)
    rows=validate_rows(list(table.values()))
    for name,values in [('FIXED_PREFIX',rows),('PROCESS',process),('TRAINING_SUPPORT',training)]:
        (public/(name+'.jsonl')).write_text(''.join(json.dumps(v,sort_keys=True)+'\n' for v in values))
        if values:
            with (public/(name+'.csv')).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    write(public/'METRICS.json',summarize(rows))
    ledger=read(ROOT/'RESOURCE_LEDGER.json')
    safe_ledger={k:v for k,v in ledger.items() if k not in ['judge_attempts','gpu_sessions']}
    safe_ledger['gpu_active_resident_seconds']=sum(time.time()-s['started_epoch'] for s in ledger['gpu_sessions'] if s.get('ended_epoch') is None)
    write(public/'RESOURCE_LEDGER.json',safe_ledger)
    statuses={str(p.relative_to(ROOT)):read(p) for p in ROOT.glob('formal*/*_STATUS.json')}
    all_scores=len(list((ROOT/'private/judge/scores').glob('*.json')))
    pending=len(list((ROOT/'private/judge/pending').glob('*.json')))
    status=dict(label=label,at=datetime.now(timezone.utc).isoformat(),campaign=read(ROOT/'RUN_MANIFEST.json')['run_id'],
        frozen_N=16,training=statuses,fixed_prefix_consumers=len(rows),fixed_prefix_unscored=sum(r['current_correct'] is None for r in rows),
        score_tasks=pending,score_tasks_complete=all_scores,scoring_complete=pending==all_scores,
        confirmation='NO_INDEPENDENT_CONFIRMATION_AVAILABLE',patient_study_independence='UNKNOWN',
        missing_qualification_panels=['independent_rephrase','positive_image','strict_unrelated_eligibility'],
        method_claim='NO_VALIDATED_CANDIDATE',disk_free_bytes=shutil.disk_usage(ROOT).free,
        backup='OFFHOST_BACKUP_NOT_AVAILABLE',historical_results_modified=False)
    # Private event IDs include source identifiers; only counts belong in public statuses.
    for state in statuses.values():
        if 'completed_slots' in state:state['completed_slot_count']=len(state.pop('completed_slots'))
    write(public/(label+'.json'),status)
    text=(f'# FreshStart {label}\n\n实际冻结 N=16；不是 DEV45，也不是患者独立确认。\n\n'
          f'固定前缀消费者 {len(rows)}，未评分 {status["fixed_prefix_unscored"]}。'
          f'评分任务 {all_scores}/{pending}。\n\n'
          '基础模型、Base 输出、CP/A2/W0 和实验状态均为新生成。历史结果未合并。'
          '缺少独立改写、图像正向面板及严格 U 资格，不能完成计划的全部保护条件；不宣称方法验证成功。'
          '本轮仅 source-answer agreement；不声称独立影像专家判定。\n\n'
          '过程消费者、训练支持和固定前缀评价分文件；缺失值为 null。'
          '源组为匿名标识，未公开问题、参考、输出、图像、身份映射和权重。\n\n'
          '恢复状态仅保存在该服务器；没有已批准的私有异机备份目标。\n')
    (public/(label+'.md')).write_text(text)
    if label in ['MIDPOINT','FINAL']:
        (public/'ADVISOR_ZH.md').write_text(text+'\n研究结论边界：新环境小规模 DEV 对照，不能当作历史结果复现或独立临床泛化证据。\n')
        (public/'PAPER_METHOD_EXPERIMENTS.md').write_text(
            '# Methods and experiments draft\n\nFrozen LLaVA-Med v1.5 Mistral 7B with CLIP-L/336, FP16 backbone, FP32 rank-4 writer at zero-based L30 down projection. '
            'Initialization recomputes native CP, 80 A2 steps and 320 CP-W0 steps. All arms share numerical per-seed/edit initialization, in independent writable states. '
            'Writers use 320 Adam steps; native and rotating four-fit losses have weights 0.5 each, H has 0 or 0.25, and active full-vocabulary Base-to-student KL has weight 0.01. '
            'Replay arms use matched 20-step slots with current-prefix ownership; missing and masked teachers are distinct.\n\n'
            'The recovered dataset contains 16 exploratory source-label edits and 22 images. Patient/study independence is unknown. '
            'No independent confirmation cohort, heldout rephrase panel, positive-image panel or verified strict-unrelated panel was available. '
            'Consequently the full method-selection criterion cannot be satisfied; results are exploratory. See JSON/CSV for measured coverage and results.\n\n'
            'Eight-page outline: 1 problem and scope; 2 source data and privacy; 3 initialization; 4 role masking and routing; '
            '5 matched maintenance; 6 measured results; 7 recovery and cost; 8 limitations and claim-evidence boundary.\n')
        write(public/'CLAIM_EVIDENCE.json',dict(fresh_computation='runtime and event receipts',
            full_protocol_validation=False,independent_confirmation=False,clinical_generalization=False,historical_checkpoint_recovery=False))
    return status
