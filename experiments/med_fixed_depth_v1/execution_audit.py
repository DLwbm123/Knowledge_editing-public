"""Publish aggregate execution counts from completed dependency receipts."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(os.environ['FIXED_SCORE_ROOT'])
WORKTREE = Path(os.environ['FIXED_WORKTREE'])


def main():
    script = '''from pathlib import Path
import json
r=Path('/data/bmw/Knowledge_editing/outputs/med-fixed-depth-v1-20261030/run')
def read(n):return json.loads((r/n).read_text())
first=read('FIRST_PAIR_TRAIN_COMPLETE.json');failed=read('ENGINEERING_RECOVERY2.json');first_eval=read('FIRST_PAIR_EVAL_COMPLETE.json')
train=[read(f'BATCH_COMPLETE_{i}.json') for i in range(4)];ev=[read(f'BATCH_EVAL_COMPLETE_{i}.json') for i in range(4)]
assert read('WORKER_EXIT.json')['code']==read('GUARD_EXIT.json')['code']==0
assert read('FIRST_EVAL_EXIT.json')['code']==read('FIRST_EVAL_GUARD_EXIT.json')['code']==0
assert read('BATCH_GUARD_EXIT.json')['codes']==read('BATCH_EVAL_GUARD_EXIT.json')['codes']==[0,0,0,0]
assert all(read(f'BATCH_EXIT_{i}.json')['code']==read(f'BATCH_EVAL_EXIT_{i}.json')['code']==0 for i in range(4))
u=first['counts']['updates']+sum(x['counts']['updates'] for x in train);b=first['counts']['backwards']+sum(x['counts']['backwards'] for x in train)
pred=first_eval['counts']['predictions']+sum(x['counts']['predictions'] for x in ev);gen=first_eval['counts']['generations']+sum(x['counts']['generations'] for x in ev)
assert u==5120 and b==10240 and pred==1016 and failed['prior_counts']['updates']==160 and failed['prior_counts']['backwards']==320
print(json.dumps({'formal_training_updates':u,'formal_training_backwards':b,'failed_extra_updates':failed['prior_counts']['updates'],'failed_extra_backwards':failed['prior_counts']['backwards'],'first_preforward_failure_updates':0,'all_actual_updates':u+failed['prior_counts']['updates'],'all_actual_backwards':b+failed['prior_counts']['backwards'],'prediction_consumers':pred,'actual_generation_attempts':gen,'reused_off_predictions':pred-gen,'gpu_train_shards':4,'gpu_evaluation_shards':4,'all_worker_and_guard_exits_zero':True,'hard_end_utc':'2026-10-31T11:57:00Z','gpu_peak_memory':'NOT_MEASURED','storage_peak':'NOT_MEASURED'}))
'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'pro5000',
                             '/data/bmw/envs/v0/bin/python', '-'], input=script, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Remote execution receipts incomplete: ' + result.stderr[-1000:])
    audit = json.loads(result.stdout)
    score = json.loads((ROOT / 'COMPLETE.json').read_text())
    if json.loads((ROOT / 'WORKER_EXIT.json').read_text())['code'] != 0 or json.loads((ROOT / 'GUARD_EXIT.json').read_text())['code'] != 0:
        raise ValueError('Score exits not successful')
    audit['scoring'] = score
    audit['first_preload_error'] = 'inherited GPU UUID mismatch; no forward or update'
    audit['first_training_check_error'] = 'disabled-hook comparison used a helper that re-enabled the hook; saved state unavailable'
    audit['comparison_limitations'] = ['No original MED curvature in new branches', 'Fixed exposed DEV panel',
                                        'Historical accepted scores reused only by full scoring payload']
    out = WORKTREE / 'reports/med_fixed_depth_v1_20261030/EXECUTION_AUDIT.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'EXECUTION_RECEIPTS_AGGREGATED', 'actual_updates': audit['all_actual_updates'],
                      'predictions': audit['prediction_consumers']}))


if __name__ == '__main__':
    main()
