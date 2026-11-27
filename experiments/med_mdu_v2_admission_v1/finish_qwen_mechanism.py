"""Score frozen Qwen mechanism answers and publish image-grouped aggregate results."""
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
import time

from follow_pilot import CODE, publish, remote, wait_for
from reference_scores import prepare_rows


def regression(worktree, panel):
    previous = worktree / 'experiments/med_mdu_family_diagnostic_v1/readout_associations.py'
    functions = previous.read_text().split('\ndef describe(', 1)[0]
    program = functions + '\npanel=' + repr(panel) + """
if panel:
    roles = sorted({x[1] for x in panel})
    design = np.asarray([[1, math.log(r), *[role == k for k in roles[1:]]]
                         for _, role, r, _ in panel], dtype=float)
    outcome = np.asarray([x[3] for x in panel], dtype=float)
    owner = np.asarray([x[0] for x in panel])
    try:
        beta, _ = logistic(design, outcome)
        [draws] = bootstrap([design], [outcome], owner)
        result = dict(log_readout=float(beta[1]), bootstrap_count=len(draws),
                      ci95=interval(draws)[1] if len(draws)>=100 else None,
                      events=len(panel), groups=len(np.unique(owner)), reference=roles[0])
    except (ValueError, np.linalg.LinAlgError):
        result = dict(status='LOGISTIC_FIT_UNAVAILABLE', events=len(panel))
else:
    result=dict(status='NO_SCORABLE_EVENTS',events=0)
print(json.dumps(result,allow_nan=False))
"""
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'pro5000', '/data/bmw/envs/v0/bin/python', '-'],
                          input=program, capture_output=True, text=True, check=True, timeout=180)
    return json.loads(proc.stdout)


def run(worktree, server, local, queue_end):
    start = json.loads(wait_for(server, ('start.json',), queue_end)['start.json'])
    deadline = start['deadline']
    closed = wait_for(server, ('worker_exit.json',), deadline)
    assert json.loads(closed['worker_exit.json'])['codes'] == [0, 0]
    names = ('done0.json', 'done1.json', 'metrics0.jsonl', 'metrics1.jsonl')
    packet = remote(server, names)
    assert set(packet) == set(names)
    local.mkdir()
    for name, content in packet.items():
        (local / name).write_text(content)
    blocks = [json.loads(line) for n in (0, 1)
              for line in packet[f'metrics{n}.jsonl'].splitlines()]
    assert len(blocks) == 27 and len({b['owner'] for b in blocks}) == 27
    assert sum(json.loads(packet[f'done{i}.json'])['updates'] for i in (0, 1)) == 4320
    by_role = defaultdict(list)
    events = []
    for b in blocks:
        values = b['measurements']
        fits = [v['projection'] for v in values if v['role'] == 'fit']
        assert len(fits) == 4 and median(fits) > 0
        for v in values:
            row = dict(v, owner=b['owner'], ratio=v['projection'] / median(fits))
            by_role[row['role']].append(row)
            if 'answer' in v:
                events.append(dict(row, answers={'Q': v['answer']}))
    score = Path(tempfile.mkdtemp(prefix='.r.', dir='/private/tmp'))
    prepare_rows(events, score, ('Q',))
    count = len(json.loads((score / 'SUPPLEMENT_INPUT.private.json').read_text())['payloads'])
    env = dict(os.environ, FIXED_SCORE_ROOT=str(score), FIXED_WORKTREE=str(worktree),
               FIXED_JUDGE_SOURCE=str(worktree / 'experiments/medtrace_full_method_comparison_v2_20261003/execution_source'),
               FIXED_CLI='/opt/homebrew/bin/codex', FIXED_ITEM_COUNT=str(count),
               FIXED_SCORING_EPOCH='QWEN_MECHANISM_20261127', EXPERIMENT_DEADLINE=str(deadline))
    with (worktree / 'experiments/med_mdu150_t2g_supplement_v1/score.py').open() as source, (score / 'log').open('x') as out:
        subprocess.run([sys.executable, '-'], stdin=source, stdout=out, stderr=subprocess.STDOUT,
                       cwd=score, env=env, check=True, timeout=max(1, deadline - time.time()))
    for name in ('COMPLETE.json', 'DECISIONS.json', 'consumers.private.json'):
        (local / name).write_text((score / name).read_text())
    decisions = json.loads((score / 'DECISIONS.json').read_text())['decisions']
    mapped = json.loads((score / 'consumers.private.json').read_text())
    for i, m in enumerate(mapped):
        assert m['event'] == i and m['owner'] == events[i]['owner']
        events[i]['core'] = decisions.get(m['key'], {}).get('core_correct')
    scored_roles = defaultdict(list)
    for v in events:
        scored_roles[v['role']].append(v['core'])
    summary = {}
    for role, rows in by_role.items():
        observed = scored_roles[role]
        summary[role] = dict(events=len(rows), core=sum(v is True for v in observed) if observed else None,
                             unknown=sum(v is None for v in observed),
                             **{name: median(r[name] for r in rows)
                                for name in ('ratio', 'projection', 'first_nll', 'nll', 'residual', 'kl')})
    heldout = by_role['holdout_modal'] + by_role['holdout_context']
    gap = median(r['ratio'] for r in heldout) - median(r['ratio'] for r in by_role['independent'])
    steps = [(r['owner'], r['role'], r['ratio'], int(r['core'])) for r in events
             if r['role'].startswith('official_') and r['ratio'] > 0 and r['core'] is not None]
    result = dict(groups=summary, owner_count=27, updates=4320, readout_margin=gap,
                  official_logistic=regression(worktree, steps), judge_snapshot=None)
    report = ['# MDU-v2 第二骨干机制复现', '',
              '27 个 Base 错题以普通 CE 各训练 160 步。读入分母为 Qwen 自身四条 FIT 的读入中位数。'
              '教师强制读取最后提示位，CORE 评分沿用原合同。', '',
              '| 问法 | 事件 | CORE 正确 | 未知 | 读入比 | 整答 NLL | KL |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for role, s in summary.items():
        core = '—' if s['core'] is None else str(s['core'])
        report.append(f"| {role} | {s['events']} | {core} | {s['unknown']} | {s['ratio']:.4f} | {s['nll']:.4g} | {s['kl']:.4g} |")
    report.extend(['', f'留出正负例选择性间隔为 {gap:.4f}。官方 T2G 控制问法组的读入系数、'
                   '图像聚类 bootstrap 及各分组残差见结果 JSON。开发数据已有第一骨干的实验曝光。', ''])
    (local / 'summary.json').write_text(json.dumps(result, allow_nan=False))
    key = publish(worktree, 'MDU_V2_QWEN_MECHANISM', '\n'.join(report), result)
    (local / 'commit').write_text(key)


if __name__ == '__main__':
    config = json.loads(Path('config.json').read_text())
    run(Path(config['worktree']), Path(config['server']), Path(config['local']), config['queue_end'])
