"""Complete the second-backbone base screen using the common frozen CORE rubric."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from follow_pilot import CODE, publish, remote
from prepare_qwen import prepare, qualify


def run(worktree, server, local):
    config = json.loads(remote(server, ('CONFIG.json',))['CONFIG.json'])
    deadline = config['hard_end']
    while time.time() < deadline:
        notes = remote(server, ('Q/COMPLETE.json', 'Q/FAILURE.json'))
        if 'Q/FAILURE.json' in notes:
            raise RuntimeError('Base generation failed')
        if 'Q/COMPLETE.json' in notes:
            break
        time.sleep(25)
    else:
        raise TimeoutError('Base run deadline elapsed')
    local.mkdir()
    run = local / 'raw'
    run.mkdir()
    (run / 'outputs').mkdir()
    remote_code = f"""
from pathlib import Path
import json
p=Path({str(server / 'Q/outputs')!r})
print(json.dumps({{x.name:x.read_text() for x in p.glob('*.json')}}))
"""
    transfer = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'pro5000', 'python3', '-'],
        input=remote_code, check=True, capture_output=True, text=True, timeout=40)
    rows = json.loads(transfer.stdout)
    assert len(rows) == 30
    for k, v in rows.items():
        assert k.endswith('.json') and k[:-5].isdigit()
        (run / 'outputs' / k).write_text(v)
    (run / 'COMPLETE.json').write_text(notes['Q/COMPLETE.json'])
    score = Path(tempfile.mkdtemp(prefix='.r.', dir='/private/tmp'))
    panel = local.parent / 'prepared_r2/stage1_panel.private.json'
    prepare(run, score, panel)
    inputs = json.loads((score / 'SUPPLEMENT_INPUT.private.json').read_text())
    n = len(inputs['payloads'])
    scorer = worktree / 'experiments/med_mdu150_t2g_supplement_v1/score.py'
    provider = worktree / 'experiments/medtrace_full_method_comparison_v2_20261003/execution_source'
    env = dict(os.environ, FIXED_SCORE_ROOT=str(score), FIXED_WORKTREE=str(worktree),
               FIXED_JUDGE_SOURCE=str(provider), FIXED_CLI='/opt/homebrew/bin/codex',
               FIXED_ITEM_COUNT=str(n), FIXED_SCORING_EPOCH='QWEN_BASE_QUALIFICATION_20261127',
               EXPERIMENT_DEADLINE=str(deadline))
    with scorer.open() as script, (score / 'log').open('x') as out:
        subprocess.run([sys.executable, '-'], cwd=score, env=env, stdin=script, stdout=out,
                       stderr=subprocess.STDOUT, timeout=max(1, deadline - time.time()), check=True)
    for filename in ('COMPLETE.json', 'DECISIONS.json', 'consumers.private.json'):
        (local / filename).write_text((score / filename).read_text())
    results = qualify(run, score, worktree, local)
    c = results['groups']
    scores = {k: c.get(k, 0) for k in ('core_correct', 'core_incorrect_admitted', 'unknown', 'invalid_generation')}
    assert sum(scores.values()) == 30
    summary = (
        '# MDU-v2 第二骨干 Base 资格结果\n\n'
        'Qwen2.5-VL-7B-Instruct 用原生 BF16、SDPA 和固定图像预处理在 GPU 7 上执行同 30 个原问。'
        '评分沿用共同的 CORE 合同与 gpt-6-astra medium，供应方未提供固定快照。\n\n'
        '| 原问状态 | 数量 |\n|---|---:|\n'
        f"| CORE 正确 | {scores['core_correct']} |\n"
        f"| CORE 错误且生成有效，准入 | {scores['core_incorrect_admitted']} |\n"
        f"| 评分未知 | {scores['unknown']} |\n"
        f"| 空回答、触顶或未到 EOS | {scores['invalid_generation']} |\n\n"
        '本轮完成 30 次生成、0 权重更新。合格来源另绑定既有 T2G 分型，'
        '只用于后续跨骨干机制实验。图像来源已用于第一骨干的开发实验。\n')
    (local / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, allow_nan=False))
    ref = publish(worktree, 'MDU_V2_QWEN_BASE', summary, results,
                  [CODE / 'prepare_qwen.py', CODE / 'qwen_mechanism.py', CODE / 'finish_qwen_base.py',
                   CODE / 'follow_pilot.py', CODE / 'group_dro.py'])
    (local / 'commit').write_text(ref)


if __name__ == '__main__':
    packet = json.loads(Path('config.json').read_text())
    root = Path(packet['worktree'])
    run(root, Path(packet['server']), Path(packet['local']))
