"""Deliver the calibration result and score, summarize, and publish the finished pilot."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


REPORT = Path('reports/med_mdu150_t2g_supplement_v1_20261106')
CODE = Path('experiments/med_mdu_v2_admission_v1')
PROXY = 'http://127.0.0.1:7897'
BRANCH = 'refs/heads/experiment/med150-20261027'


def remote(folder, names):
    source = f"""
from pathlib import Path
import json
p=Path({str(folder)!r})
print(json.dumps({{n:(p/n).read_text() for n in {list(names)!r} if (p/n).is_file()}}))
"""
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'pro5000', 'python3', '-'],
                          input=source, text=True, capture_output=True, timeout=40, check=True)
    return json.loads(proc.stdout)


def wait_for(folder, names, deadline):
    while time.time() < deadline:
        try:
            files = remote(folder, names)
            if files:
                return files
        except (subprocess.SubprocessError, json.JSONDecodeError):
            pass
        time.sleep(30)
    raise TimeoutError('Expected server receipt did not arrive by its deadline')


def publish(worktree, basename, report, result, code_files=()):
    proxy_env = {**os.environ, 'NO_PROXY': '', 'no_proxy': '', 'HTTPS_PROXY': PROXY, 'https_proxy': PROXY}
    git = ['git', '-c', 'http.proxy=' + PROXY]
    def run(*args, cwd=worktree):
        return subprocess.run([*git, *args], cwd=cwd, env=proxy_env, check=True,
                              text=True, capture_output=True).stdout.strip()
    run('fetch', 'origin', BRANCH)
    head = run('rev-parse', 'FETCH_HEAD')
    with tempfile.TemporaryDirectory(prefix='.pub.') as scratch:
        checkout = Path(scratch) / 'w'
        run('worktree', 'add', '--detach', str(checkout), head)
        try:
            md = REPORT / f'{basename}_ZH.md'
            js = REPORT / f'{basename}_RESULTS.json'
            (checkout / md).write_text(report)
            (checkout / js).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
            summary = REPORT / 'PROJECT_SUMMARY_ZH.md'
            main_report = checkout / summary
            text = main_report.read_text()
            label = {
                'MDU_V2_STAGE0_KL_CALIBRATION': '开发侧 KL 标定',
                'MDU_V2_STAGE0_KL_CALIBRATION_EXTENDED': '加大 KL 系数的开发复验',
                'MDU_V2_STAGE1_MATCHED_PILOT': '阶段 1 四臂结果',
                'MDU_V2_QWEN_BASE': '第二骨干 Base 资格',
                'MDU_V2_QWEN_MECHANISM': '第二骨干机制复现',
            }[basename]
            link = f'[{label}]({md.name})'
            assert link not in text
            mark = '## 6. GitHub 和复现入口'
            assert text.count(mark) == 1
            main_report.write_text(text.replace(mark, link + '\n\n' + mark))
            for path in code_files:
                (checkout / path).write_text((worktree / path).read_text())
            run('add', '--', str(md), str(js), str(summary), *(str(p) for p in code_files), cwd=checkout)
            run('diff', '--cached', '--check', cwd=checkout)
            run('commit', '-m', f'Publish {basename.replace("_", " ").lower()} results', cwd=checkout)
            rendered = run('rev-parse', 'HEAD', cwd=checkout)
            run('push', 'origin', f'HEAD:{BRANCH}', cwd=checkout)
        finally:
            run('worktree', 'remove', str(checkout))
    base = f'https://raw.githubusercontent.com/DLwbm123/Knowledge_editing-public/{rendered}'
    result_get = subprocess.run(['curl', '--proxy', PROXY, '--noproxy', '', '--fail', '--silent', '--show-error',
                                 '--max-time', '30', '--config', '-'], input=f'url = "{base}/{md}"\n',
                                text=True, capture_output=True, check=True)
    assert result_get.stdout == report
    return rendered


def report_calibration(result):
    values = result['candidates']
    lines = ['# MDU-v2 阶段 0 开发侧 KL 标定', '',
             '8 张开发图像与正式四臂面板无重叠。每档系数给每臂同样的 8 步预算，共 768 次弃用更新。'
             '正式训练重新从冻结 MED 初态开始。验证使用 8 个开发负例的归档 Base 前缀，'
             f"其冻结 MDU150 中位 KL 上限为 {result['reference_median']:.6g}。", '',
             '| 系数 | P1 | P2 | P3 | P4 | 四臂达标 |',
             '|---:|---:|---:|---:|---:|:---:|']
    for row in values.values():
        arms = ' | '.join(f'{v:.6g}' for v in row['validation_median'].values())
        lines.append(f"| {row['beta']:.6g} | {arms} | {'是' if row['passes'] else '否'} |")
    beta = result['beta']
    decision = f'正式四臂共用系数为 **{beta:.6g}**。' if beta is not None else '三档均未通过。正式四臂未启动。'
    lines.extend(['', decision, '阶段 1 的留出问法与官方评测均未用于标定。', ''])
    return '\n'.join(lines)


def report_pilot(result, beta, scoring):
    lines = ['# MDU-v2 阶段 1 四臂小面板', '',
             f'30 张图像，四臂各 160 步每专家，共 19,200 次更新。共同 KL 系数为 {beta:.6g}。'
             f"同合同新评分 {scoring['target_payloads']} 条，未知 {scoring['unknown']} 条。"
             '问法复核使用 GPT-6.2 Astra，独立人工复核为 0。', '',
             '| 方法 | 情态 CORE | 情境 CORE | T2L CORE | 原问 CORE | FIT CORE | 选择性间隔 | G2 | G3 |',
             '|---|---:|---:|---:|---:|---:|---:|:---:|:---:|']
    for arm, gap in result['selection_margin'].items():
        cells = []
        for role in ('holdout_modal', 'holdout_context', 'historical_t2l', 'native', 'fit'):
            group = result['groups'][role]
            stat = group['arms'][arm]
            unknown = f"（未知 {stat['unknown']}）" if stat['unknown'] else ''
            cells.append(f"{stat['core_correct']}/{group['events']}{unknown}")
        gate = result['gates'].get(arm, {})
        cells += [f'{gap:.4f}', str(gate.get('locality', '—')), str(gate.get('source', '—'))]
        lines.append('| ' + ' | '.join([arm, *cells]) + ' |')
    lines.extend(['', '## 选择性间隔配对比较', ''])
    for arm, stat in result['paired_comparisons'].items():
        lo, hi = stat['ci95']
        lines.append(f"- {arm.replace('_minus_', ' 对 ')} 为 {stat['delta']:+.4f}，95% CI [{lo:+.4f}, {hi:+.4f}]。")
    lines.extend(['', '两套负例前缀 KL、读入、残差和整答 NLL 见聚合 JSON。评分器沿用历史别名，未提供固定快照。', ''])
    return '\n'.join(lines)


def main(local, pilot, worktree, cutoff, extended=False):
    watch = local / ('followup_extended' if extended else 'followup')
    watch.mkdir(exist_ok=True)
    selection = wait_for(pilot, ('beta_selection.json',), cutoff)
    result = json.loads(selection['beta_selection.json'])
    (watch / 'beta_selection.json').write_text(selection['beta_selection.json'])
    try:
        stem = 'MDU_V2_STAGE0_KL_CALIBRATION'
        if extended:
            stem += '_EXTENDED'
        key = publish(worktree, stem, report_calibration(result), result)
        (watch / 'calibration_commit').write_text(key)
    except (subprocess.SubprocessError, AssertionError) as exc:
        (watch / 'calibration_push_error').write_text(str(exc) + '\n' + str(getattr(exc, 'stderr', '')))
    if result['beta'] is None:
        return

    start = json.loads(wait_for(pilot, ('start.json',), cutoff + 300)['start.json'])
    deadline = start['deadline']
    files = wait_for(pilot, ('worker_exit.json',), deadline)
    exits = json.loads(files['worker_exit.json'])
    (watch / 'worker_exit.json').write_text(files['worker_exit.json'])
    if exits['codes'] != [0, 0]:
        raise RuntimeError(f"GPU pilot was incomplete. Exit codes {exits['codes']}")

    packet = remote(pilot, ('done0.json', 'done1.json', 'metrics0.jsonl', 'metrics1.jsonl'))
    assert len(packet) == 4
    gpu = watch / 'gpu'
    gpu.mkdir()
    for name, content in packet.items():
        (gpu / name).write_text(content)

    score = Path(tempfile.mkdtemp(prefix='.r.', dir='/private/tmp'))
    preparer = worktree / CODE / 'reference_scores.py'
    argv = [str(preparer), 'prepare_pilot', str(gpu), str(score)]
    loader = f'import runpy,sys;sys.argv={argv!r};runpy.run_path(sys.argv[0],run_name="__main__")'
    payloads = subprocess.check_output([sys.executable, '-'], input=loader, text=True).strip()
    scorer = worktree / 'experiments/med_mdu150_t2g_supplement_v1/score.py'
    provider = worktree / 'experiments/medtrace_full_method_comparison_v2_20261003/execution_source'
    env = dict(os.environ, FIXED_SCORE_ROOT=str(score), FIXED_WORKTREE=str(worktree),
               FIXED_JUDGE_SOURCE=str(provider), FIXED_CLI='/opt/homebrew/bin/codex',
               FIXED_ITEM_COUNT=payloads, FIXED_SCORING_EPOCH='STAGE1_MATCHED_PILOT_20261127',
               EXPERIMENT_DEADLINE=str(deadline))
    with scorer.open() as source, (score / 'log').open('x') as output:
        subprocess.run([sys.executable, '-'], stdin=source, stdout=output, stderr=subprocess.STDOUT,
                       cwd=score, env=env, check=True, timeout=max(1, deadline - time.time()))
    for name in ('COMPLETE.json', 'DECISIONS.json', 'consumers.private.json'):
        (watch / name).write_text((score / name).read_text())
    sys.path.insert(0, str(worktree / CODE))
    from pilot_results import summarize
    roster = local / 'prepared_r2/inventory.private.json'
    result = summarize(gpu, watch, local / 'frozen_reference_gpu67', roster)
    scoring = json.loads((score / 'COMPLETE.json').read_text())
    (watch / 'pilot_results.json').write_text(json.dumps(result, allow_nan=False))
    key = publish(worktree, 'MDU_V2_STAGE1_MATCHED_PILOT', report_pilot(result, json.loads(selection[
        'beta_selection.json'])['beta'], scoring), dict(pilot=result, scoring=scoring))
    (watch / 'pilot_commit').write_text(key)


if __name__ == '__main__':
    config = json.loads(Path('config.json').read_text())
    main(Path(config['local']), Path(config['pilot']), Path(config['worktree']), config['calibration_cutoff'], config.get('extended', False))
