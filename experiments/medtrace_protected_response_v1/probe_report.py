"""Frozen paired local-mechanism decision; publish aggregates only."""
import json
import os
from pathlib import Path
import numpy as np

RUN = Path(os.environ['RUN_ROOT'])
ARMS = ('RAW', 'PROJECTED', 'MATCHED_RAW')


def read(path):return json.loads(path.read_text())
def write(path, value):path.write_text(json.dumps(value, indent=2)+'\n')
def mean(values):return float(np.mean(values))


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261009)
    draws = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def decide(protection, edit, lost):
    measurable = protection['matched_KL'] > 1e-8 and edit['MATCHED_RAW'] > 1e-8
    protected = protection['expert_CI'][1] < 0 and protection['source_CI'][1] < 0
    retained = edit['PROJECTED'] >= .9 * edit['MATCHED_RAW'] and lost == 0
    return 'INCONCLUSIVE_SMALL_RESPONSE' if not measurable else ('LOCAL_MECHANISM_SIGNAL' if protected and retained else 'NO_LOCAL_SUPPORT')


def selfcheck():
    protect = dict(matched_KL=1., expert_CI=[-2., -1.], source_CI=[-2., -1.])
    assert decide(protect, dict(MATCHED_RAW=1., PROJECTED=.95), 0) == 'LOCAL_MECHANISM_SIGNAL'
    assert decide(protect, dict(MATCHED_RAW=1., PROJECTED=.89), 0) == 'NO_LOCAL_SUPPORT'
    assert decide(protect, dict(MATCHED_RAW=1., PROJECTED=1.), 1) == 'NO_LOCAL_SUPPORT'
    assert decide(protect, dict(MATCHED_RAW=0., PROJECTED=1.), 0) == 'INCONCLUSIVE_SMALL_RESPONSE'
    assert interval([0.] * 8) == [0., 0.]


def main(expected_forwards=4024, transform=None):
    selfcheck()
    files = sorted((RUN/'private/results').glob('*.json'))
    data = [read(f) for f in files]
    lock = read(RUN/'private/PROBE_LOCK.json')
    assert len(data) == 8 and len({x['expert_order'] for x in data}) == 8
    assert not list((RUN/'private').glob('FAILURE_*.json'))
    assert all(x['status'] == 'COMPLETE' and x['state_restored_exact'] and x['baseline_repeat_exact'] and not x['Base_gradient'] for x in data)
    assert all(len(x['diagnostics']) == 101 for x in data)
    assert sum(x['forward_calls'] for x in data) == lock['forward_calls'] == expected_forwards
    assert sum(x['backward_calls'] for x in data) == lock['backward_calls'] == 784
    assert all(abs(x['function_matching']['actual'] - x['function_matching']['target']) <= 1e-10+1e-3*x['function_matching']['target'] for x in data)
    per_expert = []
    for x in data:
        held = [r for r in x['diagnostics'] if r['role'] == 'HELDOUT_FIT']
        edit = [r for r in x['diagnostics'] if r['role'] == 'EDIT']
        assert len(held) == 96 and len(edit) == 5
        progress = {a: -sum((.5 if r['index'] == 0 else .125)*r['candidates'][a]['NLL_change'] for r in edit) for a in ARMS}
        per_expert.append(dict(expert_order=x['expert_order'], geometry=x['geometry'],
            function_matching=x['function_matching'], map_norms=x['map_norms'],
            heldout_KL={a:mean([r['candidates'][a]['KL'] for r in held]) for a in ARMS},
            edit_NLL_progress=progress,
            edit_lost_correct_tokens={a:sum(r['candidates'][a]['lost_correct_tokens'] for r in edit) for a in ARMS},
            initial_edit_correct_tokens=sum(r['baseline']['correct_tokens'] for r in edit),
            initial_edit_tokens=sum(r['baseline']['tokens'] for r in edit)))
    held = [r for x in data for r in x['diagnostics'] if r['role'] == 'HELDOUT_FIT']
    expert_diff = [x['heldout_KL']['PROJECTED'] - x['heldout_KL']['MATCHED_RAW'] for x in per_expert]
    source_diff = [mean([r['candidates']['PROJECTED']['KL']-r['candidates']['MATCHED_RAW']['KL'] for r in held if r['group'] == g]) for g in range(32)]
    protection = dict(matched_KL=mean([x['heldout_KL']['MATCHED_RAW'] for x in per_expert]),
        projected_KL=mean([x['heldout_KL']['PROJECTED'] for x in per_expert]),
        mean_difference=mean(expert_diff), expert_CI=interval(expert_diff), source_CI=interval(source_diff))
    edit = {a:mean([x['edit_NLL_progress'][a] for x in per_expert]) for a in ARMS}
    lost = sum(x['edit_lost_correct_tokens']['PROJECTED'] for x in per_expert)
    decision = decide(protection, edit, lost)
    panels = {}
    for kind in ('all', 'yes', 'no', 'open', 'initial_all_argmax_correct'):
        subset = [r for r in held if kind == 'all' or r['kind'] == kind or (kind == 'initial_all_argmax_correct' and r['baseline']['all_argmax_correct'])]
        panels[kind] = dict(observations=len(subset), independent_sources=len({r['group'] for r in subset}),
            arms={a:dict(KL=mean([r['candidates'][a]['KL'] for r in subset]),
                        NLL_change=mean([r['candidates'][a]['NLL_change'] for r in subset]),
                        lost_correct_tokens=sum(r['candidates'][a]['lost_correct_tokens'] for r in subset)) for a in ARMS} if subset else None)
    ledger = read(RUN/'RESOURCE_LEDGER.json')
    inherited = read(RUN/'private/INHERITED_COST.json')
    assert all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    resource = dict(cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,
        new_GPU_process_hours=(ledger['gpu_seconds_used']-inherited['gpu_seconds_used'])/3600,
        cumulative_Judge=ledger['Judge_attempts'], new_Judge=0, new_generations=0,
        persistent_new_checkpoints=0, forward_calls=expected_forwards, backward_calls=784, temporary_optimizer_steps=8)
    result = dict(status='COMPLETE', decision=decision, protection=protection, edit_NLL_progress=edit,
        projected_edit_lost_correct_tokens=lost, per_expert=per_expert, heldout_panels=panels,
        resource=resource, independent_confirmation=False, clinical_protection=False,
        full_training_success=False, next_full_training_registered=False)
    if transform is not None:
        result = transform(result, data)
        decision = result['decision']
    write(RUN/'public/RESULTS.json', result)
    write(RUN/'public/RESEARCH_DECISION.json', {k:v for k,v in result.items() if k not in ('per_expert','heldout_panels')})
    write(RUN/'public/COMPLETION_AUDIT.json', dict(status='PASS', experts=8, finite_candidates=24,
        restored_states=8, exact_baseline_repeats=8, function_matches=8, source_groups=32,
        forward_calls=expected_forwards, backward_calls=784, new_generations=0, new_Judge=0, new_checkpoints=0))
    (RUN/'public/REPORT_ZH.md').write_text(f'''# 保护响应一步机制检验\n\n结论：{decision}。这不是完整训练或自由生成准确率结果。\n\n内部留出平均KL：投影 {protection['projected_KL']:.8g}；等幅RAW {protection['matched_KL']:.8g}。差 {protection['mean_difference']:.8g}；编辑专家配对95%区间 {protection['expert_CI']}，来源配对区间 {protection['source_CI']}。\n\n编辑平均NLL进展：{edit}。投影新增原正确目标token错误 {lost}。\n\n8专家、32留出图像来源、96留出问题，每来源三题，不能把768观测视作独立病例。约束仅保持源NLL一阶响应；全词表KL、有限步非线性和编辑收益另测。旧FIT历史暴露、单初态/单步、teacher forcing与未知患者独立性限制仍保留。\n\n全部{expected_forwards}前向/784反向、8次临时候选更新完成；0自由生成、0Judge、0持久新权重，初态逐核恢复。新增GPU进程小时 {resource['new_GPU_process_hours']:.6f}，累计 {resource['cumulative_GPU_process_hours']:.6f}。未登记24/146扩展或完整训练。\n''')
    write(RUN/'private/REPORT_COMPLETE.json', dict(status='COMPLETE', decision=decision))


if __name__ == '__main__':main()
