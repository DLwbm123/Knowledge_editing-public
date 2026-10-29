"""Summarize candidate versus original MED with the original support rules."""
import med133_report as common


def summarize(packet, consumers, decisions, missing, stage, cost):
    assert stage == 'A'
    raw, _, owner_panels = common.summarize(packet, consumers, decisions, missing, stage, cost)
    old_gates = raw['gates']['single_1']
    panels = owner_panels['single_1']
    gains = losses = unknown = 0
    for owner in (47, 117, 93, 87):
        a = panels['MED']['T2G']['core_correct'][owner]
        b = panels['LEGACY']['T2G']['core_correct'][owner]
        assert len(a) == len(b)
        for candidate, original in zip(a, b):
            if candidate is None or original is None:
                unknown += 1
            else:
                gains += bool(candidate and not original)
                losses += bool(original and not candidate)
    gates = dict(T2G_CORE_net_at_least_one=(gains - losses >= 1) if not unknown else None,
                 T2G_FULL_no_regression=old_gates['T2G_FULL_no_legacy_regression'],
                 source_CORE_no_regression=old_gates['source_CORE_no_legacy_regression'],
                 source_FULL_no_regression=old_gates['source_FULL_no_legacy_regression'],
                 CORE_no_new_damage=old_gates['CORE_no_new_damage'],
                 FULL_no_new_damage=old_gates['FULL_no_new_damage'])
    if any(v is False for v in gates.values()):
        signal = 'FAIL'
    elif any(v is None for v in gates.values()):
        signal = 'UNKNOWN_OR_NA'
    else:
        signal = 'DEV_PASS'
    raw.update(status=raw['status'], original146_exposed_DEV=True,
               independent_confirmation=False, arm_labels={'MED': 'candidate_R2',
               'LEGACY': 'original_MED_V133', 'BASE': 'unedited_base'},
               R2_development_gate=gates, R2_development_signal=signal,
               T2G_CORE_paired=dict(gains=gains, losses=losses, unknown=unknown))
    raw.pop('gates')
    raw.pop('scientific_negative_does_not_cancel_sequential')
    report = ('# MED 第二响应方向：四编辑开发结果\n\n'
              f"状态：{raw['status']}；DEV 综合信号：{signal}。\n\n"
              f'T2G CORE 配对新增正确 {gains}、新增错误 {losses}、未知 {unknown}。'
              '完整 CORE/FULL 指标、支持、保护损伤和费用见结果文件。\n\n'
              '结果使用已曝光开发集，LEGACY 字段代表原 MED V133；'
              '空支持不是保护通过，来源支持不是临床真值。\n')
    return raw, report, owner_panels
