"""Exercise aggregation and the UNKNOWN/empty-support gate on synthetic records."""
import med_r2_report

owners = (47, 117, 93, 87)
tasks, consumers, decisions, mask = [], [], {}, {}
for owner in owners:
    native, t2g, loc = (f'n{owner}', f'g{owner}', f'l{owner}')
    tasks.append(dict(order=owner, native=dict(source_group=f's{owner}', image_sha256='native', question=native),
                      events=[dict(task='T0', all_probe_query_ids=[native]),
                              dict(task='T2G', all_probe_query_ids=[t2g]),
                              dict(task='T1L', all_probe_query_ids=[loc])]))
    mask[t2g], mask[loc] = False, True
    for q in [native, t2g, loc] + [f'edit_{owner}_{i}' for i in range(1, 5)]:
        for arm in ('MED', 'LEGACY', 'BASE'):
            key = f'{owner}:{q}:{arm}'
            correct = not (owner == 47 and q == t2g and arm == 'LEGACY')
            decisions[key] = dict(core_correct=correct, full_supported=True)
            consumers.append(dict(arm=arm, mode='single', prefix=1, owner=owner, query_id=q,
                payload_key=key, image_sha256='native' if q == native else 'other', question=q,
                active_target=False, at_cap=False, EOS=True, effective_expert='x', prediction_reused=False))
packet = dict(tasks=tasks, base_mask=mask)
result, _, _ = med_r2_report.summarize(packet, consumers, decisions, set(), 'A', {})
assert result['R2_development_signal'] == 'DEV_PASS'
assert result['T2G_CORE_paired'] == dict(gains=1, losses=0, unknown=0)
missing_key = f'117:g117:MED'
partial = dict(decisions)
partial.pop(missing_key)
result, _, _ = med_r2_report.summarize(packet, consumers, partial, {missing_key}, 'A', {})
assert result['R2_development_signal'] == 'UNKNOWN_OR_NA'
assert result['T2G_CORE_paired']['unknown'] == 1
print('SYNTHETIC_REPORT_PASS')
