"""User-authorized continuation of previously unattempted inputs only."""


def authorized_tail(attempts, keys, receipt):
    assert receipt['status'] == 'USER_AUTHORIZED_UNATTEMPTED_ONLY'
    eligible = set(receipt['eligible_keys'])
    assert len(eligible) == 128 and set(keys) <= eligible
    acknowledged = set(receipt['acknowledged_batch_ids'])
    previous = [a for a in attempts if a['id'] in acknowledged]
    assert len(previous) == len(acknowledged) == 4
    assert all(a['status'] == 'FAILED_NO_RETRY' and a['transport_failure'] for a in previous)
    failed = {k for a in previous for k in a['keys']}
    assert len(failed) == 198 and not failed & eligible
    # Preserve failures; count the transport stop from the explicitly authorized restart.
    return [a for a in reversed(attempts) if a['id'] not in acknowledged]


def selfcheck():
    eligible = [f'new{i}' for i in range(128)]
    old = [dict(id=str(i), status='FAILED_NO_RETRY', transport_failure=True,
                keys=[f'old{k}' for k in range(i * 50, min((i + 1) * 50, 198))])
           for i in range(4)]
    receipt = dict(status='USER_AUTHORIZED_UNATTEMPTED_ONLY', eligible_keys=eligible,
                   acknowledged_batch_ids=[a['id'] for a in old])
    assert authorized_tail(old, eligible[:40], receipt) == []
    recent = [dict(id=f'next{i}', status='FAILED_NO_RETRY', transport_failure=True, keys=[eligible[i]])
              for i in range(3)]
    assert len(authorized_tail(old + recent, eligible[3:4], receipt)) == 3
    for keys in [['old0'], ['not_authorized']]:
        try:
            authorized_tail(old, keys, receipt)
        except AssertionError:
            pass
        else:
            raise AssertionError('Attempted or unauthorized input accepted')
    return dict(status='PASS', original_failures_excluded_only_from_stop_counter=True,
                failed_retry_rejected=True, new_three_failure_stop_preserved=True)


if __name__ == '__main__':
    print(selfcheck())
