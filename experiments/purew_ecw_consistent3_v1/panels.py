"""Pure metadata selectors for scoring; identical to the frozen GPU evaluator."""
# shortcut: mirror the frozen evaluator; centralize in the next protocol revision, not during this run.

def protection_ids(tasks, ledger, task=None):
    active = {(t['native']['image_sha256'], t['native']['question']) for t in tasks[:100]}
    return list(dict.fromkeys(q for t in tasks[:100] for e in t['events']
        if e['task'] in ((task,) if task else ('T1L','T2L')) for q in e['all_probe_query_ids']
        if ledger['Base_correctness'][q] and (ledger['queries'][q]['image_sha256'],ledger['queries'][q]['question']) not in active))


def expected_ids(tasks, ledger, mode, owner, prefix):
    if mode == 'protect100_R0':
        assert owner == 0 and prefix == 100
        return protection_ids(tasks, ledger)
    if mode == 'bank_R0':
        assert owner == 0 and prefix in (1,3)
        return list(dict.fromkeys(q for t in tasks[:prefix] for e in t['events'] for q in e['all_probe_query_ids']))
    assert mode == 'insertion_R0' and 1 <= owner <= 3 and prefix == owner
    t = tasks[owner-1]
    return list(dict.fromkeys([t['edit_id']] + [q for e in t['events'] if e['task']=='T2G'
        for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]]))
