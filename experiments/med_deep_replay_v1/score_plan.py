"""Partition a frozen packet into exact reuse, prior unknown and new requests.

The caller must independently enumerate the relevant historical attempt
ledgers and validate the scoring contract. This module does not load ledgers.
"""


def plan(packet, accepted, attempted, ledger_coverage, limit=508):
    if packet.get('status') != 'SCORE_PACKET_FROZEN' or packet.get('source_events') != 508:
        raise ValueError('Fixed prediction packet not frozen')
    payloads = packet['payloads']
    consumers = packet['consumers']
    if len(consumers) != 508 or len(payloads) > limit:
        raise ValueError('Consumer coverage or budget mismatch')
    if len({(x['owner'], x['query_id']) for x in consumers}) != 508:
        raise ValueError('Duplicate consumer')
    if any(c['arm'] != 'REPLAY' or c['key'] not in payloads for c in consumers):
        raise ValueError('Unbound consumer')
    if set(payloads) != {c['key'] for c in consumers}:
        raise ValueError('Orphan payload')
    if ledger_coverage != 'CONTRACT_AND_ATTEMPTS_INDEPENDENTLY_VERIFIED':
        raise ValueError('Historical attempt coverage not independently verified')
    attempted = set(attempted)
    for key, value in accepted.items():
        if value.get('opaque_query_id') != key or any(type(value.get(k)) is not bool for k in
                ('core_correct','full_supported','unsupported_expansion','extra_contradiction')):
            raise ValueError('Invalid accepted decision')
        if value['full_supported'] and (not value['core_correct'] or value['unsupported_expansion'] or value['extra_contradiction']):
            raise ValueError('Inconsistent accepted decision')
    keys = set(payloads)
    reused = keys & set(accepted)
    unknown = (keys & attempted) - reused
    pending = keys - reused - unknown
    if len(pending) > limit or pending & attempted:
        raise ValueError('New attempt limit or no-retry constraint violated')
    return dict(status='SCORE_PLAN_FROZEN',reused=sorted(reused),unknown=sorted(unknown),pending=sorted(pending),
                reuse_decisions={k:accepted[k] for k in sorted(reused)},max_new_attempts=limit,
                automatic_retry=False)
