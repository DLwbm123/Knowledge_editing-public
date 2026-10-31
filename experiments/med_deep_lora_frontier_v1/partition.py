"""One principal judgment per scorer-visible payload under the revised protocol."""
from packet import CONTRACT, digest

FIELDS = ('core_correct', 'full_supported', 'unsupported_expansion', 'extra_contradiction')


def semantic(record):
    return digest(dict(protocol=CONTRACT, model='gpt-6-astra', reasoning='medium',
                       question=record['question'], gold_answer=record['gold_answer'],
                       raw_base_answer=record['raw_base_answer']))


def partition(packet, prior_valid, prior_attempts, known_semantic, known_semantic_attempts, prior_conflicts=()):
    if packet.get('status') != 'SCORE_PACKET_FROZEN' or len(packet['consumers']) != 2032:
        raise ValueError('Four-point packet not frozen')
    payloads = packet['payloads']
    if {x['key'] for x in packet['consumers']} != set(payloads):
        raise ValueError('Consumer coverage mismatch')
    result = dict(known_valid={}, known_unknown=[], conflict_unknown=[], history_unverified=[])
    for key, data in payloads.items():
        if semantic(data['record']) != key or data['record']['opaque_query_id'] != key:
            raise ValueError('Scorer identity mismatch')
        candidates = [prior_valid[k] for k in set(data['legacy_keys']) if k in prior_valid]
        candidates += list(known_semantic.get(key, []))
        verdicts = set()
        for item in candidates:
            v = tuple(item.get(k) for k in FIELDS)
            if any(type(x) is not bool for x in v) or (v[1] and (not v[0] or v[2] or v[3])):
                raise ValueError('Invalid previous verdict')
            verdicts.add(v)
        if len(verdicts) > 1 or set(data['legacy_keys']) & set(prior_conflicts):
            result['conflict_unknown'].append(key)
        elif verdicts:
            result['known_valid'][key] = dict(opaque_query_id=key, **dict(zip(FIELDS, next(iter(verdicts)))))
        elif key in known_semantic_attempts or set(data['legacy_keys']) & set(prior_attempts):
            result['known_unknown'].append(key)
        else:
            result['history_unverified'].append(key)
    if sum(map(len, result.values())) != len(payloads):
        raise ValueError('Score partition incomplete')
    return result
