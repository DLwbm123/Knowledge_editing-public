"""Resolve frozen scorer identities against known decisions without new calls."""
import packet

FIELDS = ('core_correct', 'full_supported', 'unsupported_expansion', 'extra_contradiction')


def resolve(score_packet, history):
    if score_packet.get('status') != 'SCORE_PACKET_FROZEN' or len(score_packet['consumers']) != 3026:
        raise ValueError('selected score packet not frozen')
    payloads = score_packet['payloads']
    if {row['key'] for row in score_packet['consumers']} != set(payloads):
        raise ValueError('score payload coverage differs')
    valid = history['valid']
    conflicts = set(history['conflicts'])
    attempts = set(history['attempts'])
    semantic = history['semantic_valid']
    semantic_attempts = set(history['semantic_attempts'])
    result = {'known_valid': {}, 'known_unknown': [], 'conflict_unknown': [], 'history_unverified': []}
    for key, data in payloads.items():
        row = data['record']
        contract = {'protocol': packet.CONTRACT, 'model': 'gpt-6-astra', 'reasoning': 'medium',
                    'question': row['question'], 'gold_answer': row['gold_answer'],
                    'raw_base_answer': row['raw_base_answer']}
        if packet.digest(contract) != key or row['opaque_query_id'] != key:
            raise ValueError('semantic score identity differs')
        legacy = set(data['legacy_keys'])
        candidates = [valid[k] for k in legacy if k in valid] + list(semantic.get(key, []))
        verdicts = set()
        for candidate in candidates:
            values = tuple(candidate.get(name) for name in FIELDS)
            if any(type(v) is not bool for v in values) or (values[1] and (not values[0] or values[2] or values[3])):
                raise ValueError('invalid prior judgment')
            verdicts.add(values)
        if len(verdicts) > 1 or legacy & conflicts:
            result['conflict_unknown'].append(key)
        elif verdicts:
            result['known_valid'][key] = dict(opaque_query_id=key, **dict(zip(FIELDS, next(iter(verdicts)))))
        elif key in semantic_attempts or legacy & attempts:
            result['known_unknown'].append(key)
        else:
            result['history_unverified'].append(key)
    if sum(len(v) for v in result.values()) != len(payloads):
        raise ValueError('incomplete score partition')
    return result
