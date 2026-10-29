"""Fail-closed identity-only screening; caller must extract metadata without QA."""

OWNERS = {47, 117, 93, 87}


def admit(metadata, deny):
    if set(deny) != {'V95', 'V111'} or not all(deny.values()):
        raise ValueError('missing reserved-role identity sets')
    if set(metadata) != {'evaluation', 'mapped_protection', 'fallback_protection'}:
        raise ValueError('missing fixed source roles')
    evaluation = metadata['evaluation']
    mapped, fallback = metadata['mapped_protection'], metadata['fallback_protection']
    if len(evaluation) != 69 or len(mapped) != 31 or len(fallback) != 30:
        raise ValueError('source counts differ from frozen protocol')
    if {v['owner'] for v in evaluation} != OWNERS:
        raise ValueError('fixed owners missing')
    groups = set()
    for item in evaluation + mapped + fallback:
        group = item.get('group')
        if not isinstance(group, str) or not group:
            raise ValueError('source group unavailable')
        groups.add(group)
    if groups.intersection(set(deny['V95']) | set(deny['V111'])):
        raise ValueError('reserved-role group overlap')
    return dict(status='IDENTITY_CHECK_PASS', evaluation=69, mapped_protection=31, fallback_protection=30,
                fixed_owners=4, forbidden_group_overlap=0, input_is_identity_metadata=True,
                extraction_receipt_required=True)
