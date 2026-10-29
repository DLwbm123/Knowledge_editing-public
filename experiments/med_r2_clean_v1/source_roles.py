"""Freeze the originally indexed 59 permitted protection sources by identity."""
import identity_compare as identity
import json


def freeze(mapped, fallback, qualification, deny):
    if len(mapped) != 31 or len(fallback) != 30 or len(qualification) != 192:
        raise ValueError('original source counts differ')
    if len(deny.get('V95', [])) != 24 or len(deny.get('V111', [])) != 15:
        raise ValueError('incomplete reserved groups')
    denied = set(deny['V95']) | set(deny['V111'])
    rows = {v['row']['query_id']: v['row'] for v in qualification}
    if len(rows) != 192:
        raise ValueError('qualification query identities not unique')
    sources = []
    for item in mapped:
        row = item['row']
        if rows[row['query_id']]['source_group'] != row['source_group']:
            raise ValueError('mapped qualification identity mismatch')
        sources.append(dict(index=item['index'], query_id=row['query_id'],
                            group=identity.group(row['source_group']), kind='mapped',
                            span=item.get('_span')))
    for item in fallback:
        row = rows[item['query_id']]
        sources.append(dict(index=item['index'], query_id=item['query_id'],
                            group=identity.group(row['source_group']), kind='fallback'))
    if {v['index'] for v in sources} != set(range(61)) or len({v['query_id'] for v in sources}) != 61:
        raise ValueError('original source index or identity mismatch')
    keep = [v for v in sources if v['group'] not in denied]
    rejected = [v for v in sources if v['group'] in denied]
    if sum(v['kind'] == 'mapped' for v in keep) != 30 or sum(v['kind'] == 'fallback' for v in keep) != 29:
        raise ValueError('identity exclusions differ from frozen protocol')
    return dict(keep=sorted(keep, key=lambda v: v['index']),
                excluded=sorted(rejected, key=lambda v: v['index']),
                label='IDENTITY_LOCK_ONLY', source_QA_decoded=0)


def decode_mapped(stream, lock):
    if lock['label'] != 'IDENTITY_LOCK_ONLY':
        raise ValueError('source identities not locked')
    output = []
    for item in lock['keep']:
        if item['kind'] != 'mapped':
            continue
        span = item['span']
        if not span or not 0 < span[1] - span[0] <= 1_048_576:
            raise ValueError('invalid source record span')
        stream.seek(span[0])
        value = json.loads(stream.read(span[1] - span[0]))
        if (value['index'] != item['index'] or value['row']['query_id'] != item['query_id'] or
                identity.group(value['row']['source_group']) != item['group']):
            raise ValueError('decoded record does not match locked identity')
        output.append(value)
    if len(output) != 30:
        raise ValueError('incorrect clean mapped source count')
    return output
