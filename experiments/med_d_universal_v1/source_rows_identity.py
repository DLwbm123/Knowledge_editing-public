"""Check native/FIT group identities while skipping every QA field."""
from pathlib import Path
import identity_lock
import index
import prior_binding
import source_identity
import stream_extract as stream


def check(path):
    tasks = identity_lock.tasks(path)
    natives = {str(x['order']): x['native'] for x in tasks}
    seen = set()
    counts = {'owners': 0, 'rows': 0, 'qa_decoded': 0}
    with Path(path).open('rb') as f:
        if stream.nonspace(f) != b'{':
            raise ValueError('root object expected')
        found = False
        while True:
            t = stream.nonspace(f)
            if t == b'}':
                break
            if t != b'"':
                raise ValueError('root field expected')
            key = stream.string(f, True)
            if stream.nonspace(f) != b':':
                raise ValueError('root colon expected')
            t = stream.nonspace(f)
            if key != 'source_rows':
                index.skip(f, t)
            else:
                if found or t != b'{':
                    raise ValueError('source row map differs')
                found = True
                while True:
                    t = stream.nonspace(f)
                    if t == b'}':
                        break
                    if t != b'"':
                        raise ValueError('owner key expected')
                    owner = stream.string(f, True)
                    if owner in seen or owner not in natives or stream.nonspace(f) != b':':
                        raise ValueError('unknown or duplicate owner')
                    if stream.nonspace(f) != b'[':
                        raise ValueError('source rows must be an array')
                    rows = []
                    while True:
                        t = stream.nonspace(f)
                        if t == b']':
                            break
                        rows.append(prior_binding.select(f, t, {'source_group': True, 'image_path': True, 'audit_role': True}))
                        t = stream.nonspace(f)
                        if t == b']':
                            break
                        if t != b',':
                            raise ValueError('row separator expected')
                    if [x.get('audit_role') for x in rows] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']:
                        raise ValueError('source role sequence differs')
                    source_identity.require_same(natives[owner], rows)
                    seen.add(owner)
                    counts['owners'] += 1
                    counts['rows'] += len(rows)
                    t = stream.nonspace(f)
                    if t == b'}':
                        break
                    if t != b',':
                        raise ValueError('owner separator expected')
            t = stream.nonspace(f)
            if t == b'}':
                break
            if t != b',':
                raise ValueError('root separator expected')
    if not found or seen != set(natives) or counts['rows'] != 730:
        raise ValueError('source row coverage incomplete')
    return counts
