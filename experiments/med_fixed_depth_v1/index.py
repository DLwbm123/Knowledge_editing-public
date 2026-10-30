"""Read only identity and task-role metadata; never deserialize question/answer values."""
import json
import re
import stream_extract as s


def skip(f, token):
    if token == b'"':
        s.string(f)
    elif token in (b'{', b'['):
        s.skip_compound(f, token)
    else:
        s.scalar(f, token)


def value(f, token):
    if token == b'"':
        return s.string(f, True)
    if token in b'0123456789':
        return s.scalar(f, token, True)
    raise ValueError('Invalid identity')


def array(f, kind):
    result = []
    while True:
        token = s.nonspace(f)
        if token == b']':
            return result
        if token != b'{':
            raise ValueError('Expected array object')
        start = f.tell() - 1
        row = obj(f, kind)
        row['_span'] = (start, f.tell())
        result.append(row)
        token = s.nonspace(f)
        if token == b']':
            return result
        if token != b',':
            raise ValueError('Expected array delimiter')


def strings(f):
    result = []
    while True:
        token = s.nonspace(f)
        if token == b']':
            return result
        if token != b'"':
            raise ValueError('Expected query identity')
        result.append(s.string(f, True))
        token = s.nonspace(f)
        if token == b']':
            return result
        if token != b',':
            raise ValueError('Expected query identity delimiter')


def obj(f, kind):
    result = {}
    fields = {'task': {'order'}, 'event': {'task'}, 'native': {'image_path', 'source_group'},
              'plan': {'method', 'mode', 'prefix', 'owner'},
              'query': {'query_id', 'image_path', 'source_group', 'dataset', 'audit_role'}}[kind]
    while True:
        token = s.nonspace(f)
        if token == b'}':
            return result
        if token != b'"':
            raise ValueError('Expected object field')
        key = s.string(f, True)
        if s.nonspace(f) != b':':
            raise ValueError('Expected colon')
        token = s.nonspace(f)
        if key in fields:
            if key in result:
                raise ValueError('Duplicate identity field')
            if token == b'n':
                skip(f, token)
                result[key] = None
            else:
                result[key] = value(f, token)
        elif kind == 'task' and key == 'native':
            if token != b'{' or key in result:
                raise ValueError('Invalid native identity')
            result[key] = obj(f, 'native')
        elif kind == 'task' and key == 'events':
            if token != b'[' or key in result:
                raise ValueError('Invalid event list')
            result[key] = array(f, 'event')
        elif kind == 'event' and key == 'all_probe_query_ids':
            if token != b'[' or key in result:
                raise ValueError('Invalid probe identities')
            result[key] = strings(f)
        elif kind == 'plan' and key == 'query':
            if token != b'{' or key in result:
                raise ValueError('Invalid query identity')
            result[key] = obj(f, 'query')
        else:
            skip(f, token)
        token = s.nonspace(f)
        if token == b'}':
            return result
        if token != b',':
            raise ValueError('Expected object delimiter')


def scan(path, all_candidates=False):
    sections = {}
    with open(path, 'rb') as f:
        if s.nonspace(f) != b'{':
            raise ValueError('Expected root object')
        while True:
            token = s.nonspace(f)
            if token == b'}':
                break
            if token != b'"':
                raise ValueError('Expected root field')
            key = s.string(f, True)
            if s.nonspace(f) != b':':
                raise ValueError('Expected root colon')
            token = s.nonspace(f)
            if key in ('tasks', 'plan'):
                if token != b'[' or key in sections:
                    raise ValueError('Invalid root array')
                sections[key] = array(f, 'task' if key == 'tasks' else 'plan')
            else:
                skip(f, token)
            token = s.nonspace(f)
            if token == b'}':
                break
            if token != b',':
                raise ValueError('Invalid root delimiter')
    tasks, plan = sections['tasks'], sections['plan']
    if len(tasks) != 146 or [x.get('order') for x in tasks] != list(range(1, 147)):
        raise ValueError('Original task order mismatch')
    candidates = [x['order'] for x in tasks if {'T2G', 'T2L'} <= {v.get('task') for v in x.get('events', [])}]
    owners = candidates if all_candidates else candidates[:16]
    if len(owners) < 16:
        raise ValueError('Fewer than sixteen metadata candidates')
    selected = [x for x in plan if x.get('method') == 'MED' and x.get('mode') == 'single' and x.get('owner') in owners]
    if not selected or {x['owner'] for x in selected} != set(owners):
        raise ValueError('Incomplete selected plan coverage')
    return {'owners': owners, 'tasks': [x for x in tasks if x['order'] in owners], 'plan': selected,
            'qa_values_decoded': 0, 'scores_decoded': 0}


def read_mask(path, allowed_ids):
    """Read only pre-existing Base eligibility bits for already role-checked query IDs."""
    allowed = set(allowed_ids)
    found = {}
    with open(path, 'rb') as f:
        if s.nonspace(f) != b'{':
            raise ValueError('Expected root')
        while True:
            token = s.nonspace(f)
            if token == b'}':
                break
            if token != b'"':
                raise ValueError('Expected root key')
            key = s.string(f, True)
            if s.nonspace(f) != b':':
                raise ValueError('Expected colon')
            token = s.nonspace(f)
            if key != 'base_mask':
                skip(f, token)
            else:
                if token != b'{':
                    raise ValueError('Expected Base eligibility object')
                while True:
                    token = s.nonspace(f)
                    if token == b'}':
                        break
                    if token != b'"':
                        raise ValueError('Expected mask key')
                    qid = s.string(f, True)
                    if s.nonspace(f) != b':':
                        raise ValueError('Expected mask colon')
                    token = s.nonspace(f)
                    if qid in allowed:
                        if qid in found or token not in (b't', b'f'):
                            raise ValueError('Missing or ambiguous Base eligibility')
                        start = f.tell() - 1
                        skip(f, token)
                        end = f.tell()
                        f.seek(start)
                        found[qid] = json.loads(f.read(end - start))
                        f.seek(end)
                    else:
                        skip(f, token)
                    token = s.nonspace(f)
                    if token == b'}':
                        break
                    if token != b',':
                        raise ValueError('Invalid mask delimiter')
            token = s.nonspace(f)
            if token == b'}':
                break
            if token != b',':
                raise ValueError('Invalid root delimiter')
    if set(found) != allowed:
        raise ValueError('Base eligibility missing for allowed query')
    return found


def group(value):
    matches = {x.lower() for x in re.findall(r'xmlab\d+', value or '', re.I)}
    if len(matches) == 1:
        return next(iter(matches))
    if not matches and value and re.search(r'vqa[-_]?rad', value, re.I):
        return 'VQA_RAD_UNKNOWN_GROUP'
    return None
