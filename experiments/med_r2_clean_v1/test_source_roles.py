import io
import json
import source_roles

rows = [dict(query_id=f'q{i}', source_group=f'SLAKE:xmlab{i}', source_qid=i) for i in range(192)]
qualification = [dict(row=v) for v in rows]
mapped = [dict(index=i, row=rows[i]) for i in range(31)]
fallback = [dict(index=i, query_id=f'q{i}') for i in range(31, 61)]
deny = dict(V95=['xmlab0', 'xmlab31'] + [f'xmlab{i}' for i in range(200, 222)],
            V111=[f'xmlab{i}' for i in range(300, 315)])
lock = source_roles.freeze(mapped, fallback, qualification, deny)
assert len(lock['keep']) == 59 and len(lock['excluded']) == 2 and lock['source_QA_decoded'] == 0
buffer = io.BytesIO()
for item in mapped:
    start = buffer.tell()
    buffer.write(json.dumps(dict(index=item['index'], row=item['row'])).encode())
    item['_span'] = (start, buffer.tell())
lock = source_roles.freeze(mapped, fallback, qualification, deny)
decoded = source_roles.decode_mapped(buffer, lock)
assert len(decoded) == 30 and all(item['index'] != 0 for item in decoded)
wrong = [dict(v) for v in fallback]
wrong[0]['index'] = 1
try:
    source_roles.freeze(mapped, wrong, qualification, deny)
except ValueError:
    pass
else:
    raise AssertionError('duplicate source index accepted')
print('SOURCE_IDENTITY_GATE_PASS')
