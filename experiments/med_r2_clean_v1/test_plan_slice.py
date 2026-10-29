import io
import json
import plan_slice


def run():
    owners = [47, 117, 93, 87]
    rows = [dict(method='MED', mode='single', owner=owners[i % 4],
                 query=dict(query_id='q'+str(i), image_path=f'/SLAKE/xmlab{i}/image.png',
                            question='RESERVED_SECRET' if i == 0 else 'permitted')) for i in range(69)]
    data = json.dumps({'plan': rows}).encode()
    locked = plan_slice.lock(io.BytesIO(data), {'xmlab0'})
    assert sum(v['excluded'] for v in locked['items']) == 1
    admission = {'status': 'PLAN_CONTENT_AUTHORIZED', 'consumers': 68}
    try: plan_slice.decode(io.BytesIO(data), locked, {})
    except ValueError: pass
    else: raise AssertionError('unadmitted content accepted')
    original = plan_slice.json.loads
    def checked(value, *args, **kwargs):
        assert b'RESERVED_SECRET' not in value if isinstance(value, bytes) else 'RESERVED_SECRET' not in value
        return original(value, *args, **kwargs)
    plan_slice.json.loads = checked
    try:
        result = plan_slice.decode(io.BytesIO(data), locked, admission)
        assert len(result) == 68
    finally:
        plan_slice.json.loads = original
    changed = data.replace(b'xmlab1/', b'xmlab9/', 1)
    try: plan_slice.decode(io.BytesIO(changed), locked, admission)
    except ValueError: pass
    else: raise AssertionError('changed identity accepted')
    print('plan lock and selected decoding synthetic checks: PASS')


if __name__ == '__main__': run()
