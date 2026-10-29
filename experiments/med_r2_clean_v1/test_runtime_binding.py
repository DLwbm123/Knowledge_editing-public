import io
import json
import runtime_binding


def test_configuration_only():
    data = {'opaque': {'question': 'PRIVATE_QUESTION', 'runtime': {'precision': 'fp32'},
                       'reference': 'PRIVATE_ANSWER', 'generation': {'max_new_tokens': 1024}}}
    result = runtime_binding.extract(io.BytesIO(json.dumps(data).encode()))
    assert result == {'configuration_only': {'runtime': {'precision': 'fp32'},
                                               'generation': {'max_new_tokens': 1024}}}
    assert 'PRIVATE' not in repr(result)
    for bad in (b'{}', b'{"a":{"runtime":{}}}',
                b'{"a":{"runtime":{},"runtime":{},"generation":{}}}'):
        try: runtime_binding.extract(io.BytesIO(bad))
        except ValueError: pass
        else: raise AssertionError('invalid binding accepted')


if __name__ == '__main__':
    test_configuration_only()
    print('configuration extraction synthetic checks: PASS')
