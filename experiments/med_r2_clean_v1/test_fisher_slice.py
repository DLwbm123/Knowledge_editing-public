import io
import torch

source = dict(x=torch.arange(61 * 3, dtype=torch.float32).reshape(61, 3),
              y=torch.arange(61 * 2 * 4, dtype=torch.float32).reshape(61, 2, 4),
              token_counts=[1] * 61)
buffer = io.BytesIO()
torch.save(source, buffer)
fd = os.memfd_create('synthetic')
try:
    os.write(fd, buffer.getvalue())
    selected = list(range(0, 58, 2))
    blocks = extract_fd(fd, selected, (61, 3, 2, 4))
    assert len(blocks) == 29
    for block in blocks:
        i = block['index']
        np.testing.assert_array_equal(block['x'], source['x'][i:i+1].numpy())
        np.testing.assert_array_equal(block['y'], source['y'][i:i+1].numpy())
    lock = dict(label='IDENTITY_LOCK_ONLY', keep=[dict(index=i, kind='fallback') for i in selected],
                excluded=[dict(index=1)])
    assert len(extract_from_lock(fd, lock, (61, 3, 2, 4))) == 29
    additional = [i for i in range(61) if i not in selected and i not in (1, 3)]
    assert len(additional) == 30
    lock['keep'].extend(dict(index=i, kind='mapped') for i in additional)
    lock['excluded'] = [dict(index=1), dict(index=3)]
    assert len(extract_from_lock(fd, lock, (61, 3, 2, 4), include_mapped=True)) == 59
    lock['excluded'] = [dict(index=0)]
    try:
        extract_from_lock(fd, lock, (61, 3, 2, 4))
    except ValueError:
        pass
    else:
        raise AssertionError('excluded source accepted')
    try:
        extract_fd(fd, [0] * 29, (61, 3, 2, 4))
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate source accepted')
finally:
    os.close(fd)
buffer = io.BytesIO()
torch.save(dict(x=source['x'].T, y=source['y'], token_counts=source['token_counts']), buffer)
fd = os.memfd_create('synthetic-invalid-layout')
try:
    os.write(fd, buffer.getvalue())
    try:
        extract_fd(fd, list(range(0, 58, 2)), (61, 3, 2, 4))
    except ValueError:
        pass
    else:
        raise AssertionError('wrong tensor stride accepted')
finally:
    os.close(fd)
print('FISHER_RANGE_SYNTHETIC_PASS')
