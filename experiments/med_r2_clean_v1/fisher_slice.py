"""Read only admitted row ranges from a frozen uncompressed Torch ZIP archive."""
import os
import pickletools
import struct
import zipfile
import numpy as np


def _data_offset(fd, info):
    header = os.pread(fd, 30, info.header_offset)
    if len(header) != 30:
        raise ValueError('short ZIP local header')
    sig, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack('<IHHHHHIIIHH', header)
    if sig != 0x04034B50 or flags & 1 or method != 0 or info.compress_type != 0:
        raise ValueError('unsupported ZIP entry')
    return info.header_offset + 30 + name_len + extra_len


def _counts(pickle_bytes):
    ops = list(pickletools.genops(pickle_bytes))
    globals_seen = [arg for op, arg, _ in ops if op.name == 'GLOBAL']
    if globals_seen != ['torch._utils _rebuild_tensor_v2', 'torch FloatStorage', 'collections OrderedDict']:
        raise ValueError('unexpected storage format')
    start = next((i for i, (op, arg, _) in enumerate(ops)
                  if op.name == 'BINUNICODE' and arg == 'token_counts'), None)
    if start is None:
        raise ValueError('missing token counts')
    numbers = []
    for op, arg, _ in ops[start + 1:]:
        if op.name == 'APPENDS': break
        if op.name in ('BININT', 'BININT1', 'BININT2'): numbers.append(arg)
    else: raise ValueError('unterminated token counts')
    if len(numbers) != 61 or any(not isinstance(n, int) or n <= 0 for n in numbers):
        raise ValueError('invalid original token counts')
    return numbers


def extract_fd(fd, allowed_indices, expected_shape):
    """Return FP32 x/y arrays for exactly the 29 allowed fallback sources."""
    if len(allowed_indices) != 29 or len(set(allowed_indices)) != 29:
        raise ValueError('expected 29 unique permitted fallback indices')
    if any(not isinstance(i, int) or not 0 <= i < 61 for i in allowed_indices):
        raise ValueError('invalid source index')
    with os.fdopen(os.dup(fd), 'rb') as stream, zipfile.ZipFile(stream) as archive:
        infos = archive.infolist()
        def one(suffix):
            hits = [v for v in infos if v.filename.endswith(suffix)]
            if len(hits) != 1: raise ValueError('unexpected archive members')
            return hits[0]
        if archive.read(one('/byteorder')) != b'little':
            raise ValueError('unsupported byte order')
        counts = _counts(archive.read(one('/data.pkl')))
        total, xdim, directions, ydim = expected_shape
        if sum(counts) != total:
            raise ValueError('tensor row count mismatch')
        xinfo, yinfo = one('/data/0'), one('/data/1')
        if (xinfo.file_size != total * xdim * 4 or
                yinfo.file_size != total * directions * ydim * 4):
            raise ValueError('tensor byte size mismatch')
        xstart, ystart = _data_offset(fd, xinfo), _data_offset(fd, yinfo)
        offsets = [0]
        for n in counts: offsets.append(offsets[-1] + n)
        blocks = []
        for i in sorted(allowed_indices):
            n, offset = counts[i], offsets[i]
            nx, ny = n * xdim * 4, n * directions * ydim * 4
            xb = os.pread(fd, nx, xstart + offset * xdim * 4)
            yb = os.pread(fd, ny, ystart + offset * directions * ydim * 4)
            if len(xb) != nx or len(yb) != ny:
                raise ValueError('short admitted storage range')
            blocks.append(dict(index=i, x=np.frombuffer(xb, dtype='<f4').reshape(n, xdim).copy(),
                               y=np.frombuffer(yb, dtype='<f4').reshape(n, directions, ydim).copy()))
        return blocks


def extract_from_lock(fd, lock, expected_shape):
    if lock.get('label') != 'IDENTITY_LOCK_ONLY':
        raise ValueError('source identity lock missing')
    kept = lock['keep']
    selected = [v['index'] for v in kept if v['kind'] == 'fallback']
    excluded = {v['index'] for v in lock['excluded']}
    if excluded.intersection(selected):
        raise ValueError('excluded source requested')
    return extract_fd(fd, selected, expected_shape)
