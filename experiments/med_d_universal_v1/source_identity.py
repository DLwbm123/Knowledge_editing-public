"""Extract a concrete dataset-qualified group without reading question or answer."""
import re

_SLAKE = re.compile(r'(?i)(?<![a-z0-9])xmlab([0-9]+)(?![a-z0-9])')
_VQA_FILE = re.compile(r'(?i)(?<![a-z0-9])synpic([0-9]+)\.(?:jpg|jpeg|png)(?![a-z0-9])')


def group(source_group, image_path):
    texts = [str(x) for x in (source_group, image_path) if x]
    slake = {'slake:xmlab' + str(int(m.group(1))) for s in texts for m in _SLAKE.finditer(s)}
    vqa = {'vqa-rad:synpic' + str(int(m.group(1))) for s in texts for m in _VQA_FILE.finditer(s)}
    if vqa and not any(re.search(r'(?i)vqa[-_]?rad', s) for s in texts):
        raise ValueError('VQA dataset identity absent')
    identities = slake | vqa
    if len(identities) != 1:
        raise ValueError('Missing or conflicting exact source identity')
    return next(iter(identities))


def require_same(reference, rows):
    expected = group(reference.get('source_group'), reference.get('image_path'))
    if not rows:
        raise ValueError('No source rows')
    for row in rows:
        if group(row.get('source_group'), row.get('image_path')) != expected:
            raise ValueError('Source/FIT exact group mismatch')
    return expected
