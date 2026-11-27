"""Reserve SLAKE image groups by metadata, without opening any answer or running a model."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import sys

from PIL import Image, ImageOps


GROUP = re.compile(r'xmlab\d+', re.I)


def group(value):
    match = GROUP.search(value)
    return match.group().casefold() if match else None


def fingerprint(pixels):
    gray = ImageOps.grayscale(pixels).resize((9, 8), Image.Resampling.LANCZOS)
    values = list(gray.getdata())
    comparisons = [
        values[9 * row + col] > values[9 * row + col + 1]
        for row in range(8) for col in range(8)
    ]
    return sum(v << i for i, v in enumerate(comparisons))


def reserve(inputs, root, output):
    exposure = set(inputs['history_groups'])
    ledger = Path(inputs['role_ledger'])
    for event in json.loads(ledger.read_text()):
        if event['dataset'].casefold() == 'slake':
            image = group(event['image_id'])
            if image:
                exposure.add(image)
    questions = defaultdict(list)
    split_roles = defaultdict(set)
    image_paths = {}
    for split in ('train', 'validation', 'test'):
        for row in json.loads((root / f'{split}.json').read_text()):
            image = group(row['img_name'])
            image_paths[image] = root / 'imgs' / row['img_name']
            split_roles[image].add(split)
            if row['q_lang'] == 'en':
                questions[image].append((row['qid'], row['content_type'].casefold(), row['answer_type'].casefold()))
    assert None not in image_paths
    excluded_sha = set(inputs['all_known_encoded_images'])
    excluded = set(exposure)
    matched = exposure & image_paths.keys()
    signatures = []
    for name in sorted(matched):
        path = image_paths[name]
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB')
            signatures.extend([fingerprint(image), fingerprint(ImageOps.mirror(image))])
    eligible = []
    count = Counter()
    rng = random.Random(20261129)
    order = sorted(questions)
    rng.shuffle(order)
    for name in order:
        if name in excluded:
            count['recorded_history_or_reserved'] += 1
            continue
        path = image_paths[name]
        if sha256(path.read_bytes()).hexdigest() in excluded_sha:
            count['encoded_duplicate'] += 1
            continue
        items = sorted(set(questions[name]))
        edit_ids = [qid for qid, category, form in items if category == 'abnormality' and form == 'open']
        independent_ids = [qid for qid, category, _ in items if category in ('modality', 'plane')]
        if not edit_ids or not independent_ids:
            count['no_required_metadata_roles'] += 1
            continue
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB')
            normal, mirror = fingerprint(image), fingerprint(ImageOps.mirror(image))
        if any((normal ^ x).bit_count() <= 5 or (mirror ^ x).bit_count() <= 5 for x in signatures):
            count['suspect_near_or_mirror'] += 1
            continue
        signatures.extend([normal, mirror])
        eligible.append(dict(image_group=name, edit_qids=edit_ids, independent_qids=independent_ids,
                             dataset_splits=sorted(split_roles[name])))
    receipt = dict(status='METADATA_ONLY_PENDING_SOURCE_AND_FULL_EXPOSURE_ADMISSION',
                   seed=20261129, shortlisted=len(eligible), english_image_groups=len(questions),
                   shortlisted_test_only=sum(r['dataset_splits'] == ['test'] for r in eligible),
                   metadata_cross_split_groups=sum(len(split_roles[n]) > 1 for n in questions),
                   preexisting_excluded_image_groups=len(exposure),
                   historically_matched_in_source=len(matched), exclusions=dict(count),
                   reads_of_reference_text_by_reviewer=0, generations=0, model_calls=0,
                   patient_identity_verified=False)
    with (output / 'reserve.private.json').open('x') as stream:
        json.dump(dict(receipt, candidates=eligible), stream)
    print(json.dumps(receipt))


if __name__ == '__main__':
    inputs, root, output = (Path(arg) for arg in sys.argv[1:])
    reserve(json.loads(inputs.read_text()), root, output)
