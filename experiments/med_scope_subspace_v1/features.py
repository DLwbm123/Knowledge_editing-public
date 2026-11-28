"""Private role bindings and Base-only acquisition at the edited layer."""
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path

import torch


DEV = {'native', 'fit', 'train_eq', 'train_ind', 'train_vis', 'bg', 'val_pos', 'val_neg'}
OFFICIAL = {'official_ind','official_paraphrase','official_numeric','official_distractor'}
TEST = OFFICIAL | {'cal_ind', 'cal_vis', 'hold_pos', 'hold_ind', 'hold_vis', 'official_ind', 't1g', 't2l', 'ood'}
POSITIVES = OFFICIAL | {'native', 'fit', 'train_eq', 'val_pos', 'hold_pos', 'official_ind', 't1g'}
VISUAL_NEGATIVE = {'black', 'strong_noise', 'strong_blur'}


def load_manifest(path):
    plan = json.loads(Path(path).read_text())
    rows = plan['rows']
    if not rows or not plan['protected_image_sha256']:
        raise ValueError('An explicit stage-5 exclusion inventory is required')
    permitted = DEV | TEST
    excluded = set(plan['protected_image_sha256'])
    if any(r['role'] not in permitted for r in rows):
        raise ValueError('Unknown event role')
    if any((r.get('owner') is not None) == (r['role'] in {'bg','ood'}) for r in rows):
        raise ValueError('Expert ownership does not match the role')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Repeated event ID')
    if any({r['source_sha256'], r['image_sha256']} & excluded for r in rows):
        raise ValueError('Stage-5 image in run manifest')
    edited_images = {r['source_sha256'] for r in rows if r['role'] == 'native'}
    forbidden_background = edited_images | set(plan.get('development_image_sha256', []))
    if any(r['source_sha256'] in forbidden_background or r['image_sha256'] in forbidden_background
           for r in rows if r['role'] in {'bg', 'ood'}):
        raise ValueError('Background/OOD overlaps development images')
    examples = {}
    for r in rows:
        event = (r['source_sha256'], ' '.join(r['question'].casefold().split()),
                 r.get('transform', 'identity'), r['image_sha256'])
        old = examples.setdefault(event, r['role'])
        if old != r['role']:
            raise ValueError('Same input bound to different roles')
        if r['role'] in POSITIVES and not r.get('answer'):
            raise ValueError('Positive answer missing')
        if r['role'] in {'train_ind','cal_ind','hold_ind'} and r.get('relation') != 'independent':
            raise ValueError('Independent negatives need a relation label')
        family = r.get('transform', 'identity')
        if r['role'] == 'train_eq' and family != 'identity':
            if not plan.get('t1g_transforms') or family in set(plan['t1g_transforms']) | VISUAL_NEGATIVE:
                raise ValueError('Positive image transform conflicts with T1G/negative contract')
            if not r.get('answer_preserved'):
                raise ValueError('Augmented positive needs an answer-preservation decision')
        if r['role'] == 'train_vis' and family not in {'black', 'strong_noise'}:
            raise ValueError('Visual training transform not admitted')
        if r['role'] in {'cal_vis', 'hold_vis'} and family != 'strong_blur':
            raise ValueError('Visual evaluation transform not admitted')
    calibration_owners = {r['owner'] for r in rows if r['role'] in {'cal_ind','cal_vis'}}
    if calibration_owners != set(plan.get('calibration_owners',[])):
        raise ValueError('Calibration owner roster differs from calibration events')
    if any(r['owner'] in calibration_owners for r in rows if r['role'] in {'hold_ind','hold_vis'}):
        raise ValueError('A calibration owner also contributed a held-out negative')
    calibration = {r['source_sha256'] for r in rows if r['role'].startswith('cal_')}
    evaluation = {r['source_sha256'] for r in rows if r['role'] in {'hold_ind','hold_vis','ood'}}
    if calibration & evaluation:
        raise ValueError('Calibration and evaluation negatives share an image group')
    if plan.get('train_families') is None or plan.get('holdout_families') is None:
        raise ValueError('Explicit training and held-out template families required')
    if set(plan['train_families']) & set(plan['holdout_families']):
        raise ValueError('Train and held-out template families overlap')
    for r in rows:
        if r['role'] == 'hold_pos' and r.get('family') not in plan['holdout_families']:
            raise ValueError('Undeclared held-out template family')
        if r['role'] == 'train_eq' and r.get('family') not in plan['train_families']:
            raise ValueError('Undeclared training template family')
    parity = plan.get('projection_parity',{})
    if parity:
        dev_ids = {x['id'] for x in rows if x['role'] in {'native','fit','train_eq','val_pos'}}
        if len(parity)!=5 or not set(parity)<=dev_ids:
            raise ValueError('Projection parity needs five locked positive development events')
    return plan


@contextmanager
def capture(layer):
    found = []
    handle = layer.register_forward_pre_hook(lambda _, args: found.append(args[0].detach()))
    try:
        yield found
    finally:
        handle.remove()


def normalized_positions(x, labels):
    answer = torch.where(labels[0] != -100)[0]
    if not answer.numel() or int(answer[0]) == 0 or answer.numel() != int(answer[-1]-answer[0]+1):
        raise ValueError('Expected a contiguous one-example teacher mask')
    x = x[0, int(answer[0])-1:int(answer[-1])].float()
    return x / (x.square().mean(-1, keepdim=True).sqrt()+1e-6)


def acquire(runtime, plan, part, shard=0, shards=1, roles=None):
    allowed = DEV if part=='development' else TEST
    if roles:
        if not set(roles)<=allowed:
            raise ValueError('Requested feature role is outside the declared partition')
        allowed = set(roles)
    rows = [r for r in plan['rows'] if r['role'] in allowed]
    data = {}
    parity = plan.get('projection_parity',{})
    checked_zero = False
    with torch.no_grad():
        for index, r in enumerate(rows[shard::shards], 1):
            runtime.budget()
            prefix = 'reference' if r['role'] in POSITIVES else 'base'
            batch = runtime.teacher_batch(r, prefix)
            with capture(runtime.layer) as found:
                logits = runtime.forward(batch)
            if len(found) != 1:
                raise RuntimeError('Edited layer was not called exactly once')
            feature = normalized_positions(found[0], batch['labels'])
            row = dict(owner=r.get('owner'), role=r['role'], image=r['source_sha256'],
                       feature=feature.half().cpu())
            if r['id'] in parity:
                from .scope_solver import row_basis
                expert = runtime.expert(r['owner'],runtime.checkpoint('MED',r['owner']))
                actual = float((feature[0] @ row_basis(expert.factors()[1]).T).norm())
                if abs(actual-parity[r['id']])>5e-4:
                    raise ValueError('Frozen MED projection parity failed')
                row['projection_parity'] = True
            if r['role']=='native' and not checked_zero:
                from .train_value import mounted
                expert = runtime.expert(r['owner'])
                with mounted(runtime,expert) as hook:
                    hook.set_teacher_routing(batch['labels'])
                    if not torch.equal(logits,runtime.forward(batch)):
                        raise ValueError('Actual zero-residual logits differ from Base')
                checked_zero = True
                row['zero_residual_parity'] = True
            data[r['id']] = row
            if index == 1 or index % 20 == 0:
                print('FEATURES', index, flush=True)
    return data


def read_features(files):
    result = {}
    for file in files:
        part = torch.load(file,map_location='cpu',weights_only=True)
        if set(result) & set(part):
            raise ValueError('Duplicate feature ID across shards')
        result.update(part)
    return result


def prepare_image(source, target, family, seed):
    """Frozen transforms. Save the resulting SHA with the private manifest."""
    from PIL import Image, ImageFilter
    with Image.open(source) as handle:
        im = handle.convert('RGB')
    if family == 'black':
        im = Image.new('RGB', im.size)
    elif family == 'strong_noise':
        import numpy as np
        rng = np.random.default_rng(seed)
        im = Image.fromarray(rng.integers(0, 256, (im.height, im.width, 3), dtype=np.uint8))
    elif family == 'strong_blur':
        im = im.filter(ImageFilter.GaussianBlur(radius=max(1, min(im.size) / 8)))
    elif family == 'gamma':
        im = im.point([round(255 * (i / 255) ** 1.15) for i in range(256)] * 3)
    else:
        raise ValueError(f'Unknown transform: {family}')
    with Path(target).open('xb') as out:
        im.save(out, format='PNG')
    return sha256(Path(target).read_bytes()).hexdigest()
