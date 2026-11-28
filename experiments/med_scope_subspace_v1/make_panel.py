"""Convert the frozen Stage-1 panel plus an explicitly admitted supplement."""
import argparse
import json
import os
from pathlib import Path

from .features import load_manifest, prepare_image


def make(stage1, supplement, image_dir):
    entries = stage1['entries']
    owners = {e['owner'] for e in entries}
    cal = set(supplement['calibration_owners'])
    if not cal < owners or not cal:
        raise ValueError('Calibration must be a strict nonempty subset of image owners')
    rows = list(supplement['rows'])
    scheduled = {}
    for e in entries:
        o = e['owner']
        src = e['source']
        native = src['source_rows'][0]
        source = native['image_sha256']
        frame = dict(owner=o,source_sha256=source,image_sha256=source,
                     image_path=native['image_path'],original_image_path=native['original_image_path'],
                     dataset=src['dataset'])
        def add(id,role,question,answer,**extra):
            row = dict(frame,id=f'{o}-{id}',role=role,question=question,answer=answer)
            row.update(extra)
            rows.append(row)
            return row['id']
        source_ids = [add(f'source-{idx}','native' if idx==0 else 'fit',r['question'],r['reference'])
                      for idx,r in enumerate(src['source_rows'])]
        by_text = {rows[-5+i]['question']:key for i,key in enumerate(source_ids)}
        train_views = {}
        for i,v in enumerate(e['views']):
            role = v['role']
            if role=='train':
                if v['question'] in by_text:
                    train_views[i] = by_text[v['question']]
                else:
                    train_views[i] = add(f'expanded-{i}','train_eq',v['question'],v['reference'],family=v['template'])
            elif role in {'validation','holdout_modal','holdout_context'}:
                add(f'expanded-{i}','val_pos' if role=='validation' else 'hold_pos',
                    v['question'],v['reference'],family=v['template'])
        def neg(row,role):
            if row['image_sha256']!=source or row['label']!='independent':
                raise ValueError('Frozen independent negative binding differs')
            continuation = row['base_continuation']
            if continuation['sequence_contract']!='continuation_only':
                raise ValueError('Invalid Base prefix identity')
            return add(row['query_id'],role,row['question'],row.get('gold_answer'),
                       base_tokens=continuation['raw_token_ids'],relation='independent')
        negatives = [neg(r,'train_ind') for r in e['negative']['train']]
        for r in e['negative']['holdout']:
            neg(r,'cal_ind' if o in cal else 'hold_ind')
        for idx,r in enumerate(e.get('historical_t2l',[])):
            add(f'historical-{idx}','t2l',r['question'],r['reference'])
        visual = []
        original = Path(native['image_path'])
        if not original.is_file() and '/DataP/' in str(original):
            original = Path(os.environ['FROZEN_IMAGE_ROOT']) / str(original).split('/DataP/',1)[1]
        for i,family in enumerate(('black','strong_noise','strong_blur')):
            image = image_dir/f'{o}-{i}.png'
            encoded = prepare_image(original,image,family,src['seed'])
            role = 'train_vis' if i<2 else 'cal_vis' if o in cal else 'hold_vis'
            id = add(f'image-{i}',role,native['question'],native['reference'],
                     image_path=str(image),image_sha256=encoded,transform=family)
            if i<2:
                visual.append(id)
        if supplement.get('reserve_visual_validation'):
            image = image_dir/f'{o}-validation.png'
            encoded = prepare_image(original,image,'strong_noise',src['seed']+104729)
            add('validation-image','val_neg',native['question'],native['reference'],
                image_path=str(image),image_sha256=encoded,transform='strong_noise')
        if o in set(supplement.get('gamma_positive_owners',[])):
            image = image_dir/f'{o}-gamma.png'
            encoded = prepare_image(original,image,'gamma',src['seed'])
            add('gamma-image','train_eq',native['question'],native['reference'],
                image_path=str(image),image_sha256=encoded,transform='gamma',
                family='gamma_native',answer_preserved=True)
        scheduled[str(o)] = [dict(native=source_ids[0],
             views=[train_views[i] for i in s['expanded_view_indices']],
             negatives=[negatives[s['independent_train_negative_indices'][0]],visual[(i-1)%len(visual)]])
             for i,s in enumerate(e['steps'],1)]
    combined = {k:v for k,v in supplement.items() if k != 'rows'}
    return dict(combined,rows=rows,schedules=scheduled,stage1_owners=sorted(owners))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--stage1',type=Path,required=True)
    cli.add_argument('--supplement',type=Path,required=True)
    cli.add_argument('--images',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.images.mkdir()
    result = make(json.loads(args.stage1.read_text()),json.loads(args.supplement.read_text()),args.images)
    tmp = args.output.with_suffix('.pending.json')
    with tmp.open('x') as out:
        json.dump(result,out,ensure_ascii=False)
    load_manifest(tmp)
    tmp.rename(args.output)


if __name__=='__main__':
    main()
