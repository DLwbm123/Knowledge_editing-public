"""Deterministic frozen selection, without model imports."""
from pathlib import Path


def select_records(rows,manifest,pilot,data_root):
    blocked=set(manifest['calibration_source_groups'])|set(manifest['excluded_source_groups'])
    blocked.add('SLAKE:'+pilot['relative_image_path'].split('/')[0])
    ids=set(manifest['calibration_ids'])|set(manifest['excluded_ids'])|{pilot['record_id']}
    selected=[];seen=set()
    for row in sorted(rows,key=lambda r:int(r['qid'])):
        group='SLAKE:'+row['img_name'].split('/')[0]
        rid='train.SLAKE.'+str(row['qid'])
        if row['q_lang']!='en' or group in blocked or group in seen or rid in ids:continue
        selected.append(dict(record_id=rid,dataset='SLAKE',question=row['question'],target=row['answer'],
            official_rephrase=row['question'],image_path=str(Path(data_root)/'imgs'/row['img_name']),
            relative_image_path=row['img_name'],formal_sequence_position=len(selected)+1,
            question_type='TRAINING_BATCH_DIAGNOSTIC',router_positive_source='TRAIN_ONLY_NO_EDIT_LABELS'))
        seen.add(group)
        if len(selected)==8:break
    if len(selected)!=8:raise ValueError('Fewer than eight eligible image groups; do not relax selection')
    if any(not r['target'].strip() for r in selected):raise ValueError('Selected target is empty; do not replace')
    return selected

