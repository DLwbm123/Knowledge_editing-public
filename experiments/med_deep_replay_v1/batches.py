"""Build CPU replay batches in the already admitted historical FIT order."""
from dataclasses import replace
from pathlib import Path


def prepare_replay(runtime, r, task, rows, role_receipt, budget):
    if role_receipt.get('status') != 'FIT_CONTENT_AUTHORIZED' or task['order'] not in role_receipt.get('owners', []):
        raise ValueError('Independent FIT content admission missing')
    if not rows or any(row.get('role') != 'REPLAY_FIT' for row in rows):
        raise ValueError('Only admitted replay FIT records are permitted')
    if len({row['query_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate replay identity')
    batches = []
    runtime.model.half()
    try:
        for row in rows:
            budget()
            # Match the original SOURCE_REPLAY record construction, including
            # dataset and original image path; never inspect held-out records.
            record = replace(r.c.record(task), question=row['question'], target=row['reference'],
                             image_path=Path(r.c.local_path(row['image_path'])),
                             relative_image_path=row['original_image_path'], official_rephrase='',
                             dataset=row['dataset'])
            batches.append(r.old.cpu(runtime.build_edit_batch(record)))
    finally:
        r.old.fp32(runtime)
    return batches
