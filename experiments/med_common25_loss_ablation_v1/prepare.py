"""Select the authorized training records and freeze bank identities privately."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
import torch
import selected_records

old = Path(os.environ['PRIOR_RUN'])
new = Path(os.environ['EXPERIMENT_ROOT'])
deadline = float(os.environ['EXPERIMENT_DEADLINE'])


def write(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(value, f, ensure_ascii=False, separators=(',', ':'))


def digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    if time.time() >= deadline - 600:
        raise TimeoutError('original deadline')
    resource.setrlimit(resource.RLIMIT_FSIZE, (256_000_000, 256_000_000))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('preparation budget')))
    signal.alarm(min(300, max(1, int(deadline - time.time() - 120))))
    fs = os.statvfs(new.parent.parent)
    if fs.f_bavail * fs.f_frsize < 8 * 1024 ** 3:
        raise OSError('insufficient free space')
    d, du = old / 'D', old / 'DU'
    admission = json.loads((d / 'ROLE_ADMISSION.json').read_text())
    weak = json.loads((d / 'SPAN_LOCK.private.json').read_text())
    tokens = json.loads((d / 'TOKEN_LOCK.private.json').read_text())
    if admission.get('status') != 'TRAINING_SOURCE_ROLE_ADMITTED' or len(set(admission['owners'])) != 101:
        raise ValueError('source admission differs')
    if weak.get('status') != 'WEAK_TOKEN_PROXY_ONLY' or [x['order'] for x in weak['records']] != admission['owners']:
        raise ValueError('weak source lock differs')
    owners = {x['order'] for x in weak['records'] if x['status'] == 'WEAK_TOKEN_MATCH'}
    if len(owners) != 25 or not owners.issubset(admission['owners']):
        raise ValueError('weak coverage differs')
    if [x['order'] for x in tokens['records']] != admission['owners'] or any(x['status'] != 'TOKEN_BOUND' for x in tokens['records']):
        raise ValueError('token lock differs')
    select = lambda path: selected_records.top_array(path, {'task': {'order': True}}, lambda r: r['task']['order'], owners)
    source = select(d / 'TRAIN_SOURCE_BOUND_PRIVATE.json')
    if source != select(du / 'TRAIN_SOURCE_BOUND_PRIVATE.json'):
        raise ValueError('selected D and DU training inputs differ')
    if any([r['audit_role'] for r in row['source_rows']] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT'] for row in source):
        raise ValueError('training source role differs')
    experts = {x['task']['edit_id'] for x in source}
    if len(experts) != 25:
        raise ValueError('expert identities not unique')
    initial = {}
    for branch, label in ((d, 'D'), (du, 'DU')):
        if any(json.loads((branch / f'{phase}_GUARD_EXIT.json').read_text())['code'] != 0 for phase in ('TRAIN', 'EVAL')):
            raise ValueError('historical guard failed')
        results = [row for i in range(4) for row in json.loads((branch / f'TRAIN_COMPLETE_{i}.json').read_text())['results']]
        for owner in owners:
            obj = torch.load(branch / 'weights' / f'{owner}_{label}.pt', map_location='cpu', weights_only=True)
            matched = [x for x in results if x['owner'] == owner]
            if len(matched) != 1 or matched[0]['final_digest'] != digest(obj['state']) or obj['steps'] != 160:
                raise ValueError('historical weight binding differs')
            if owner in initial and initial[owner] != obj['initial_digest']:
                raise ValueError('historical initial states differ')
            initial[owner] = obj['initial_digest']
    fields = {'consumer_id': True, 'owner': True, 'query': {'query_id': True}, 'route': True,
              'effective_expert': True, 'original_filename': True}
    metadata = selected_records.identities(d / 'BANK_INPUTS_PRIVATE.json', 'records', fields)
    if len(metadata) != 2148 or len({x['consumer_id'] for x in metadata}) != 2148:
        raise ValueError('bank identity coverage differs')
    selected = {x['consumer_id'] for x in metadata if x['effective_expert'] in experts}
    if len(selected) != 392:
        raise ValueError('routed event coverage differs')
    detail_fields = dict(fields, query={'query_id': True, 'question': True, 'image_path': True,
                         'image_sha256': True, 'audit_role': True},
                         judge_input={'question': True, 'image_sha256': True, 'prompt_ids': True,
                                      'attention_mask': True, 'generation': True})
    details = selected_records.read_selected(d / 'BANK_INPUTS_PRIVATE.json', 'records', {'consumer_id': True},
                                               lambda r: r['consumer_id'], selected, detail_fields)
    for item in metadata:
        if item['consumer_id'] in selected:
            detail = details[item['consumer_id']]
            if any(detail[k] != item[k] for k in ('consumer_id', 'owner', 'route', 'effective_expert', 'original_filename')) or detail['query']['query_id'] != item['query']['query_id']:
                raise ValueError('selected bank identity changed')
    records = [details.get(item['consumer_id'], item) for item in metadata]
    for arm in ('CE', 'UNIFORM'):
        root = new / arm
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        (root / 'tools').mkdir(mode=0o700)
        for name, value in (('ROLE_ADMISSION.json', admission), ('SPAN_LOCK.private.json', weak),
                            ('TOKEN_LOCK.private.json', tokens), ('TRAIN_SOURCE_BOUND_PRIVATE.json', source),
                            ('BANK_INPUTS_PRIVATE.json', {'status': 'BANK_INPUTS_BOUND', 'records': records, 'candidate_count': 392}),
                            ('STATE_BINDING.private.json', {'status': 'STATE_BINDING_PASS', 'initial_digests': initial})):
            write(root / name, value)
    write(new / 'INPUT_BINDING.private.json', {'status': 'SELECTED_INPUTS_BOUND', 'events': 2148,
          'selected': 392, 'owners': 25, 'original_clock_end': deadline})
    print(json.dumps({'status': 'SELECTED_INPUTS_BOUND', 'owners': 25, 'events': 2148, 'selected': 392}))


if __name__ == '__main__':
    main()
