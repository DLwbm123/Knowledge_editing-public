"""Test packet binding rejection with synthetic files only."""
import json
import os
from pathlib import Path
import tempfile
import med_r2_score_packet as packet


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


with tempfile.TemporaryDirectory() as temp:
    home = Path(temp)
    root, parent, original = (home / v for v in ('candidate', 'parent', 'original'))
    os.environ.update(EXPERIMENT_ROOT=str(root), PARENT_RUN=str(parent), REFERENCE_MED_RUN=str(original))
    owners = (47, 117, 93, 87)
    plan, tasks = [], []
    for owner in owners:
        qid = f'q{owner}'
        query = dict(query_id=qid, question=f'question{owner}', reference='reference')
        row = dict(method='MED', mode='single', prefix=1, owner=owner, query=query)
        plan.append(row)
        tasks.append(dict(order=owner))
        cid = json.dumps(['MED', 'single', 1, owner, qid], separators=(',', ':'))
        binding = dict(question=query['question'], reference='reference', image_sha256='image',
                       prompt_ids=[1], attention_mask=[1], runtime='runtime', generation='generation')
        output = dict(binding=dict(judge_input=binding), R0=dict(raw_token_ids=[2], raw_answer='answer'),
                      active_target=False)
        save(root/'outputs'/(packet.digest(cid)+'.json'), output)
        save(original/'outputs'/(packet.digest(cid)+'.json'), output)
        save(parent/'private/outputs/BASE_base_p0/0'/(packet.digest(qid)+'.json'), output)
    save(root/'ROLE_ADMISSION.json', dict(status='PASS'))
    save(root/'STAGE_A_GENERATION_COMPLETE.json', {})
    save(root/'PROTOCOL_LOCK.json', dict(single_consumers=4, input_binding='binding'))
    save(root/'CLOCK.json', dict(hard_end=1))
    save(root/'INPUTS_FROZEN.json', dict(tasks=tasks, plan=plan, base_mask={}))
    assert len(packet.gather('A')['consumers']) == 12
    prior = original/'outputs'/(packet.digest(json.dumps(['MED', 'single', 1, 47, 'q47'], separators=(',', ':')))+'.json')
    altered = packet.read(prior)
    altered['binding']['judge_input']['prompt_ids'] = [999]
    save(prior, altered)
    try:
        packet.gather('A')
    except AssertionError:
        pass
    else:
        raise AssertionError('mismatched prompt was accepted')
print('SYNTHETIC_PACKET_PASS')
