"""Freeze a four-edit development panel and a conservative source-only mapping."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    root, parent = Path(os.environ['EXPERIMENT_ROOT']), Path(os.environ['PARENT_RUN'])
    assert not root.exists(), 'A prepared experiment must never be overwritten'
    parent.stat()
    root.mkdir(parents=True)
    probe = root / 'write_probe'
    probe.write_text('132')
    assert probe.read_text() == '132'
    probe.unlink()
    assert os.stat(root).st_dev == os.stat(parent).st_dev
    assert os.statvfs(root).f_bavail * os.statvfs(root).f_frsize > 8 * 1024**3 + 2 * 1024**3
    sys.path[:0] = [str(parent/'private/tools'), str(parent/'private/source'),
                    str(parent/'private/official_llava')]
    import module_run as r
    import trajectory
    basis, _ = r.q.split()
    assert len(basis) == 61
    packet = json.loads((parent/'private/M13_SOURCE_PACKET.json').read_text())
    key = 'medical-proposition-small-v132:20261026:'
    tasks = sorted(packet['original_tasks'],
                   key=lambda t: hashlib.sha256((key+t['edit_id']).encode()).hexdigest())[:4]
    assert [t['order'] for t in tasks] == [47, 117, 93, 87]
    assert len({t['native']['source_group'] for t in tasks}) == 4
    mapping, unmapped = [], []
    for index, row in enumerate(basis):
        match = re.fullmatch(r'Does the picture contain ([a-z ]+)\?', row['question'])
        if match and row['reference'] in ('Yes', 'No'):
            organ = match[1]
            assert row['source_annotation']['content_type'] == 'Organ'
            assert row['semantic_correct'] and row['source_role'] == 'train'
            saved = r.c.read(row['response_path'])
            assert saved['unedited_Base'] and saved['binding']['input']['query_id'] == row['query_id']
            yes = [f'Yes, the image contains the {organ}.', f'The {organ} is present in this image.']
            no = [f'No, the image does not contain the {organ}.', f'The {organ} is absent from this image.']
            source = r.c.read(row['source_file'])
            matches = [v for v in source if v.get('qid') == row['source_qid']]
            assert len(matches) == 1 and matches[0]['answer'] == row['reference']
            assert matches[0]['question'] == row['question']
            mapping.append(dict(index=index, row=row, classes=[yes,no] if row['reference']=='Yes' else [no,yes],
                subject='depicted_image', relation='anatomical_structure_visible', value=organ,
                true_present=row['reference']=='Yes', source_annotation_checked=True,
                Base_full_response_review='reviewed; anatomy, modality and location agree with source',
                competitor_rule='Explicit opposite of a source-annotated binary presence answer; no synthetic case',
                alias_review='Both aliases state the same image-level presence, without patient-level absence'))
        else:
            unmapped.append(dict(index=index, query_id=row['query_id'],
                reason='Outside the frozen conservative binary anatomical-presence mapping; common legacy FULL'))
    assert len(mapping) == 31 and len(unmapped) == 30
    assert len({v['row']['source_group'] for v in mapping}) == 24
    assert not {v['row']['source_group'] for v in mapping} & {t['native']['source_group'] for t in tasks}
    plan = r.c.read(parent/'private/EVAL_PLAN146.json')
    consumers = []
    for task in tasks:
        original = [v for v in plan if v['method']=='M13P' and v['mode']=='single' and v['owner']==task['order']]
        assert original
        for row in original:
            q = row['query']
            roles = [e['task'] for e in task['events'] if q['query_id'] in e['all_probe_query_ids']]
            if q['query_id'].startswith('edit_'):
                roles = ['FIT']
            assert roles, q['query_id']
            consumers.append(dict(owner=task['order'], query=q, panels=roles,
                original_Base_qualified=packet['base_mask'].get(q['query_id'],False),
                legacy_output=str(parent/'private/outputs'/f'M13P_single_p1' /
                    str(task['order'])/(r.c.digest(q['query_id'])+'.json'))))
        assert (parent/'private/candidates/C'/f"{task['order']}.pt").is_file()
        assert (parent/'private/results/C'/f"{task['order']}.json").is_file()
        trajectory.source_rows(packet, task['order'])
    assert len(consumers) == 69
    for v in consumers:
        assert Path(v['legacy_output']).is_file()
    frozen = dict(tasks=tasks, mapping=mapping, unmapped=unmapped, consumers=consumers,
                  source_rows={str(t['order']):trajectory.source_rows(packet,t['order']) for t in tasks})
    write(root/'INPUTS_FROZEN.json', frozen)
    write(root/'PROTOCOL_LOCK.json', dict(
        experiment='medical_proposition_small4_v132', seed=20261026, GEN_seed=20261026132,
        arms=['FULL_CTX','MED','GEN'], owners=[t['order'] for t in tasks], steps=160,
        training_updates=1920, new_generations=207, expected_consumers_per_arm=69,
        selection_rule=key+' + original edit_id, SHA256 ascending, first4; no outcomes',
        input_binding=r.c.digest(frozen), mapping_binding=r.c.digest(mapping),
        numeric=dict(rank_rtol=1e-10, PSD_rtol=1e-7, trace_rtol=3e-5,
                     derivative_rtol=.03, derivative_atol=2e-4, degenerate_trace=1e-20),
        scopes=dict(data='original exposed146 DEV subset', independent_confirmation=False,
                    medical_claim='Anatomical presence only, not diagnosis or clinical safety'),
        budgets=dict(max_elapsed_seconds=21600, new_bytes_limit=2_000_000_000,
                     global_managed_new_bytes_limit=50_000_000_000, minimum_free_bytes=8*1024**3,
                     forwards=230000, backwards=5500, updates=1920, operator_backwards=1920,
                     generations=215, vision_forwards=1000),
        checkpoints=dict(owned_final=12, final_consumers='save/load verification and 69 full answers per arm',
                         resume='one latest checkpoint per trajectory',
                         deletion='only explicit owned file list after all declared consumers'),
        scientific_stops=['invalid source mapping', 'no effective medical intervention',
                          'mask/derivative/scale mismatch', 'nonfinite calibration', 'budget'],
        decision='Report all four edits; no automatic146 expansion or performance-based recipe changes',
        monitoring='OFF_BY_USER; no recurring monitoring task', prepared_time=time.time()))
    write(root/'DATA_ADMISSION.json', dict(status='PASS', basis=61, mapped=31, common_fallback=30,
        source_groups=24, relation_types=['anatomical_structure_visible'],
        evaluation_used_for_mapping=False, sealed_V111_opened=False,
        full_Base_correctness='Conservative manual source-only review of these31 original complete responses',
        source_annotation_not_clinician_review=True, mapping_binding=r.c.digest(mapping)))
    print(json.dumps(dict(status='PREPARED', owners=[t['order'] for t in tasks], mapped=31,
                          fallback=30, consumers=69, root=str(root))))


if __name__ == '__main__':
    main()
