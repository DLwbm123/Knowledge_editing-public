"""Freeze accepted Base masks and source-only support on the relocated input pool."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.prepare_stage2_sources import normalized, reviewed_attribute
from scripts.medtrace.stage13r_sources import canonical, conflict
from scripts.medtrace.stage15_sources import paraphrases
from scripts.medtrace.stage17_contract import support_masks
from scripts.medtrace.stage17_prepare import digest, lines


def relocate(path, project):
    original = Path(path)
    if len(original.parts) < 4 or original.parts[:2] != ('/', 'remote-home'):
        raise ValueError('Unregistered source root')
    # The owner component comes from the already-accepted private input binding,
    # not from a public hard-coded account name.
    prefix = Path(*original.parts[:3])
    result = project/'imported'/original.relative_to(prefix)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def accepted(bindings, verdicts):
    """Do not use a verdict detached from its exact accepted opaque binding."""
    if len(verdicts) != len(bindings) or len({r['opaque_query_id'] for r in verdicts}) != len(bindings):
        raise ValueError('Incomplete or duplicate accepted verdicts')
    result = {}
    for v in verdicts:
        b = bindings[v['opaque_query_id']]
        if (v['query_id'] != b['query_id'] or type(v['is_correct']) is not bool
                or v['protocol'] != b['judge']['protocol']
                or v['judge_model'] != b['judge']['model']
                or v['immutable_snapshot'] != b['judge']['immutable_snapshot']
                or digest(b) != v['opaque_query_id'] or b['query_id'] in result):
            raise ValueError('Judge/input identity mismatch')
        result[b['query_id']] = v['is_correct']
    return result


def freeze(project, run, report):
    p = run/'private'
    (run/'public').mkdir(parents=True, exist_ok=True)
    destination = p/'COHORT_AND_SUPPORT_LEDGER.json'
    if destination.exists():
        raise FileExistsError('Preserve the frozen queue; do not select again after outputs')
    bindings = read(p/'BINDINGS.json')
    c0 = accepted(bindings, lines(p/'VERDICTS_ASTRA.jsonl'))
    role = read(p/'ROLE_BOUNDARY_JOIN.json')
    overlay = read(p/'SOURCE_OVERLAY.json')
    events = [e for e in role['retained_events'] if c0[e['edit_query_id']] is False]
    needed = {e['edit_query_id'] for e in events} | {q for e in events for q in e['all_probe_query_ids']}
    rows = {}
    for opaque, b in bindings.items():
        if b['query_id'] not in needed:
            continue
        path = relocate(b['image_path'], project)
        dataset = 'SLAKE' if '/SLAKE/' in b['image_path'] else 'VQA-RAD'
        group = canonical(dataset, b['image_path'])
        rows[b['query_id']] = dict(query_id=b['query_id'], dataset=dataset,
            image_path=str(path), original_image_path=b['image_path'],
            image_sha256=b['image_sha256'], source_group=':'.join(group),
            question=b['question'], reference=b['reference'], opaque_Base_id=opaque,
            base_correct=c0[b['query_id']])
    # No old verdicts, sealed inputs or new source discovery.
    if not needed <= set(role['required_query_ids']):
        raise ValueError('Queue expanded outside the score-independent role join')
    excluded = {tuple(g) for g in role['reserved_source_groups']}
    if any(canonical(rows[q]['dataset'], rows[q]['image_path']) in excluded for q in needed):
        raise ValueError('Preserved reservation entered the queue')
    # One cheap decode/readability check per actually required image; no bulk hashes.
    from PIL import Image
    images = {rows[q]['image_path'] for q in needed}
    for path in sorted(images):
        with Image.open(path) as im:
            im.verify()
    allowed = {(r['dataset'], str(r['source_qid']), r['source_group'])
               for r in role['support_candidate_source_ids']}
    support = [dict(r, original_image_path=r['image_path'],
                    image_path=str(relocate(r['image_path'], project)))
               for r in overlay['rows']
               if (r['dataset'], str(r['source_qid']), r['source_group']) in allowed]
    if len(support) != len(allowed):
        raise ValueError('Missing or duplicate source support identities')
    raw = {str(r['qid']): r for r in read(project/'data/SLAKE/train.json') if r.get('q_lang') == 'en'}
    for r in support:
        source = raw[str(r['source_qid'])]
        if (source['question'], source['answer'], canonical('SLAKE', source['img_name'])) != (
                r['question'], r['reference'], canonical('SLAKE', r['image_path'])):
            raise ValueError('Relocated train QA differs from the reviewed source')
        with Image.open(r['image_path']) as im:
            im.verify()
    support.sort(key=lambda r: (r['dataset'], str(r['source_qid'])))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(project/'models/medical_vlms/llava_med_v1_5_mistral_7b',
                                               use_fast=False, local_files_only=True)
    token_length = lambda r: len(tokenizer.encode(r['reference'], add_special_tokens=False)) + 1
    by_native = defaultdict(list)
    for e in events:
        by_native[e['edit_query_id']].append(e)
    main = [e['edit_query_id'] for e in events if e['task'] == 'T0']
    order = list(dict.fromkeys(main + [e['edit_query_id'] for e in events]))
    eval_questions = {normalized(rows[q]['question']) for e in events if e['task'] != 'T0'
                      for q in e['all_probe_query_ids']}
    tasks = []
    for index, q in enumerate(order, 1):
        n = rows[q]
        attr = reviewed_attribute(n['question'])
        fits = paraphrases(n['question'])
        fit_ok = len(fits) == 4 and not any(normalized(f) in eval_questions for f in fits)
        h = [r for r in support if conflict(n, r)][:1]
        # U is a Base-preservation input, not an H proposition proof. Reuse the
        # existing unrelated-source rule; an unknown native family is not a new
        # eligibility gate for NO_H. All these images are globally role-isolated.
        u = [r for r in support if reviewed_attribute(r['question']) not in (None, attr)
             and normalized(r['question']) != normalized(n['question'])][:1]
        candidates = [r for r in support if attr is not None
                      and reviewed_attribute(r['question']) not in (None, attr)
                      and r not in u]
        g = []
        if h and candidates:
            # Frozen existing same-H-image, answer-type, answer-token-distance rule.
            g = [min(candidates, key=lambda r: (
                r['source_group'] != h[0]['source_group'],
                (normalized(r['reference']) in ('yes', 'no')) != (normalized(h[0]['reference']) in ('yes', 'no')),
                abs(token_length(r)-token_length(h[0]))))]
        locality = list(dict.fromkeys(probe for e in by_native[q] if e['task'].endswith('L')
                                     for probe in e['all_probe_query_ids']))
        heval = [rows[probe] for probe in locality if conflict(n, rows[probe])
                 and canonical(rows[probe]['dataset'], rows[probe]['image_path'])
                 not in {canonical(r['dataset'], r['image_path']) for r in h}][:2]
        tasks.append(dict(edit_id=q, order=index, native=n, events=by_native[q],
            fit_questions=fits, fit_status='SUPPORTED' if fit_ok else 'EVALUATION_TEXT_COLLISION',
            H_fit=h, U_fit=u, G=g, H_eval=heval,
            roles=dict(U_fit=bool(u) and fit_ok, H_fit=bool(h) and fit_ok,
                       G=bool(g) and fit_ok, H_eval=bool(heval)),
            support_rule='existing finite reviewed proposition families; no new clinical signoff',
            missing_reason='NO_VERIFIED_NATIVE_PROPOSITION' if attr is None else 'ABSENT_ROLE_IN_FROZEN_TRAIN_POOL',
            seed=int(digest([20260912, q])[:8], 16)))
    masks = support_masks(tasks)
    inputs = defaultdict(list)
    for q in order:
        n = rows[q]
        inputs[(n['image_sha256'], normalized(n['question']))].append(q)
    duplicates = [qs for qs in inputs.values() if len(qs) > 1]
    summary = dict(status='COHORT_AND_SUPPORT_FROZEN', main_T0_N=len(main),
        unique_task_specific_natives=len(order), independent_task_events=dict(Counter(e['task'] for e in events)),
        task_specific_counts={task: dict(edits=sum(e['task']==task for e in events),
            probes=sum(len(e['all_probe_query_ids']) for e in events if e['task']==task),
            Base_correct_probes=sum(c0[q] for e in events if e['task']==task for q in e['all_probe_query_ids']))
            for task in sorted({e['task'] for e in events})},
        support_masks_all_task_natives={key:sum(v is True for v in values.values()) for key,values in masks.items()},
        support_masks_main_T0={key:sum(values[q] is True for q in main) for key,values in masks.items()},
        reviewed_train_QA=len(support), reviewed_train_source_groups=len({r['source_group'] for r in support}),
        verified_evaluation_images=len(images), accepted_Base_verdicts=len(c0),
        duplicate_complete_native_input_groups=len(duplicates),
        order='prior T0 amended order; frozen candidate order within other task-specific queues',
        all_task_natives_are_NOT_one_common_N=True, confirmed_unseen=False, T5='NOT_AUTHORIZED_NA',
        human_clinical_signoff=False, old_experiments_rerun=False, source_downloads=0,
        Base_masks_frozen_before_students=True,
        support_selection_used_student_outputs=False,
        source_pool='29 previously role-isolated reviewed train QA; no rescan of exhausted sources',
        initial_dispatch='BE single main T0 queue only; all other task queues/methods/sequential remain registered pending',
        storage_note='594 full BE matrix checkpoints exceed disk capacity; first dispatch limited to main T0, not a scientific exclusion')
    ledger = dict(summary=summary, main_T0=main, tasks=tasks, events=events,
        queries={q:rows[q] for q in needed}, Base_correctness=c0, masks=masks,
        duplicates=duplicates, support_source_provenance=overlay['provenance'])
    ledger['freeze_id'] = digest(ledger)
    write_new(destination, ledger)
    write_new(run/'public/COHORT_AND_SUPPORT_SUMMARY.json', dict(summary, freeze_id=ledger['freeze_id']))
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        write_new(report, dict(summary, freeze_id=ledger['freeze_id']))
    import json
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', required=True, type=Path)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    freeze(a.project, a.run, a.report)
