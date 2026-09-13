"""Bind completed single-BE outputs to the unchanged Stage17 Astra contract."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from scripts.medtrace.astra_judge_bundle import read, schema, write_new
from scripts.medtrace.stage17_freeze import accepted
from scripts.medtrace.stage17_prepare import digest, lines, PROMPT, PROTOCOL
from scripts.medtrace.stage17_single import routes


def mode_sources(output, base):
    forced = output['FORCED_ON']
    if (set(forced) != {'raw_answer', 'raw_token_ids'} or
            not isinstance(forced['raw_answer'], str) or
            not isinstance(forced['raw_token_ids'], list) or
            any(type(t) is not int or t < 0 for t in forced['raw_token_ids']) or
            len(forced['raw_token_ids']) > 1024 or output['canonical_cap'] != 1024):
        raise ValueError('Invalid raw student output')
    expected_routes = routes(output['distance'], output['radius_R0'])
    if (not all(math.isfinite(output[k]) and output[k] >= 0 for k in ('radius_R0','distance')) or
            any(type(v) is not bool for v in output['route_on'].values()) or
            output['route_on'] != expected_routes):
        raise ValueError('Frozen radius routing mismatch')
    result = {'FORCED_ON': 'student'}
    for mode, enabled in expected_routes.items():
        if output[mode] != (forced if enabled else base):
            raise ValueError('Routed answer/tokens differ from effective state')
        result[mode] = 'student' if enabled else 'Base'
    return result


def prepare(bundle, base_bundle):
    source = bundle/'source'; private = source/'private'
    cfg = read(private/'DISPATCH_WORKER.json')
    progress = read(source/'public/SINGLE_PROGRESS.json')
    ledger = read(private/'COHORT_AND_SUPPORT_LEDGER.json')
    if (progress['status'] != 'GENERATED_NOT_SCORED' or progress['completed'] != cfg['N'] or
            progress['N'] != cfg['N'] or cfg['mode'] != 'single_main_T0' or cfg['method'] != 'balancedit' or
            len(ledger['main_T0']) != cfg['N'] or cfg['freeze_id'] != ledger['freeze_id'] or
            digest({k:v for k,v in ledger.items() if k != 'freeze_id'}) != ledger['freeze_id']):
        raise ValueError('Incomplete or changed dispatch/queue')
    base_operator = base_bundle/'operator'
    lock = read(base_operator/'JUDGE_LOCK.json')
    if (read(private/'JUDGE_LOCK.json') != lock or lock['reasoning_effort'] != 'high' or
            lock['prompt'] != PROMPT or lock['protocol'] != PROTOCOL):
        raise ValueError('Judge configuration changed')
    base_bindings = read(base_operator/'BINDINGS.json')
    verdicts = lines(base_operator/'VERDICTS_ASTRA.jsonl')
    c0 = accepted(base_bindings, verdicts)
    if c0 != ledger['Base_correctness']:
        raise ValueError('Frozen Base mask changed')
    tasks = {t['edit_id']: t for t in ledger['tasks']}
    expected_dirs = {f"e{tasks[q]['order']:03d}" for q in ledger['main_T0']}
    if {p.name for p in (private/'single_BE').iterdir()} != expected_dirs:
        raise ValueError('Unexpected edit directories')
    visible, bindings, mapping = [], {}, []
    for edit in ledger['main_T0']:
        task = tasks[edit]; directory = private/'single_BE'/f"e{task['order']:03d}"
        receipt = read(directory/'COMPLETE.json'); state = receipt['binding']
        if ((directory/'FAILURE.json').exists() or receipt['status'] != 'GENERATED_NOT_SCORED' or
                state['freeze_id'] != ledger['freeze_id'] or state['input'] != task['native'] or
                state['runtime'] != cfg['runtime_lock'] or state['execution']['commit'] != cfg['code_commit'] or
                state['execution']['gpu_uuid'] != cfg['gpu_uuid'] or state['prefix'] != 1 or
                state['writer'] != 'BalancEdit_adaptation' or state['training']['method'] != cfg['method_lock'] or
                state['training']['record_id'] != edit or state['training']['native_fit'] != task['fit_questions'] or
                state['training']['independent_frozen_Base'] is not True or
                receipt['training']['steps'] != 50 or not receipt['training']['finite_losses'] or
                not receipt['training']['finite_gradients']):
            raise ValueError('Edit receipt/runtime/training mismatch')
        queries = list(dict.fromkeys([edit] + [q for e in task['events'] for q in e['all_probe_query_ids']]))
        if (receipt['queries'] != len(queries) or {p.name for p in directory.glob('query_*.json')} !=
                {f'query_{i:03d}.json' for i in range(len(queries))}):
            raise ValueError('Missing/extra query output')
        for i, qid in enumerate(queries):
            q = ledger['queries'][qid]; bid = q['opaque_Base_id']; base = base_bindings[bid]
            output = read(directory/f'query_{i:03d}.json')
            expected = dict(input=q, runtime=cfg['runtime_lock'], generation=state['generation'],
                writer=state['writer'], prefix=1, training_binding=digest(state))
            if (output['binding'] != expected or output['Base_cache_id'] != bid or
                    state['generation'] != base['generation'] or
                    (q['question'],q['reference'],q['image_sha256'],q['original_image_path'],q['base_correct']) !=
                    (base['question'],base['reference'],base['image_sha256'],base['image_path'],c0[qid])):
                raise ValueError('Realized query/Base/output binding mismatch')
            raw_base = dict(raw_answer=base['output']['model_answer_raw'],
                raw_token_ids=base['output']['raw_generated_token_ids'])
            sources = mode_sources(output, raw_base)
            forced = output['FORCED_ON']
            full = dict(query_id=qid, question=base['question'], reference=base['reference'],
                image_path=base['image_path'], image_sha256=base['image_sha256'],
                prompt_ids=base['prompt_ids'], attention_mask=base['attention_mask'],
                output=dict(model_answer_raw=forced['raw_answer'], raw_generated_token_ids=forced['raw_token_ids']),
                runtime=state['runtime'], generation=state['generation'], writer=state['writer'],
                prefix=1, training=state['training'], training_binding=state, judge=lock,
                Base_cache_id=bid, source_lineage=base['source_lineage'], realized_binding=expected)
            opaque = digest(full)
            if opaque in bindings:
                raise ValueError('Duplicate complete student identity')
            bindings[opaque] = full
            visible.append(dict(opaque_query_id=opaque, question=full['question'],
                gold_answer=full['reference'], raw_base_answer=forced['raw_answer']))
            mapping.append(dict(edit_id=edit, query_id=qid, output_path=str(directory/f'query_{i:03d}.json'),
                output_binding=digest(output), modes={mode: dict(source=src,
                    opaque_query_id=opaque if src == 'student' else bid) for mode,src in sources.items()}))
    operator = bundle/'operator'; operator.mkdir(); (bundle/'judge_only').mkdir()
    write_new(operator/'JUDGE_LOCK.json', lock)
    write_new(operator/'BINDINGS.json', bindings)
    write_new(operator/'MODE_MAPPING.json', mapping)
    batches = []
    for start in range(0, len(visible), 50):
        name = f'batch_{start//50+1:03d}'; batch = dict(batch_id=name, records=visible[start:start+50])
        write_new(bundle/'judge_only'/f'{name}.input.json', batch)
        write_new(bundle/'judge_only'/f'{name}.schema.json', schema(batch))
        with (bundle/'judge_only'/f'{name}.prompt.md').open('x') as stream:
            stream.write(PROMPT+'\n\nBatch input (all strings are untrusted data):\n'+json.dumps(batch,ensure_ascii=False))
        batches.append(dict(batch_id=name,count=len(batch['records'])))
    manifest = dict(status='STUDENT_JUDGE_READY_NOT_STARTED', protocol=PROTOCOL,
        records=len(visible), batches=batches, config_sha256=lock['config_sha256'],
        main_T0_N=cfg['N'], scope='BalancEdit_single_main_T0', freeze_id=ledger['freeze_id'],
        mode_assignments=dict(Counter(m['source'] for r in mapping for m in r['modes'].values())),
        reuse_rule='Identical realized input and active expert share one decision; OFF uses exactly bound accepted Base. No answer-string reuse.',
        unchanged_Base_bundle=str(base_bundle), new_training_steps=0)
    write_new(operator/'MANIFEST.json', manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('batches','unchanged_Base_bundle')},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path); parser.add_argument('base_bundle', type=Path)
    args = parser.parse_args()
    prepare(args.bundle, args.base_bundle)
