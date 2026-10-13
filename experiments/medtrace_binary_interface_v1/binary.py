"""Fixed canonical-answer likelihood decoding; retain TT knowledge on polar queries."""
import math
import os
import time
import traceback
from pathlib import Path
import torch
import paper as p
from question_type import question_type
from methods.medtrace.core import MedTraceLayerHook

c, RUN = p.c, p.RUN
GATE = Path(os.environ['BINARY_GATE_PARENT'])
MARGIN = Path(os.environ['BINARY_MARGIN_PARENT'])
TYPE = Path(os.environ['BINARY_TYPE_PARENT'])
ARMS = ('B_BASE_POLAR_CONTROL', 'B_TT_POLAR')
ANSWERS = ('No', 'Yes')


def select_answer(scores):
    assert set(scores) == set(ANSWERS) and all(math.isfinite(v) for v in scores.values())
    return max(ANSWERS, key=lambda a: scores[a])


def sequence_score(logits, mask, labels):
    logp = logits[mask].float().log_softmax(-1)
    tokens = labels[labels != -100]
    assert len(logp) == len(tokens)
    return float(logp.gather(1, tokens[:, None]).sum())


def prepare():
    assert select_answer({'No': -2., 'Yes': -1.}) == 'Yes'
    assert select_answer({'No': -1., 'Yes': -1.}) == 'No'
    logits = torch.tensor([[[0., 2.], [3., 0.], [0., 0.]]])
    labels = torch.tensor([[-100, 1, 0]])
    mask = torch.tensor([[True, True, False]])
    expected = torch.log_softmax(logits, -1)[0, 0, 1]+torch.log_softmax(logits, -1)[0, 1, 0]
    assert abs(sequence_score(logits, mask, labels)-float(expected)) < 1e-7
    selected = {x['query_id']: x for x in c.read(MARGIN/'private/SELECTIONS.json') if x['arm'] == 'MARGIN_002'}
    veto = {x['query_id']: x for x in c.read(TYPE/'private/DECISIONS.json')}
    queries = p.queries(146); inputs = []
    for qid, decision in veto.items():
        if not decision['blocked']:
            continue
        row = queries[qid]
        assert decision['query_type'] == 'polar' and decision['edit_type'] == 'open'
        origin = c.read(selected[qid]['source_path'])
        assert c.digest(origin) == selected[qid]['source_output_binding']
        item = dict(row=row, original_qid=qid, origin_path=selected[qid]['source_path'],
                    off_path=decision['source_path'], expert=origin['effective_expert'], negative=False)
        inputs.append(item)
        if row['question'].startswith('Does this image show ') and row['question'].endswith('?'):
            assert row['reference'].strip().lower() == 'yes'
            question = 'Is it false that this image shows '+row['question'][21:-1]+'?'
            assert question_type(question) == 'polar'
            neg = dict(row, query_id='NEG_'+c.digest([qid, question]), question=question,
                       reference='No', role='NEGATION_DIAGNOSTIC')
            neg.pop('opaque_Base_id', None); neg.pop('base_correct', None)
            inputs.append(dict(item, row=neg, negative=True))
    assert len(inputs) == 183 and sum(x['negative'] for x in inputs) == 90
    weights = c.read(GATE/'private/REFERENCE_WEIGHTS.json')
    for task in p.tasks():
        assert p.load_state(p.initial(task))['state_hash'] == weights[task['edit_id']]['hash']
    c.write(RUN/'private/INPUTS.json', inputs)
    c.write(RUN/'private/REFERENCE_WEIGHTS.json', weights)
    c.write(RUN/'public/ADMISSION.json', dict(status='PASS', CPU_checks=3,
        original_inputs=93, negative_fixed_expert_probes=90, conditions=ARMS,
        new_output_consumers=366, original_weights=146, new_training=0, CP_enabled=False))


def evaluate():
    gpu, part = int(os.environ['GPU']), int(os.environ['PARTITION'])
    tasks = p.tasks(); indices = {t['edit_id']: i for i, t in enumerate(tasks)}
    with p.lease(gpu):
        runtime, _ = c.load(gpu)
        experts = [p.expert(p.load_state(p.initial(t))['expert'], t['seed'], runtime.device).requires_grad_(False) for t in tasks]
        mixture = p.Mixture(experts); hook = MedTraceLayerHook(runtime.get_module(c.LAYER), mixture); hook.attach()
        tokens = {a: runtime.adapter.tokenizer.encode(a, add_special_tokens=False)+[runtime.adapter.tokenizer.eos_token_id] for a in ANSWERS}
        assert len(tokens['No']) == len(tokens['Yes']) and len(tokens['No']) > 1
        assert all(runtime.adapter.tokenizer.decode(v[:-1]).strip() == a for a, v in tokens.items())
        try:
            for index, item in enumerate(c.read(RUN/'private/INPUTS.json')[part::8]):
                c.budget(); row = item['row']; origin = c.read(item['origin_path'])
                assert origin['binding']['phase']['weights'] == c.read(RUN/'private/REFERENCE_WEIGHTS.json')
                mixture.ids = [indices[item['expert']]]; mixture.weights = [1.]
                raw = runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])), row['question'], None)
                assert raw['image_sha256'] == row['image_sha256']
                old_binding = origin['binding']['judge_input']
                if not item['negative']:
                    assert raw['input_ids'][0].tolist() == old_binding['prompt_ids']
                    assert raw['attention_mask'][0].tolist() == old_binding['attention_mask']
                    assert runtime.generation_config == old_binding['generation']
                likelihoods = {arm: {} for arm in ARMS}; began = time.time()
                for answer in ANSWERS:
                    kwargs, labels, mask, binding = p.tr.teacher_batch(runtime, row, tokens[answer])
                    with torch.inference_mode():
                        hook.clear_request_routing()
                        likelihoods[ARMS[0]][answer] = sequence_score(runtime.model(**kwargs).logits, mask, labels)
                        hook.set_teacher_routing(labels)
                        assert torch.equal(hook.token_mask, mask)
                        likelihoods[ARMS[1]][answer] = sequence_score(runtime.model(**kwargs).logits, mask, labels)
                        hook.clear_request_routing()
                    del kwargs, labels, mask
                generation = dict(runtime.generation_config, decoder='canonical_binary_sequence_likelihood_v1',
                    candidates=list(ANSWERS), score='sum conditional token log probabilities including EOS', tie='No')
                judge_input = dict(question=row['question'], reference=row['reference'],
                    image_sha256=row['image_sha256'], image_path=old_binding['image_path'],
                    prompt_ids=raw['input_ids'][0].tolist(), attention_mask=raw['attention_mask'][0].tolist(),
                    runtime=old_binding['runtime'], generation=generation)
                off = c.read(item['off_path'])
                for arm in ARMS:
                    answer = select_answer(likelihoods[arm]); selected = item['expert'] if arm == ARMS[1] else None
                    phase = dict(arm=arm, node=0, prefix=146, slot=0,
                        weights=c.read(RUN/'private/REFERENCE_WEIGHTS.json'),
                        execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'),
                        binary_lock=c.digest(c.read(RUN/'private/BINARY_LOCK.json')))
                    result = dict(binding=dict(input=row, judge_input=judge_input, phase=phase, arm=arm,
                        mode='fixed_expert_polarity' if item['negative'] else 'bank_R0', prefix=146, owner_order=146),
                        R0=dict(raw_answer=answer, raw_token_ids=tokens[answer]), effective_expert=selected,
                        route=dict(activated=selected is not None, logical_edit_id=selected, original_qid=item['original_qid'],
                            expert_held_from_original_query=item['negative']),
                        likelihoods=likelihoods[arm], U_KL=None,
                        KL_status='NOT_COMPARABLE_DECODER_SUPPORT_CHANGED',
                        Base_token_consistency=tokens[answer] == off['R0']['raw_token_ids'] if not item['negative'] else None,
                        active_target=origin['active_target'] if not item['negative'] else False,
                        diagnostic_only=item['negative'], seconds=time.time()-began,
                        reused_source_binding=c.digest(origin), training=False)
                    c.write(RUN/'private/outputs'/arm/(c.digest(row['query_id'])+'.json'), result)
                if index == 0:
                    p.done('GPU_CHECK_'+str(part), dict(prompt_match=True, canonical_tokens=tokens,
                        teacher_predictor_mask_exact=True, Base_gradient=any(x.grad is not None for x in runtime.model.parameters())))
                print('QUERY', part, index, flush=True)
        finally:
            hook.detach()
    p.done('BINARY_EVAL_'+str(part))


def consumers():
    inputs = c.read(RUN/'private/INPUTS.json'); changed = {x['original_qid'] for x in inputs}
    margin = {x['query_id']: x for x in c.read(MARGIN/'private/SELECTIONS.json') if x['arm'] == 'MARGIN_002'}
    veto = {x['query_id']: x for x in c.read(TYPE/'private/DECISIONS.json')}
    rows = []
    for qid in p.queries(146):
        for label, source in [('MARGIN_002', margin[qid]), ('TYPE_GUARD_002', veto[qid])]:
            rows.append(dict(arm=label, mode='bank_R0', query_id=qid, path=source['source_path']))
        for arm in ARMS:
            path = RUN/'private/outputs'/arm/(c.digest(qid)+'.json') if qid in changed else Path(margin[qid]['source_path'])
            rows.append(dict(arm=arm, mode='bank_R0', query_id=qid, path=str(path)))
    for item in inputs:
        if item['negative']:
            for arm in ARMS:
                qid = item['row']['query_id']
                rows.append(dict(arm=arm, mode='fixed_expert_polarity', query_id=qid,
                    path=str(RUN/'private/outputs'/arm/(c.digest(qid)+'.json'))))
    assert len(rows) == 6232
    c.write(RUN/'private/CONSUMERS.json', rows)


def controller():
    import pipeline
    p.progress('BINARY_INTERFACE_EVALUATION')
    pipeline.wait([pipeline.launch('binary.py', 'binary_eval', i, i) for i in range(8)])
    consumers(); p.done('GENERATION_COMPLETE')
    pipeline.wait([pipeline.launch('binary_queue.py', 'binary_ingest')])
    p.progress('ASTRA_SCORING')
    root = RUN/'private/judge_binary_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget(); assert not list((root/'workers').glob('*/SCORER_FAILURE.json')); time.sleep(30)
    pipeline.wait([pipeline.launch('binary_report.py', 'binary_report')])


if __name__ == '__main__':
    try:
        {'binary_prepare': prepare, 'binary_eval': evaluate, 'binary_controller': controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ['ACTION']+'_'+os.environ.get('PARTITION','all')+'.json'),
            dict(error=repr(error), traceback=traceback.format_exc(), epoch=time.time(), retry=False))
        raise
