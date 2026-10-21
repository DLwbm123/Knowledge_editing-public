"""Check causal token alignment, EOS exclusion, and missing-consumer rejection."""
import json
import math
import tempfile
from pathlib import Path


def main():
    import torch
    import consumers as c
    from scoring import answer_scores, first_token_difference
    logits = torch.tensor([[[0., 0., 0.], [0., 2., 0.], [0., 0., 3.], [8., 0., 0.]]])
    labels = torch.tensor([[-100, -100, 1, 2]])
    s = answer_scores(logits, labels, 2)
    expected = 2-math.log(2+math.exp(2))
    assert abs(s['mean_no_eos']-expected) < 1e-6 and s['answer_tokens'] == 1
    assert s['target_ids'] == [1, 2] and len(s['eos_logp']) == 1
    assert first_token_difference([1, 2], [1, 2], 2)['index'] is None
    assert first_token_difference([0, 2], [1, 2], 2)['index'] == 0
    try:
        answer_scores(logits, torch.tensor([[-100, -100, -100, 2]]), 2)
    except AssertionError:
        pass
    else:
        raise AssertionError('EOS-only score accepted')
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        def w(path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        tasks = []
        for i, n in enumerate((8, 8, 7, 8)):
            key = f'X{i}'
            folder = root/str(i % 2)/key
            qs = [dict(query_id=f'{key}_{j}', question=f'question{j}', reference='answer', image_path='private-image') for j in range(n)]
            candidates = [dict(qs[j//3], query_id=f'{key}_C{j}', parent_query_id=qs[j//3]['query_id'], candidate_type=('CORRECT','OTHER_RELATION','WRONG_VALUE')[j%3]) for j in range(12)]
            tasks.append(dict(key=key, eval_rows=qs, candidate_rows=candidates))
            for arm in ('BASE','JOINT'):
                for q in qs:
                    w(folder/'outputs'/arm/(q['query_id']+'.json'), dict(binding=q, raw_answer='answer', raw_token_ids=[1,2], EOS=True, at_cap=False))
                for q in candidates:
                    w(folder/'scores'/arm/(q['query_id']+'.json'), dict(binding=q, parent_query_id=q['parent_query_id'], candidate_type=q['candidate_type'], scores=dict(s, critical_span={'status':'BOUND'})))
            w(folder/'JOINT/TRAIN_COMPLETE.json', dict(steps=160,parameters=7168,save_load_exact=True))
            w(folder/'LIFECYCLE.json', dict(status='RETIRED'))
        w(root/'DATA_ADMISSION.json', dict(tasks=tasks))
        for part in ('0','1'):
            w(root/part/'EXIT.json',dict(code=0))
            w(root/part/'MODEL_COMPLETE.json',dict(status='PASS'))
        assert c.validate(root)['candidate_sequences'] == 96
        c.blind(root)
        missing = root/'0/X0/scores/BASE/X0_C0.json'
        saved = missing.read_text()
        missing.unlink()
        try:
            c.validate(root)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Missing score accepted')
        missing.write_text(saved)
        w(root/'1/EXIT.json',dict(code=1))
        try:
            c.validate(root)
        except AssertionError:
            pass
        else:
            raise AssertionError('Failed shard accepted')
    return dict(status='PASS', causal_shift=True, prompt_mask=True, eos_separate=True,
                missing_score_rejected=True, failed_shard_rejected=True,
                real_model_calls=0, synthetic_free_outputs=62, synthetic_candidate_sequences=96)


if __name__ == '__main__':
    print(json.dumps(main()))
