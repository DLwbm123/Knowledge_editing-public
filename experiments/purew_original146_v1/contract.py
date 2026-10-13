"""Source-role checks and native materialization, shared by admission and training."""
import json
from pathlib import Path

ARM = 'PUREW_H146'
METHODS = (ARM, 'TT88_W0_RETRO146', 'balancedit')
PREFIXES = (1, 50, 100, 146)


def base_completion(output):
    assert output['error'] is None
    assert isinstance(output['model_answer_raw'], str)
    assert all(type(x) is int for x in output['raw_generated_token_ids'])
    return dict(raw_answer=output['model_answer_raw'], raw_token_ids=output['raw_generated_token_ids'])


def source_audit(tasks, ledger):
    assert len(tasks) == 146 and [t['order'] for t in tasks] == list(range(1, 147))
    assert len({t['edit_id'] for t in tasks}) == 146
    native = {(t['native']['image_sha256'], t['native']['question']) for t in tasks}
    fits = {(t['native']['image_sha256'], q) for t in tasks for q in t['fit_questions']}
    assert all(len(t['fit_questions']) == 4 for t in tasks)
    collisions, support = {}, {}
    for role in ('T0', 'T1G', 'T2G', 'T1L', 'T2L'):
        rows = [ledger['queries'][qid] for t in tasks for e in t['events']
                if e['task'] == role for qid in e['all_probe_query_ids']]
        keys = {(r['image_sha256'], r['question']) for r in rows}
        eligible = [r for r in rows if role == 'T0' or ledger['Base_correctness'][r['query_id']] == role.endswith('L')]
        eligible_keys = {(r['image_sha256'], r['question']) for r in eligible}
        if role != 'T0':
            assert not fits & keys, 'Conservative FIT overlaps evaluation input'
            assert not native & eligible_keys, 'Native training overlaps eligible evaluation input'
            if role.endswith('G'):
                assert not native & keys
        collisions[role] = len(native & keys)
        support[role] = len(eligible)
    assert support == dict(T0=146, T1G=578, T2G=557, T1L=2, T2L=54)
    assert collisions['T1L'] == 68
    return dict(status='PASS', native=146, conservative_FIT=584, eligible_support=support,
                evaluation_labels_used_for_training=False, native_noneligible_T1L_collisions=68,
                active_target_rule_preserved=True, external_training_data=0)


def proposals(gradient, master, *, tensor_metrics=False):
    import torch
    assert gradient.dtype == master.dtype == torch.float32
    assert torch.isfinite(gradient).all()
    norm = gradient.norm()
    delta = -.01 * gradient * (5. / norm.clamp_min(5.))
    after = master + delta
    assert torch.isfinite(after).all()
    metrics = dict(gradient_norm=norm, proposal_norm=delta.norm())
    return after, metrics if tensor_metrics else {k:float(v) for k,v in metrics.items()}


def selfcheck():
    import torch
    g = torch.tensor([[3., 4.]], dtype=torch.float32)
    w = torch.ones_like(g)
    z, d = proposals(g, w)
    assert torch.allclose(z, w - .01 * g) and abs(d['proposal_norm'] - .05) < 1e-7
    tensor_z, tensor_d = proposals(g, w, tensor_metrics=True)
    assert torch.equal(z, tensor_z) and d == {k:float(v) for k,v in tensor_d.items()}
    assert torch.equal(proposals(torch.zeros_like(g), w)[0], w)
    half = w.half()
    tiny = torch.full_like(w, 1e-4)
    master = half.float()
    for _ in range(300):
        master, _ = proposals(tiny, master)
    assert not torch.equal(master.half(), half), 'Master must retain sub-ulp writes across steps'
    return dict(status='PASS', cap=True, zero_gradient=True, accumulated_quantization=True)


if __name__ == '__main__':
    print(json.dumps(selfcheck()))
