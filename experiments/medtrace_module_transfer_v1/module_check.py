"""CPU admission against retained native TT states; no VLM or Judge calls."""
import json
from types import SimpleNamespace
import torch
import module_run as r
import module_report as report


def main():
    torch.set_num_threads(2)
    checks = dict(math=r.mm.selfcheck(), reporting=report.selfcheck())
    tasks = r.s.tasks()
    assert [t['order'] for t in tasks] == list(range(1, 9))
    assert sorted(t['order'] for part in range(len(r.GPUS)) for t in tasks[part::len(r.GPUS)]) == list(range(1, 9))
    assert sum(len(r.s.source(t)) for t in tasks) == 56
    assert sum(len(r.s.events(t)) for t in tasks) == 115
    assert len(r.s.previous.data.held()) == 96
    basis, held = r.q.split()
    assert len(basis) == 61 and len(held) == 96
    assert not {x['source_group'] for x in basis} & {x['source_group'] for x in held}
    saved = r.p.load_state(r.REFERENCE / 'private/candidates/C/1.pt')['states']
    generator = torch.Generator().manual_seed(20261011)
    residuals = []
    for name in ('BASE', 'FINAL'):
        original = saved[name]
        assert sum(v.numel() for v in original.values()) == 7168
        canonical, _ = r.mm.canonicalize(original)
        other = {k: v.double() for k, v in original.items()}
        for i in range(3):
            key, next_key = r.mm.KEYS[i:i + 2]
            rank = other[key].shape[-1]
            u, _, vh = torch.linalg.svd(torch.randn(rank, rank, generator=generator, dtype=torch.float64))
            matrix = u @ torch.diag(torch.linspace(.5, 2., rank, dtype=torch.float64)) @ vh
            other[key] = torch.einsum('aib,bc->aic', other[key], matrix)
            other[next_key] = torch.einsum('ab,bic->aic', torch.linalg.inv(matrix), other[next_key])
        gauged, _ = r.mm.canonicalize({k: v.float() for k, v in other.items()})
        for k in r.mm.KEYS:
            torch.testing.assert_close(canonical[k], gauged[k], rtol=2e-4, atol=2e-5)
        x = torch.randn(3, 14336, generator=generator)
        b, a = r.mm.factors(original); bc, ac = r.mm.factors(canonical)
        output, transformed = (x @ a.T) @ b.T, (x @ ac.T) @ bc.T
        error = float((output - transformed).norm() / output.norm().clamp_min(1e-12))
        assert error <= 2e-5
        residuals.append(dict(state=name, relative_output_error=error))
        proposed = {k: v + torch.randn(v.shape, generator=generator) * 1e-5 for k, v in canonical.items()}
        target = r.fm.map_norm(canonical, proposed) * .8
        matched, _ = r.match_proposal(canonical, proposed, target)
        recanonical, _ = r.mm.canonicalize(matched)
        assert abs(r.fm.map_norm(canonical, recanonical) - target) <= 1e-10 + .002 * target

    expert = r.p.expert(saved['BASE'], tasks[0]['seed'])
    coefficients = [torch.randn(7168, generator=generator) for _ in range(7)]
    batches = [SimpleNamespace(labels=i) for i in range(7)]
    hook = SimpleNamespace(set_teacher_routing=lambda _: None)
    def loss(batch):
        vector = torch.cat([v.flatten() for v in expert.parameters()])
        return (vector * coefficients[batch.labels]).sum() + .01 * vector.square().sum()
    runtime = SimpleNamespace(compute_loss=loss)
    matrix, _ = r.source_gradients(runtime, hook, expert, batches, 2, True)
    gradient = torch.cat([v.grad.flatten() for v in expert.parameters()]).clone()
    r.source_gradients(runtime, hook, expert, batches, 2, False)
    torch.testing.assert_close(gradient, torch.cat([v.grad.flatten() for v in expert.parameters()]), rtol=1e-6, atol=1e-6)
    before = r.old.snapshot(expert)
    delta, _ = r.mm.project_seven(-matrix, -matrix.sum(0))
    assert (-matrix @ delta).min() >= -1e-7
    proposed = r.q.add(before, torch.ones(7168, dtype=torch.float64) * .01)
    chosen, receipt = r.finite_candidate(runtime, hook, expert, before, proposed, batches, [-1e10] * 7, False)
    assert receipt['status'] == 'SKIPPED_FINITE_CONFLICT' and all(torch.equal(chosen[k], before[k]) for k in before)
    chosen, receipt = r.finite_candidate(runtime, hook, expert, before, proposed, batches, [1e10] * 7, False)
    assert receipt['status'] == 'ACCEPTED' and receipt['fraction'] == 1.
    for t in tasks:
        assert (r.REFERENCE / 'private/candidates/C' / f"{t['order']}.pt").is_file()
        for row in r.s.source(t) + r.s.events(t) + held:
            assert r.s.baseline(t, row).is_file()
    for k in r.COUNTS:
        r.COUNTS[k] = 0
    checks.update(status='PASS', native_TT_parameters=7168, native_FP32_gauge_checks=residuals,
        seven_gradient_accumulation=True, finite_accept_and_reject=True, three_GPU_partition=True,
        original_baselines_readable=True, VLM_forwards=0, Judge=0)
    r.c.write(r.RUN / 'public/CPU_ADMISSION.json', checks)
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
