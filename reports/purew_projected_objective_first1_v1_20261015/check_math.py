"""Synthetic math/protocol checks only: no model, data, network or GPU."""
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "prior_algebra", HERE.parent / "purew_objective_scope_v1_20261015/check_algebra.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
kl, softmax, split, distribution = prior.kl, prior.softmax, prior.split, prior.distribution


def run():
    p = json.loads((HERE / "PROTOCOL.json").read_text())
    alpha = p["objective"]["alpha"]
    p0 = [.2, .3, .5]
    a0, q0 = split(p0)
    astar = max(a0, alpha)
    star = distribution(astar, q0)
    z = [-.2, .3, -.4]

    def marginal(z):
        a, _ = split(softmax(z))
        return kl([astar, 1-astar], [a, 1-a])

    def unified(z):
        return kl(star, softmax(z))

    def components(z):
        a, q = split(softmax(z))
        return marginal(z) + (1-astar)*kl(q0, q)

    assert abs(unified(z)-components(z)) < 1e-12
    current = softmax(z)
    a, q = split(current)
    full_grad = [x-y for x, y in zip(current, star)]
    marginal_grad = [a-astar] + [(astar-a)*v for v in q]
    h, max_error = 1e-5, 0.
    for f, expected in [(unified, full_grad), (marginal, marginal_grad)]:
        for i in range(3):
            plus, minus = list(z), list(z)
            plus[i] += h
            minus[i] -= h
            error = abs((f(plus)-f(minus))/(2*h)-expected[i])
            max_error = max(max_error, error)
    assert max_error < 1e-8

    candidates = 0
    for teacher in [p0, [.98, .01, .01]]:
        birth_a, birth_q = split(teacher)
        target = distribution(max(birth_a, alpha), birth_q)
        for mass in [alpha, .96, .98, .995]:
            for conditional in [birth_q, [.2, .8], [.5, .5], [.9, .1]]:
                candidate = distribution(mass, conditional)
                assert kl(target, teacher) <= kl(candidate, teacher) + 1e-12
                candidates += 1
        if birth_a >= alpha:
            assert max(abs(x-y) for x,y in zip(target, teacher)) < 1e-12
    unavoidable = kl([a0,1-a0], [alpha,1-alpha])
    assert abs(kl(p0,star)-unavoidable) < 1e-12
    assert unavoidable > .01
    assert -math.log(alpha) < p["gates"]["complete_CE_strict_below"]

    # Sensitivity can be positive with an incorrect decision or worsened full answer.
    assert math.log(.4/.2) > 0 and .4 < .6
    assert math.log(.6/.3) > math.log(.9/.8) and .6 < .9
    tiny_flip = kl([.501,.499],[.499,.501])
    assert tiny_flip < .01
    # Verify the sufficient same-context Pinsker criterion on a finite grid.
    teacher, checks = [.7,.3], 0
    gamma = teacher[0]-teacher[1]
    for n in range(1,1000):
        candidate = [n/1000,1-n/1000]
        if kl(teacher,candidate) < gamma**2/2:
            assert candidate[0] > candidate[1]
            checks += 1

    # Birth-point quadratic recurrence includes cross terms, including cancellation.
    row, delta, step = [1.,2.], [.3,-.1], [-.2,.05]
    dot = lambda x,y: sum(a*b for a,b in zip(x,y))
    old = .5*dot(row,delta)**2
    linear = dot(row,delta)*dot(row,step)
    updated = old+linear+.5*dot(row,step)**2
    direct = .5*dot(row,[x+y for x,y in zip(delta,step)])**2
    assert abs(updated-direct) < 1e-12
    assert updated < old  # cumulative proxy is not an irreversible per-step sum.

    assert not any(p["authorization"][k] for k in
                   ["execution","install","waiter","monitoring","model_calls","Judge"])
    assert p["design"]["new_arms"] == ["PM","PU"]
    assert p["reference"]["consumers"] == 87
    assert p["evaluation"]["new_total"] == 2*87
    assert p["evaluation"]["total_with_historical_SH00"] == 3*87
    for key in ["GPU_process_hours","formal_slots","native_writes_including_admission",
                "backwards","forwards_including_vision","generations","new_Judge",
                "candidate_checks","active_set_checks"]:
        assert p["budgets"][key] == 2*p["budgets"]["per_new_arm"][key]
    assert p["budgets"]["activation_epoch"] is None
    assert p["gates"]["all_ten_evidence_gaps_strict_positive_finite"]
    assert p["gates"]["allowed_individual_damage"] == 0
    assert p["gates"]["unknown_fails"]
    assert p["hardware"]["GPU"] is None
    return dict(status="PASS_SYNTHETIC_MATH_AND_PROTOCOL_ONLY",
                projection_grid_checks=candidates, gradient_max_error=max_error,
                unavoidable_same_context_KL_example=unavoidable,
                positive_evidence_wrong_answer_counterexample=True,
                tiny_KL_top1_flip=tiny_flip, Pinsker_grid_checks=checks,
                cumulative_cross_term_check=True,
                VLM_calls=0, GPU_calls=0, Judge_calls=0,
                production_masks_tested=False, production_CPU=False,
                note="Numerical sanity checks supplement conditional proofs; no external coverage or clinical certificate.")


if __name__ == "__main__":
    print(json.dumps(run(),indent=2))
