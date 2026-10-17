"""Small synthetic algebra checks; no model/data/network/GPU imports or execution."""
import json
import math
from pathlib import Path


def kl(p, q):
    return sum(x * math.log(x / y) for x, y in zip(p, q))


def softmax(z):
    values = [math.exp(v - max(z)) for v in z]
    return [v / sum(values) for v in values]


def split(p):
    return p[0], [v / (1 - p[0]) for v in p[1:]]


def distribution(a, q):
    return [a] + [(1 - a) * v for v in q]


def run():
    chain_error = 0.0
    projection_checks = 0
    for p0, p in [([.2, .3, .5], [.7, .1, .2]),
                  ([.98, .01, .01], [.95, .02, .03]),
                  ([.1, .8, .1], [.11, .79, .1])]:
        a0, q0 = split(p0)
        a, q = split(p)
        rhs = kl([a0, 1-a0], [a, 1-a]) + (1-a0)*kl(q0, q)
        chain_error = max(chain_error, abs(kl(p0, p)-rhs))
        assert abs(kl(p0, p)-rhs) < 1e-12
        alpha = .95
        star = distribution(max(a0, alpha), q0)
        for candidate_a in [alpha, .96, .98, .995]:
            for candidate_q in [q0, [.2, .8], [.5, .5], [.9, .1]]:
                candidate = distribution(candidate_a, candidate_q)
                assert kl(star, p0) <= kl(candidate, p0) + 1e-12
                projection_checks += 1
        if a0 >= alpha:
            assert max(abs(x-y) for x, y in zip(star, p0)) < 1e-12

    # Independent finite differences: CE gradient is (1-a) times odds gradient.
    z = [-.2, .3, -.4]
    p = softmax(z)
    h = 1e-5
    def ce(v):
        return -math.log(softmax(v)[0])
    def odds_loss(v):
        return math.log(sum(math.exp(x) for x in v[1:])) - v[0]
    gradient_error = 0.0
    for i in range(len(z)):
        plus, minus = list(z), list(z)
        plus[i] += h
        minus[i] -= h
        gce = (ce(plus)-ce(minus))/(2*h)
        godds = (odds_loss(plus)-odds_loss(minus))/(2*h)
        gradient_error = max(gradient_error, abs(gce-(1-p[0])*godds))
    assert gradient_error < 1e-8

    # Output Fisher decomposition along a generic logit direction.
    dz = [.7, -.3, .2]
    a, q = split(p)
    mean = sum(x*y for x, y in zip(p, dz))
    full = sum(x*(y-mean)**2 for x, y in zip(p, dz))
    qmean = sum(x*y for x, y in zip(q, dz[1:]))
    qvar = sum(x*(y-qmean)**2 for x, y in zip(q, dz[1:]))
    fisher_error = abs(full-(a*(1-a)*(dz[0]-qmean)**2+(1-a)*qvar))
    assert fisher_error < 1e-12

    # Small KL is not a top-1 preservation guarantee at a small decision gap.
    flip_kl = kl([.501, .499], [.499, .501])
    assert flip_kl < .01
    assert -math.log(.85) > .1 and -math.log(.95) < .1

    protocol = json.loads((Path(__file__).parent/'PROTOCOL.json').read_text())
    assert not protocol['authorization']['execution']
    assert not protocol['authorization']['monitoring']
    assert protocol['design']['no_new_objective_arm']
    assert {(a['soft'], a['hard']) for a in protocol['design']['new_arms']} == {(0,0),(1,0),(0,1)}
    b = protocol['budgets']
    for key in ['formal_slots','native_writes_including_admission','backwards',
                'forwards_including_vision','generations','new_Judge','candidate_checks','active_set_checks']:
        assert b[key] == 3*b['per_new_arm'][key]
    assert protocol['evaluation']['new_total'] == 3*87
    assert protocol['evaluation']['total_including_historical_R'] == 4*87
    return dict(status='PASS_SYNTHETIC_ALGEBRA_AND_PROTOCOL_ONLY',
                chain_rule_max_error=chain_error,
                grid_projection_checks=projection_checks,
                gradient_relation_max_error=gradient_error,
                fisher_split_error=fisher_error,
                tiny_KL_argmax_flip=flip_kl,
                official_code_executed=False, production_CPU=False,
                VLM_calls=0, GPU_calls=0, Judge_calls=0,
                note='Numerical sanity checks supplement derivations, not global proofs or runtime admission.')


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
