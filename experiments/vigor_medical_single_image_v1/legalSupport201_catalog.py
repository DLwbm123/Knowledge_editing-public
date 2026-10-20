"""Static candidate catalog and bounded abstract control-flow checks; no torch import."""
import ast
import itertools
import json
from pathlib import Path


def run():
    root = Path(__file__).resolve().parents[2]; code = Path(__file__).parent
    evidence = []
    for name in ['scalar3_ops.py', 'anchors3_ops.py']:
        tree = ast.parse((code/name).read_text()); repair = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'repair')
        loop = next(n for n in ast.walk(repair) if isinstance(n, ast.For) and ast.unparse(n.iter) == 'range(25)')
        stops = [n for n in ast.walk(loop) if isinstance(n, ast.If) and ast.unparse(n.test) == "abs(endpoint['cumulative_null_change']) <= 0.01" and any(isinstance(x, ast.Break) for x in n.body)]
        trust = [n for n in ast.walk(loop) if isinstance(n, ast.If) and ast.unparse(n.test) == 'not trust' and any(isinstance(x, ast.Break) for x in n.body)]
        assert len(stops) == 2 and len(trust) == 1 and not any(isinstance(n, ast.Continue) for n in ast.walk(loop))
        assert any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'current' for t in n.targets) for n in loop.body)
        evidence.append(dict(file='experiments/vigor_medical_single_image_v1/'+name, repair_line=repair.lineno,
                             first_null_pass_stops=True, trust_reject_rolls_back_and_stops=True,
                             full_gate_distinct_state_upper_bound_per_fixed_trajectory=1))
    count = 0; maximum = 0
    for length in range(5):
        for initial in [-.02, -.01, .01, .02]:
            for steps in itertools.product(itertools.product([-.02, -.01, .01, .02], [False, True]), repeat=length):
                count += 1; qualified = {0} if abs(initial) <= .01 else set()
                if not qualified:
                    for index, (drift, trust) in enumerate(steps, 1):
                        if not trust:
                            break
                        if abs(drift) <= .01:
                            qualified.add(index); break
                maximum = max(maximum, len(qualified)); assert len(qualified) <= 1
    assert count == 18724 and maximum == 1
    catalog = [
        dict(id='TQ_T', process='fixed teacher fit', endpoints_per_process=1,
             source='ftmq_ops.py:fit', numeric_feasibility='not guaranteed; original T null 0/3', new_generator_admitted=False),
        dict(id='I', process='one identity-metric rewrite of fixed teacher', endpoints_per_process=1,
             source='equiv3_ops.py:rewrite', numeric_feasibility='not guaranteed; old I null 0/3', new_generator_admitted=False),
        dict(id='U', process='one fixed diagonal-metric rewrite', endpoints_per_process=1,
             source='equiv3_ops.py:rewrite', numeric_feasibility='not guaranteed; old U null 0/3', new_generator_admitted=False),
        dict(id='S', process='fixed first-null-pass repair', endpoints_per_process=1,
             source='scalar3_ops.py:repair', numeric_feasibility='at most one null+trust feasible state per fixed path', new_generator_admitted=False),
        dict(id='A32', process='fixed first-null-pass response-anchor repair', endpoints_per_process=1,
             source='anchors3_ops.py:repair', numeric_feasibility='at most one null+trust feasible state per fixed path', new_generator_admitted=False),
        dict(id='OLD_N_L6_CONTROLLERS', process='historical accepted trajectories', endpoints_per_process='multiple numerical states structurally possible',
             source='nll_control.py and closed branch executors', numeric_feasibility='semantic correctness and native distinctness not certified for multiple states; closed recipes not reactivated', new_generator_admitted=False),
    ]
    value = dict(status='NO_ADMITTED_COMPARABLE_MULTI_ENDPOINT_GENERATOR', candidate_catalog=catalog,
        source_code_evidence=evidence, abstract_trajectories_checked=count, abstract_max_null_trust_states=maximum,
        proof='initial null pass ends immediately; all preceding trusted states fail null; first passing trusted state ends; trust failure rejects and rolls back',
        extra_source_visual_semantics_can_only_reduce_set=True, abstract_check_not_real_model_experiment=True,
        different_files_steps_or_seeds_not_native_distinctness_proof=True,
        no_historical_recipe_pooling=True, native_weights_not_persisted_or_compared=True,
        four_sources_with_two_correct_compliant_states='NOT_ESTABLISHED',
        impossibility_claim_scope='fixed first-null-pass paths only; not all mechanisms or Pure-W',
        model=0, GPU=0, Judge=0, old_training_extended=False)
    (root/'reports/vigor_legalSupport201_v1_20261020/CANDIDATE_CATALOG.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(dict(status=value['status'], abstract_trajectories=count, maximal_feasible_states=maximum)))


if __name__ == '__main__':
    run()
