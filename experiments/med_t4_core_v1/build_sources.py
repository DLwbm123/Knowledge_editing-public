"""Derive count-generalized T4 runners from the selected, fixed training recipes.

Only admission/count/path plumbing changes. Optimizers, losses, steps, operator
calibration and update rules are inherited verbatim from the declared sources.
"""
from pathlib import Path


def replace_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def build(worktree):
    w = Path(worktree)
    med = (w/'experiments/med150_v1/train.py').read_text()
    med = replace_once(med, 'sys.path[:0]=[str(ROOT/\'tools\'),',
        'sys.path[:0]=[str(ROOT/\'tools\'),os.environ[\'MED150_TOOLS\'],')
    med = replace_once(med, "assert len(data['tasks'])==len(expected)==4 and data['candidate_binding']==LOCK['candidate_binding']",
        "assert 0 < len(data['tasks']) == len(expected) <= 137 and data['candidate_binding']==LOCK['candidate_binding']")
    med = replace_once(med, "assert data['original146_input_binding']==oldlock['input_binding']",
        "assert data['status']=='T4_SOURCE_ADMITTED' and data['candidate_binding']==LOCK['candidate_binding']")
    med = replace_once(med, 'assert owner in (147,148,149,150)',
        "assert owner in LOCK['owners'] and 1 <= owner <= 594")
    marker = "        finally:hook.detach()\n    assert COUNTS['updates']==1280 and len(list((ROOT/'candidates').glob('MED_*.pt')))==4"
    addition = '''        finally:hook.detach()
        import matched_train as mt
        token_rows=json.loads((ROOT/'TOKEN_LOCK.private.json').read_text())['records']
        token=next(x for x in token_rows if x['order']==owner)
        initial_med=loaded
        admission={'status':'T4_SOURCE_ADMITTED','owners':LOCK['owners']}
        mt.train(runtime,r,task,entry['source_rows'],bindings,admission,token,'MDU',ROOT/'weights',budget,
                 mt.state_digest(initial_med),initial_med)
        write('MDU_PROGRESS.json',dict(owner=owner,counts=COUNTS,time=time.time()))
    assert COUNTS['updates']==480*len(data['tasks']) and len(list((ROOT/'candidates').glob('MED_*.pt')))==len(data['tasks'])'''
    med = replace_once(med, marker, addition)
    med = replace_once(med, "status='FOUR_MED_AND_FOUR_C_COMPLETE'", "status='T4_C_MED_MDU_SHARD_COMPLETE'")
    med = replace_once(med, 'new_MED_experts=4,new_C_calibration_trajectories=4,actual_new_scientific_updates=1280,',
        "new_MED_experts=len(data['tasks']),new_C_calibration_trajectories=len(data['tasks']),actual_new_scientific_updates=480*len(data['tasks']),")
    med = replace_once(med, 'inherited_original_MED_updates=23360,', 'inherited_original_MED_updates=0,')
    matched = (w/'experiments/med_mdu150_v1/matched_train.py').read_text()
    matched = replace_once(matched, "admission.get('status') != 'SOURCE_49_INPUTS_CHECKED' or len(admission.get('owners', [])) != 49",
        "admission.get('status') != 'T4_SOURCE_ADMITTED' or not 1 <= len(admission.get('owners', [])) <= 137")
    matched = replace_once(matched, "len(set(admission['owners'])) != 49 or any(not 1 <= x <= 150 for x in admission['owners'])",
        "len(set(admission['owners'])) != len(admission['owners']) or any(not 1 <= x <= 594 for x in admission['owners'])")
    coverage = (w/'experiments/med_d_universal_v1/coverage.py').read_text()
    coverage = replace_once(coverage, 'if len(prior) != 25:', 'if len(prior) != 0:')
    coverage = replace_once(coverage, "len(entries) != 101", "not 1 <= len(entries) <= 137")
    coverage = coverage.replace("'unresolved_not_read': 49", "'unresolved_not_read': 0")
    sources = {'initialize.py':med, 'matched_train.py':matched, 'coverage.py':coverage,
        'train_pair.py':(w/'experiments/med_fixed_depth_v1/train_pair.py').read_text(),
        'span_train.py':(w/'experiments/med_proposition_generalization_v1/span_train.py').read_text(),
        'robust_objective.py':(w/'experiments/med_proposition_generalization_v1/robust_objective.py').read_text()}
    for name, source in sources.items():
        compile(source, name, 'exec')
    return sources
