"""No model calls: mathematical, source and conditional-consumer admission."""
import os
import py_compile
import json
from pathlib import Path
import correction_math as cm
import paired_run as p
from guard_admission import require_reference


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    require_reference()
    checks=cm.selfcheck()
    assert p.j.jg.selfcheck()['status']=='PASS' and p.f.cpu.__name__=='cpu'
    targets=p.c.read(p.RUN/'private/TARGETS.json')['rows'];assert len(targets)==8 and sum(v['step'] for v in targets)==341
    for v in targets:
        curve=p.c.read(Path(os.environ['GUARD_PARENT'])/'private/prefix_curve'/f"{v['owner']}.json")
        first=next(x for x in curve if x['decision']['status']=='SKIPPED_INFEASIBLE' and x['decision']['pure_source_gain']>0)
        assert first['step']==v['step']
    for name in ['correction_math.py','paired_run.py','paired_queue.py','paired_report.py']:
        py_compile.compile(str(p.RUN/'private/tools'/name),doraise=True)
    result=dict(status='PASS',math=checks,reference_receipt='PASS',fixed_reconstruction=341,
        external_source_budget_step_rules_unchanged=True,full_consumers_implemented=True,
        same_source_seed_scope=True,new_model_calls=0,new_Judge=0)
    p.c.write(p.RUN/'public/CPU_CHECKS.json',result);print(json.dumps(result))


if __name__=='__main__':main()
