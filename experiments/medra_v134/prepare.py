"""CPU admission into a new run root; V133 remains read only."""
import collections
import json
import os
from pathlib import Path
import sys
import time


def main():
    root,raw,parent=map(lambda k:Path(os.environ[k]),('EXPERIMENT_ROOT','RAW_MED_ROOT','PARENT_RUN'))
    assert not root.exists()
    sys.path[:0]=[str(Path(__file__).parent),str(parent/'private/tools'),str(parent/'private/source'),str(parent/'private/official_llava')]
    import torch
    import module_run as r
    import ra_risk_math as risk_math
    import ra_evaluate as evaluate
    assert not torch.cuda.is_available()
    old=r.c.read(raw/'INPUTS_FROZEN.json');oldlock=r.c.read(raw/'PROTOCOL_LOCK.json')
    assert r.c.digest(old)==oldlock['input_binding']
    inputs=dict(old,plan=[dict(v,method='MED_RA') for v in old['plan']])
    assert collections.Counter(v['mode'] for v in inputs['plan'])==dict(single=2956,insertion=146,bank=4392)
    canonical=r.p.load_state(raw/'retained/MED146.pt')
    assert len(canonical['states'])==len(canonical['R0_entries'])==146
    assert canonical['tasks']==inputs['tasks'] and canonical['source_rows']==inputs['source_rows']
    assert canonical['protocol']['input_binding']==oldlock['input_binding']
    for t in inputs['tasks']:
        key=t['edit_id'];state=canonical['states'][key]
        assert evaluate.state_id(state)==canonical['state_hashes'][key]
        receipt=r.c.read(raw/'training'/f"MED_{t['order']}.json")
        assert receipt['steps']==160 and receipt['save_load_exact']
        assert receipt['input_binding']==oldlock['input_binding']
        assert canonical['R0_entries'][t['order']-1]['logical_edit_id']==key
    cpu=risk_math.selfcheck(r.mm.factors)
    assert (parent/'private/OPERATOR_FISHER.pt').exists()
    assert all((raw/name).exists() for name in ('CURVATURE_ADMISSION.json','CURVATURE_BINDING.json','ACTUAL_MASK_DERIVATIVE_CHECK.json'))
    root.mkdir(parents=True)
    assert root.stat().st_dev==parent.stat().st_dev
    probe=root/'probe';probe.write_text('134');assert probe.read_text()=='134';probe.unlink()
    assert os.statvfs(root).f_bavail*os.statvfs(root).f_frsize>8*1024**3
    clock=json.loads(os.environ['EXPERIMENT_CLOCK_JSON'])
    assert clock['hard_end']-clock['original_start']==86400 and time.time()<clock['hard_end']-3600
    lock=dict(experiment='MED_RA_V134',seed=oldlock['seed'],GEN_seed=oldlock['GEN_seed'],
        input_binding=r.c.digest(inputs),raw_input_binding=oldlock['input_binding'],mapping_binding=oldlock['mapping_binding'],
        numeric=oldlock['numeric'],steps=160,inherited_raw_training_updates=23360,new_gradient_updates=0,
        source_reference=oldlock['source_reference'],selected_physical_GPU=int(os.environ['GPU']),
        expected_GPU_UUID=os.environ['M3BENCH_FORMAL_EXPECTED_GPU_UUID'],minimum_free_MiB=45*1024,
        alpha_grid=list(risk_math.GRID),source_gain_fraction=.90,source_atol=risk_math.SOURCE_ATOL,
        risk_epsilon=risk_math.EPSILON,risk_rtol=risk_math.RISK_RTOL,q_quantile=.75,q_interpolation='linear',
        risk_rule='R(alpha*w)/(trace(F)*norm(w_raw)^2+epsilon)',
        budgets=dict(forwards=9000000,backwards=1500,operator_backwards=300,source_forwards=6000,
            updates=0,generations=7600,vision_forwards=40000,new_bytes_limit=47_000_000_000,minimum_free_bytes=8*1024**3),
        no_monitor=True,source_only_coefficient_selection=True,independent_confirmation=False,
        allow_sequential_only_after_preregistered_single_gate=True)
    for name,value in [('INPUTS_FROZEN.json',inputs),('PROTOCOL_LOCK.json',lock),('CLOCK.json',clock),
        ('CPU_ADMISSION.json',dict(cpu,raw146_state_hashes_exact=True,raw146_training_receipts_complete=True,
            original_source_and_router_unchanged=True,no_GPU_calls=True,planned_consumers=7494)),
        ('USER_APPROVAL.json',dict(answer='采用修正方案，重建后继续',risk_rule=lock['risk_rule'],source_gain_fraction=.90))]:
        r.c.write(root/name,value)
    print(json.dumps(dict(status='PREPARED',GPU=lock['selected_physical_GPU'],clock=clock)))


if __name__=='__main__':main()
