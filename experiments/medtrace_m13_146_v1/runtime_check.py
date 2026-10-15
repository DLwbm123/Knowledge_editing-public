"""CPU cold-initializer/R0/guard/config checks; never runs a GPU worker or scorer."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import torch
import module_run as r

code=Path(os.environ['M13_CPU_CODE']);sys.path.insert(0,str(code))
import contract
import runtime146 as worker
import bank146
import scorer_start
import source_workplan
import admit146
from m3bench_repro.editors.routing import MemoryRouter


def rejected(fn):
    try:fn()
    except (AssertionError,FileNotFoundError):return
    raise AssertionError('Unadmitted runtime accepted')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
    torch.set_num_threads(2);packet=json.loads(Path(os.environ['M13_CPU_PACKET']).read_text())
    for task in packet['original_tasks']:
        state=r.p.tr.TT4(task['seed'],8,8).state_dict()
        assert sum(v.numel() for v in state.values())==7168 and torch.count_nonzero(state['G1'])==0
        assert all(torch.count_nonzero(state[k])>0 for k in ('G2','G3','G4'))
    first=r.p.tr.TT4(packet['original_tasks'][0]['seed'],8,8).state_dict()
    prior=r.p.load_state(Path(os.environ['M13_CPU_REFERENCE']))['states']['BASE']
    assert all(torch.equal(first[k],prior[k]) for k in first)
    entries=[dict(logical_edit_id=f'fixture_{i}',key=[float(i),0.],radius=.1) for i in range(1,147)]
    routes=0
    for method in ('M1','M3','M13'):
        for n in contract.PREFIXES:
            row=dict(method=method,mode='bank',prefix=n,owner=0)
            selected=bank146.bank_entries(entries,row);assert len(selected)==n
            router=MemoryRouter.from_state(dict(distance='euclidean',entries=selected),device='cpu')
            assert router.route(torch.tensor([float(n),0.])).logical_edit_id==f'fixture_{n}'
            assert not router.route(torch.tensor([float(n+1),0.])).activated;routes+=1
        for n in range(1,147):
            assert len(bank146.bank_entries(entries,dict(method=method,mode='insertion',prefix=n,owner=n)))==n
    assert len(bank146.bank_entries(entries,dict(method='M13',mode='single',prefix=1,owner=146)))==1
    admit146.check_router(entries[0],entries[0])
    changed=copy.deepcopy(entries[0]);changed['radius']+=1
    rejected(lambda:admit146.check_router(changed,entries[0]))
    changed=copy.deepcopy(entries[0]);changed['key'][0]+=1
    rejected(lambda:admit146.check_router(changed,entries[0]))
    duplicate=copy.deepcopy(entries);duplicate[1]=duplicate[0]
    rejected(lambda:bank146.bank_entries(duplicate,dict(method='M13',mode='bank',prefix=146,owner=0)))
    original_run=worker.RUN
    with tempfile.TemporaryDirectory(prefix='e.',dir=os.environ['M13_CPU_TMP']) as tmp:
        worker.RUN=Path(tmp);(worker.RUN/'private').mkdir()
        (worker.RUN/'private/M13_SOURCE_PACKET.json').write_text(json.dumps(packet))
        admission=dict(status='PASS',launch_authorized=True,source_commit='fixture')
        (worker.RUN/'private/M13_146_RUNTIME_ADMISSION.json').write_text(json.dumps(admission))
        (worker.RUN/'private/GPU_SOURCE_VERSION.json').write_text(json.dumps(dict(commit='fixture')))
        rejected(worker.require_ready) # Actual139 missing, before any GPU loading.
        worker.RUN=original_run
        cfg=scorer_start.configs(Path(tmp),'fixture remote','fixture source','fixture generic scorer')
        assert {c['JUDGE_WORKER'] for c in cfg}=={'0','1','2','3'}
        assert len({c['JUDGE_RESULT_FOLDER'] for c in cfg})==4 and all(c['JUDGE_EFFORT']=='medium' for c in cfg)
    plan=source_workplan.build(packet)['public']
    assert (plan['qualified'],plan['bounded_review_required'],plan['exhausted_waiting_original_evidence'])==(7,138,1)
    result=dict(status='PASS',cold_initializers=146,original_C_BASE_exact=True,
        actual_R0_CPU_routes=routes,all_438_insertion_prefixes=True,no_future_expert_or_duplicate_bank=True,
        actual_source_gap_blocks_worker=True,four_new_isolated_scorer_configs=True,source_plan=plan,
        original_router_key_and_radius_changes_rejected=True,
        new_GPU_leases=0,new_VLM_calls=0,new_Judge=0,training_ready=False,
        live_network_and_operational_admission_pending=True)
    Path(os.environ['M13_CPU_REPORT']).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
