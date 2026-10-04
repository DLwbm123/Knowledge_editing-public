"""CPU qualification for independent initialization/P1 only; later consumers stay gated."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
from audit import read,write,digest


def main():
    began=time.time();plan=read(RUN/'PLAN_CONFIG.json');parent=Path(plan['parent_run'])
    assert read(RUN/'EXECUTION_ORDER_AMENDMENT.json')['early_GPU6']
    assert read(parent/'private/CPU_ADMISSION.json')['status']=='PASS'
    assert not read(parent/'private/H_SEMANTIC_BINDING_AUDIT.json')['blocked']
    ledger=read(RUN/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json')
    assert digest({k:v for k,v in ledger.items() if k!='freeze_id'})==ledger['freeze_id']
    by={t['edit_id']:t for t in ledger['tasks']};tasks=[by[e] for e in ledger['main_T0']]
    assert len(tasks)==len(set(ledger['main_T0']))==146
    H=read(RUN/'private/H_AVAILABLE.json')
    assert H==read(parent/'private/H_AVAILABLE.json')
    assert sorted(by[e]['order'] for e in H)==plan['H8_orders'] and sum(map(len,H.values()))==12
    from admit import relocate
    paths=set()
    for t in tasks:
        assert len(t['fit_questions'])==4 and t['fit_status']=='SUPPORTED' and t['U_fit']
        paths.update(relocate(u['image_path']) for u in t['U_fit'])
        paths.add(relocate(t['native']['image_path']))
        for e in t['events']:
            paths.update(relocate(ledger['queries'][q]['image_path']) for q in e['all_probe_query_ids'])
    for rs in H.values():paths.update(r['image_path'] for r in rs)
    assert all(Path(x).is_file() for x in paths)
    source=RUN/'private/source';lock=read(RUN/'private/SOURCE_COMMIT.json')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==lock['commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
    official=RUN/'private/official_llava'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=official,text=True).strip()==lock['official_source_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=official,text=True).strip()
    import selfcheck
    selfcheck.main()
    import torch,worker
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    from structures import TuckerC4,optimizer_for
    torch.manual_seed(20260912);cp=AsymmetricCPExpert(14336,4096,4)
    with torch.no_grad():cp.rho.copy_(torch.tensor([.1,-.2,.3,-.4]))
    for e in (cp,TuckerC4(cp),LowRankExpert(cp,20260912,rank=4)):
        clone=worker.clone(e.state_dict(),20260912,'cpu');free=worker.clone(worker.inference_state(e.state_dict()),20260912,'cpu')
        x=torch.randn(2,14336);torch.testing.assert_close(clone.residual(x),free.residual(x),rtol=2e-5,atol=2e-6)
        opt=optimizer_for(clone,torch.nn.Identity());clone.residual(x).square().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in clone.parameters());opt.step()
        assert set(clone.state_dict())==set(e.state_dict())
    assert set(worker.ARMS)==set(plan['arms']) and worker.legacy.GPUS=={int(k):v for k,v in plan['hardware']['UUIDs'].items()}
    import controller
    controller.guard()
    write(RUN/'private/CPU_ADMISSION.json',dict(status='PASS',phase='INDEPENDENT_MECHANICAL_AND_P1_ONLY',N=146,freeze_id=ledger['freeze_id'],H8=8,H_relations=12,paths_checked=len(paths),source_read_only=True,compact_clone_optimizer_inference_three_structures='PASS',full_model_GPU_mechanical='PENDING',parent_bank_admission='PENDING',Judge_and_report_implementation='PENDING',seconds=time.time()-began,execution_source=read(RUN/'private/GPU_SOURCE_VERSION.json')))
    with worker.legacy.locked_ledger() as costs:costs['CPU_admission_seconds']=time.time()-began
    print('CPU_INDEPENDENT_PHASE_ADMISSION_PASS',flush=True)


if __name__=='__main__':main()
