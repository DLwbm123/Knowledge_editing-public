"""CPU surrogate160 loop and snapshot audit; no VLM, GPU, or auto-resume."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
import module_run as r

path=Path(os.environ['M13_CPU_CODE'])
spec=importlib.util.spec_from_file_location('m13_trajectory',path/'trajectory.py')
trajectory=importlib.util.module_from_spec(spec);spec.loader.exec_module(trajectory)


def rejected(fn):
    try:fn()
    except AssertionError:return
    raise AssertionError('Invalid training binding accepted')


class FixtureRuntime:
    def __init__(self,expert):
        self.expert=expert;self.model=torch.nn.Linear(1,1);self.model.requires_grad_(False);self.calls=[]
    def compute_loss(self,batch):
        self.calls.append(batch.labels)
        # A differentiable surrogate with stochastic draws exercises all saved RNGs.
        scale=(batch.labels+1)*(.1+float(torch.rand(()))+random.random()+float(np.random.random()))
        return sum((v-(.01*(batch.labels+1))).square().mean()*scale for v in self.expert.parameters())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
    torch.set_num_threads(2)
    packet=json.loads(Path(os.environ['M13_CPU_PACKET']).read_text())
    rows=trajectory.source_rows(packet,1);assert [v['audit_role'] for v in rows]==['NATIVE']+['FIT']*4
    bad=copy.deepcopy(packet);bad['native_fit_availability'][bad['original_tasks'][7]['native']['image_path']]=False
    rejected(lambda:trajectory.source_rows(bad,8))
    initial=r.p.load_state(Path(os.environ['M13_CPU_REFERENCE']))['states']['BASE']
    cache=dict(x=torch.randn(2,14336),y=torch.randn(2,1,4096),token_counts=[1,1])
    batches=[SimpleNamespace(labels=i) for i in range(5)];hook=SimpleNamespace(set_teacher_routing=lambda _:None)
    budget_calls=[];r.c.budget=lambda:budget_calls.append(True)
    targets=None;checks=[]
    with tempfile.TemporaryDirectory(prefix='e.',dir=os.environ['M13_CPU_TMP']) as tmp:
        for arm in ('C','M1','M3','M13'):
            torch.manual_seed(20261014);random.seed(20261014);np.random.seed(20261014)
            expert=r.p.expert(initial,20261014);runtime=FixtureRuntime(expert)
            binding=dict(arm=arm,seed=20261014,steps=160,parameters=7168,source_recipe=trajectory.contract.SOURCE_RECIPE,
                source_order=[row['query_id'] for row in rows],source_binding='SYNTHETIC_FIXTURE',
                scientific_lock='SYNTHETIC_FIXTURE',source_version='PREPARATION_ONLY',budget_binding='CPU_NO_GPU_JUDGE')
            seen=[]
            def checkpoint(saved):
                dest=Path(tmp)/'snapshot.pt';torch.save(saved,dest)
                loaded=torch.load(dest,map_location='cpu',weights_only=False)
                trajectory.validate_snapshot(r,loaded,binding)
                assert all(torch.equal(saved['state'][k],loaded['state'][k]) for k in saved['state'])
                before_rng=r.c.rng();before_numpy=np.random.get_state()
                r.c.restore_rng(loaded['rng']);np.random.set_state(loaded['numpy_rng'])
                a=(float(torch.rand(())),random.random(),float(np.random.random()))
                r.c.restore_rng(saved['rng']);np.random.set_state(saved['numpy_rng'])
                b=(float(torch.rand(())),random.random(),float(np.random.random()));assert a==b
                r.c.restore_rng(before_rng);np.random.set_state(before_numpy)
                seen.append(saved['step'])
            saved=trajectory.run(r,runtime,hook,expert,batches,arm,cache,.001 if arm in ('M3','M13') else 0.,
                targets,binding,checkpoint)
            assert saved['step']==160 and seen==list(range(20,161,20)) and len(runtime.calls)==320
            assert len(budget_calls)==160*(len(checks)+1)
            assert runtime.calls==[i for step in range(1,161) for i in (0,1+(step-1)%4)]
            if arm=='C':targets=[row['operator_step_norm'] for row in saved['curve']]
            rejected(lambda:trajectory.validate_snapshot(r,saved,dict(binding,seed=binding['seed']+1)))
            rejected(lambda:trajectory.validate_snapshot(r,saved,dict(binding,source_order=list(reversed(binding['source_order'])))))
            altered=copy.deepcopy(saved)
            if arm in ('M1','M13'):altered['scalar']['step']-=1
            else:next(iter(altered['optimizer']['state'].values()))['step']-=1
            rejected(lambda:trajectory.validate_snapshot(r,altered,binding))
            altered=copy.deepcopy(saved);altered['counts']['backwards']=0
            rejected(lambda:trajectory.validate_snapshot(r,altered,binding))
            if arm in ('C','M3'):
                altered=copy.deepcopy(saved);next(iter(altered['optimizer']['state'].values()))['exp_avg_sq'].fill_(-1)
                rejected(lambda:trajectory.validate_snapshot(r,altered,binding))
            checks.append(dict(arm=arm,steps=160,source_backwards=320,operator_backwards=160 if arm in ('M3','M13') else 0,
                snapshots=8,source_cycle_exact=True,state_RNG_clock_binding_checks=True,budget_checks=160))
    result=dict(status='PASS',synthetic_surrogate_only=True,cases=checks,unqualified_source_rejected=True,
        seed_source_order_optimizer_clock_changes_rejected=True,automatic_resume=False,
        new_GPU_leases=0,new_VLM_calls=0,new_Judge=0,training_ready=False,
        GPU_dispatch_and_true_bank_implemented=False)
    Path(os.environ['M13_CPU_REPORT']).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
