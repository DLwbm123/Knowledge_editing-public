"""Matched normalized rank-one LoRA structural reference; independent zero start."""
import os,sys,time
from pathlib import Path
from dataclasses import replace
RUN=Path(os.environ['RUN_ROOT']);BASE=Path(os.environ['BASE_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
import core
common=core.common;worker=core.worker;train=core.train
import torch
from torch import nn
from methods.medtrace.core import MedTraceLayerHook

class LoRA(nn.Module):
    def __init__(self,seed):
        super().__init__();g=torch.Generator().manual_seed(seed);a=torch.randn(1,14336,generator=g);a=a/a.norm(dim=1,keepdim=True)
        self.A=nn.Parameter(a);self.B=nn.Parameter(torch.zeros(4096,1));self.rank=1;self.d_in=14336;self.d_out=4096;self.epsilon=1e-6
    def residual(self,x):
        x=x.float();x=x/(x.square().mean(-1,keepdim=True).sqrt()+self.epsilon);return (x@self.A.T)@self.B.T
    def normalize_factors_(self,**_):pass

def clone(state,seed,device):
    e=LoRA(seed).to(device);e.load_state_dict(state);return e

def warm(runtime,t,directory=None):
    d=Path(directory) if directory is not None else RUN/'private/lora'/t['anonymous_edit'];d.mkdir(parents=True,exist_ok=True);final=d/'W0.pt';latest=d/'latest.pt'
    if final.exists():return final
    e=LoRA(t['seed']).to(runtime.device);assert sum(p.numel() for p in e.parameters())==18432
    batches=[runtime.build_edit_batch(common.record(t))]+[runtime.build_edit_batch(replace(common.record(t),question=q)) for q in t['fit_questions']]
    saved=torch.load(latest,map_location='cpu',weights_only=True) if latest.exists() else None
    if saved:e.load_state_dict(saved['expert'])
    else:assert e.residual(torch.ones(1,14336,device=runtime.device)).abs().max()==0
    h=MedTraceLayerHook(runtime.get_module(common.LAYER),e);h.attach();began=time.time();counts=saved['counts'] if saved else {'forwards':0,'backwards':0,'tokens':0}
    try:
        for phase,steps in [('native',140),('A2',80),('W0',320)]:
            phases=['native','A2','W0']
            if saved and phases.index(phase)<phases.index(saved['phase']):continue
            if phase=='W0':opt=torch.optim.Adam([{'params':[e.A],'lr':1e-4},{'params':[e.B],'lr':1e-3}],betas=(.9,.999),eps=1e-8,weight_decay=0)
            else:opt=torch.optim.AdamW(e.parameters(),lr=1e-3,weight_decay=0)
            start=0
            if saved and phase==saved['phase']:opt.load_state_dict(saved['optimizer']);common.restore_rng(saved);start=saved['step']
            for step in range(start+1,steps+1):
                common.budget();opt.zero_grad(set_to_none=True);indices=[0] if phase=='native' else [0,1+(step-1)%4]
                for i in indices:
                    h.set_teacher_routing(batches[i].labels);ce=runtime.compute_loss(batches[i]);(ce/len(indices)).backward();counts['forwards']+=1;counts['backwards']+=1;counts['tokens']+=len(batches[i].target_token_ids)
                norm=torch.nn.utils.clip_grad_norm_(e.parameters(),1.);assert torch.isfinite(norm) and not any(p.grad is not None for p in runtime.model.parameters());opt.step();assert all(torch.isfinite(p).all() for p in e.parameters())
                if step%20==0 or step==steps:
                    common.save(latest,dict(expert=e.state_dict(),optimizer=opt.state_dict(),phase=phase,step=step,counts=counts,**common.rng()));print('STRUCTURE_UPDATE',t['order'],phase,step,flush=True)
            saved=None
        common.save(final,dict(expert=e.state_dict(),state_hash=common.state_hash(e),binding=dict(task=t,rank=1,parameters=18432,normalization='Same input RMS as TT',scaling=1.,warmup=[140,80,320],U_supervision=False,execution=common.read(RUN/'private/GPU_SOURCE_VERSION.json')),step=320))
        common.write(d/'TRAINING.json',dict(status='COMPLETE',parameters=18432,rank=1,warmup_updates=540,counts=counts,seconds=time.time()-began,seed=t['seed'],same_backbone_layer_data_R0=True));latest.unlink();return final
    finally:h.detach()

def main():
    part=int(os.environ['PARTITION']);gpu=int(os.environ['GPU']);ts=[t for t in common.read(BASE/'private/QUEUES.json')['tasks'] if t['cohort']=='P2'][:8]
    with common.lease(gpu):
        runtime,bindings=common.load(gpu)
        for t in ts[part::2]:warm(runtime,t)
    common.write(RUN/'private'/('LORA_PART_'+str(part)+'.json'),dict(status='COMPLETE',edits=4))

def evaluate():
    gpu=int(os.environ['GPU']);ts=[t for t in common.read(BASE/'private/QUEUES.json')['tasks'] if t['cohort']=='P2'][:8]
    with common.lease(gpu):
        runtime,bindings=common.load(gpu);bank=[];points={};worker.clone=clone;oldwrite=worker.write
        def write(p,d):
            if 'TT_parameters_per_expert' in d:d.pop('TT_parameters_per_expert');d.update(expert_parameters_per_expert=18432,representation='NORMALIZED_LORA_R1')
            oldwrite(p,d)
        worker.write=write
        for i,t in enumerate(ts,1):
            bank+=worker.router(runtime,t);points[t['edit_id']]=RUN/'private/lora'/t['anonymous_edit']/'W0.pt'
            worker.evaluate(runtime,bindings,t,0,'LORA_W0',0,points,bank,[dict(t,events=[e for e in t['events'] if e['task']=='T0'])],mode='insertion',prefix=i)
        worker.evaluate(runtime,bindings,ts[-1],0,'LORA_W0',0,points,bank,ts,mode='bank',prefix=8)
        for t in ts:worker.evaluate(runtime,bindings,t,0,'LORA_W0',0,{t['edit_id']:points[t['edit_id']]},worker.router(runtime,t),[t],mode='single',prefix=1)
        # Matched TT W0 singles, not historical CP or another TT rank.
        worker.clone=train.clone;worker.write=oldwrite
        for t in ts:worker.evaluate(runtime,bindings,t,0,'TT88_W0',0,{t['edit_id']:worker.w0_path(t,0)},worker.router(runtime,t),[t],mode='single',prefix=1)
    common.write(RUN/'public/FINAL_SCALE.json',dict(structural_comparison='TT88 vs normalized LoRA r1',TT_parameters=7168,LoRA_parameters=18432,LoRA_formula='1*(14336+4096)',budget_ratio=18432/7168,same_parameter_budget=False,single_edits=8,sequential_prefixes=[8],benchmark146_complete=False,remaining_TT88_matched_edits=138,independent_confirmation=False,additional_U_in_selected_version=False,no_146_claim=True))
if __name__=='__main__':
    if os.environ['ACTION']=='lora_eval':evaluate()
    else:main()
