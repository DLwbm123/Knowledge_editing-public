"""Real frozen 24-expert banks for the six new routing diagnostic combinations."""
import os,sys,time
from pathlib import Path
import torch
from dataclasses import replace
RUN=Path(os.environ['RUN_ROOT']);BASE=Path(os.environ['BASE_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
import common
sys.path.insert(0,str(RUN/'private/tools'))
import worker
from train import teachers_for
from m3bench_repro.editors.routing import MemoryRouter,RouteDecision
original=MemoryRouter.route


def route(self,query):
    config=common.read(RUN/'private/B_ACTIVE_CONFIG.json')
    if config['router']=='R_RADIUS':
        d=original(self,query)
        if d.radius is None:return d
        radius=d.radius*config['scale'];on=d.nearest_distance<=radius
        return replace(d,logical_edit_id=d.nearest_logical_edit_id if on else None,radius=radius,activated=on)
    q=query.detach().float().reshape(-1);k=self._key_matrix(q.device);w=torch.tensor(config['weights'],device=q.device);sizes=config['sizes'];expanded=torch.repeat_interleave(w,torch.tensor(sizes,device=q.device));dist=((k-q).square()*expanded).sum(-1).sqrt();i=int(dist.argmin());radius=self.radii[i]*config['scale'];on=float(dist[i])<=radius
    return RouteDecision(self.logical_ids[i] if on else None,self.logical_ids[i],float(dist[i]),radius,on,'euclidean')


def main():
    gpu=int(os.environ['GPU']);cfg=torch.load(RUN/'private/ROUTE_DIAGNOSTIC_CONFIG.pt',map_location='cpu',weights_only=True);ts=[t for t in common.read(BASE/'private/QUEUES.json')['tasks'] if t['cohort']=='P2'];bank=[]
    for t in ts:bank+=torch.load(BASE/'private/edits'/t['anonymous_edit']/'ROUTER.pt',map_location='cpu',weights_only=True)['entries']
    MemoryRouter.route=route
    with common.lease(gpu):
        runtime,bindings=common.load(gpu);check=teachers_for(runtime,common.read(BASE/'private/U_ROLES.json')['CHECK']);actual_read=worker.read
        for arm in ['FROZEN_W0','CE_U_SINGLE','CE_U_MULTI']:
            points={t['edit_id']:Path(common.read(BASE/'private/edits'/t['anonymous_edit']/'s0/COMPLETE.json')['points'][arm]) for t in ts}
            for name in ['R_RADIUS','R_SCOPE']:
                common.budget();marker=RUN/'private'/('B_DONE_'+arm+'_'+name+'.json')
                if marker.exists():continue
                config=dict(router=name,scale=cfg['R_RADIUS_scale'] if name=='R_RADIUS' else cfg['R_SCOPE_scale'],weights=cfg['R_SCOPE_weights'].tolist(),sizes=cfg['group_sizes'].tolist(),qualification='Positive-only DEV diagnostic; no verified scope-negative training',cal_binding=common.digest(common.read(RUN/'public/B_CAL_DIAGNOSTIC.json')))
                common.write(RUN/'private/B_ACTIVE_CONFIG.json',config)
                def bound_read(path):
                    d=actual_read(path)
                    if Path(path)==RUN/'private/GPU_SOURCE_VERSION.json':d=dict(d,B_execution=common.read(RUN/'private/B_SOURCE.json'),routing=config)
                    return d
                worker.read=bound_read
                label=arm+'__'+name;worker.evaluate(runtime,bindings,ts[-1],0,label,0 if arm=='FROZEN_W0' else 160,points,bank,ts,mode='Bbank',prefix=24,teacher_rows=check)
                common.write(marker,dict(status='GENERATED_NOT_SCORED',base_arm=arm,router=name,real_expert_bank=True,experts=24,TT_training=False,reference_R0='Exact frozen PR31 normal bank outputs; retained read-only'))
                print('B_COMBINATION_COMPLETE',arm,name,flush=True)
        worker.read=actual_read
    common.write(RUN/'private/B_GENERATION_COMPLETE.json',dict(status='GENERATED_SCORING_PENDING',new_combinations=6,exact_R0_reference_combinations=3,formal_candidate='NOT_CONFIRMED',epoch=time.time()))

if __name__=='__main__':main()
