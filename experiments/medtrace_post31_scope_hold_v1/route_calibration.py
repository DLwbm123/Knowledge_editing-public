"""Pre-registered positive-only routing diagnostics; no fabricated negative labels."""
import os,sys
from pathlib import Path
import torch
RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
from common import read,write,save,budget


def group_square(query,keys):
    sq=(query[:,None,:]-keys[None,:,:]).square();return torch.stack([v.mean(-1) for v in torch.tensor_split(sq,32,dim=-1)],-1)


def main():
    budget();d=torch.load(RUN/'private/FEATURES.pt',weights_only=True,map_location='cpu');entries=d['bank'];ids=[e['logical_edit_id'] for e in entries];keys=torch.stack([e['key'].reshape(-1) for e in entries])
    train=[v for v in d['features'].values() if v['row'].get('role')=='ROUTE_TRAIN_POSITIVE'];cal=[v for v in d['features'].values() if v['row'].get('role')=='ROUTE_CAL'];assert len(train)==96 and len(cal)==24
    q=torch.stack([v['key'].reshape(-1) for v in train]);c=torch.stack([v['key'].reshape(-1) for v in cal]);owners=torch.tensor([ids.index(v['row']['owner']) for v in train]);co=torch.tensor([ids.index(v['row']['owner']) for v in cal]);radii=torch.tensor([e['radius'] for e in entries]);sizes=torch.tensor([len(x) for x in torch.tensor_split(torch.arange(keys.shape[1]),32)])
    tr=group_square(q,keys)[torch.arange(len(q)),owners];ca=group_square(c,keys);tr=tr/tr.mean().clamp_min(1e-12)
    results=[];weights=[]
    for reg in [.1,1.,10.]:
        z=torch.zeros(32,requires_grad=True);opt=torch.optim.Adam([z],lr=.05)
        for _ in range(100):
            opt.zero_grad();w=z.softmax(0)*32;loss=(tr*w).sum(-1).mean()/32+reg*(w-1).square().mean();loss.backward();opt.step()
        w=(z.detach().softmax(0)*32);dist=(ca*w*sizes).sum(-1).sqrt();nearest=dist.argmin(-1);on=dist[torch.arange(len(c)),nearest]<=radii[nearest]
        results.append(dict(router='R_SCOPE',config_regularizer=reg,positive_recall=float(on.float().mean()),recorded_owner_selection=float((nearest==co).float().mean()),joint_recorded_owner_and_ON=float(((nearest==co)&on).float().mean()),OOD_false_trigger=None,OOD_rejection=None,qualification='POSITIVE_ONLY_DEV_PROXY_MULTIPLE_COMPATIBLE_OWNERS_UNKNOWN'));weights.append(w)
    best=max(range(3),key=lambda i:(results[i]['joint_recorded_owner_and_ON'],results[i]['positive_recall'],-i));dist=ca.mul(sizes).sum(-1).sqrt();nearest=dist.argmin(-1)
    radius=[]
    for scale in [.5,.75,1.]:
        on=dist[torch.arange(len(c)),nearest]<=radii[nearest]*scale;radius.append(dict(router='R_RADIUS',scale=scale,positive_recall=float(on.float().mean()),recorded_owner_selection=float((nearest==co).float().mean()),joint_recorded_owner_and_ON=float(((nearest==co)&on).float().mean()),OOD_false_trigger=None,OOD_rejection=None,qualification='POSITIVE_ONLY_DEV_DIAGNOSTIC'))
    chosen=next((x['scale'] for x in radius if x['positive_recall']>=.98),1.)
    save(RUN/'private/ROUTE_DIAGNOSTIC_CONFIG.pt',dict(R_RADIUS_scale=chosen,R_SCOPE_weights=weights[best],R_SCOPE_scale=1.,group_sizes=sizes,selection='Recorded owner+ON then positive recall; no negative claims',candidate_status='NOT_CONFIRMED_SCOPE_SUPERVISION_BLOCKED'))
    write(RUN/'public/B_CAL_DIAGNOSTIC.json',dict(status='CPU_DIAGNOSTIC_COMPLETE_NOT_METHOD_QUALIFICATION',source='Existing native/fit positive information only; three fit paraphrases train and fourth CAL; no T1G/T2G/CHECK training',shared_trainable_scalars=32,configs_scope=results,configs_radius=radius,selected_scope_config=best,selected_radius=chosen,OOD_status='NA_NO_VERIFIED_SCOPE_NEGATIVES',scope_train_sources=24,scope_cal_sources=24,protection_TRAIN_sources=0,protection_CAL_sources=0,protection_CHECK_sources=0,candidate='NOT_CONFIRMED',normal_bank_generation='NOT_RUN_YET',independent_confirmation='BLOCKED_CONFIRMATION'))
    print('ROUTE_CAL_DIAGNOSTIC_PASS',flush=True)

if __name__=='__main__':main()
