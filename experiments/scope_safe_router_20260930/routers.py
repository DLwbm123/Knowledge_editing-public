"""Frozen R0, insertion radius cap, and zero-threshold contrastive veto."""
from dataclasses import replace
import torch
from router_r3 import RejectRouter
from m3bench_repro.editors.routing import distances

class ScopeRouter(RejectRouter):
    def __init__(self,entries,method):
        super().__init__(1.,0.);assert method in ['R0','RCAP','NEG0','SAFE'];self.method=method;self.prototypes={}
        for e in entries:
            radius=e['safe_radius'] if method in ['RCAP','SAFE'] else e['radius']
            if radius<0:raise ValueError('UNSAFE_NO_RADIUS_SEPARATION')
            self.add(e['edit'],e['key'],radius);self.prototypes[e['edit']]=(e['positive'],e['negative'])
    def diagnostic(self,q):
        decision,extra=super().diagnostic(q)
        if self.method in ['NEG0','SAFE'] and decision.activated:
            positive,negative=self.prototypes[decision.logical_edit_id]
            if not negative.numel():raise ValueError('UNSUPPORTED_NEGATIVE_PROTO')
            dp=float(distances(positive.to(q.device),q,'euclidean').min())
            dn=float(distances(negative.to(q.device),q,'euclidean').min())
            quotient=(dn-dp)/(dn+dp+1e-12);active=quotient>=0
            decision=replace(decision,logical_edit_id=decision.logical_edit_id if active else None,activated=active)
            extra=dict(extra,d_pos=dp,d_neg=dn,q=quotient,tau=0,negative_veto=not active)
        return decision,extra

def cap_insert(previous,e,background):
    before=ScopeRouter(previous,'RCAP');trial=ScopeRouter(previous+[dict(e,safe_radius=e['radius'])],'RCAP')
    captures=[i for i,q in enumerate(background) if before.route(q).logical_edit_id!=e['edit'] and trial.route(q).logical_edit_id==e['edit']]
    safe=e['radius']
    if captures:
        # Match routing's float32 representation, including add()'s rounding.
        minimum=min(trial.route(background[i]).nearest_distance for i in captures)
        lower=torch.nextafter(torch.tensor(minimum,dtype=torch.float32),torch.tensor(float('-inf'),dtype=torch.float32)).item()
        safe=min(float(e['radius']),lower)
    result=dict(e,safe_radius=safe,capture_indices=captures)
    if safe<0:result['cap_status']='UNSAFE_NO_RADIUS_SEPARATION';return result
    frozen=ScopeRouter(previous+[result],'RCAP')
    result['cap_status']='PASS' if frozen.route(e['key']).logical_edit_id==e['edit'] else 'UNSAFE_NO_RADIUS_SEPARATION'
    assert all(frozen.route(background[i]).logical_edit_id!=e['edit'] for i in captures),'Captured boundary reactivated'
    return result

def tensors_to(entries,device):
    return [dict(e,**{k:e[k].to(device) for k in ['key','positive','negative']}) for e in entries]

def selfcheck():
    make=lambda edit,x,r:dict(edit=edit,key=torch.tensor([x,0.]),radius=r,safe_radius=r,positive=torch.tensor([[x,0.]]),negative=torch.tensor([[1.,0.]]))
    e=make('a',0.,2.);bg=torch.tensor([[1.,0.],[2.,0.]])
    c=cap_insert([],e,bg);assert c['safe_radius']<1 and c['cap_status']=='PASS'
    assert all(not ScopeRouter([c],'RCAP').route(q).activated for q in bg)
    assert ScopeRouter([c],'RCAP').route(e['key']).activated
    assert ScopeRouter([e],'NEG0').route(torch.tensor([.5,0.])).activated # tau equality inclusive.
    assert not ScopeRouter([e],'NEG0').route(torch.tensor([.75,0.])).activated
    z=cap_insert([],e,torch.zeros(1,2));assert z['cap_status']=='UNSAFE_NO_RADIUS_SEPARATION'
    try:ScopeRouter([dict(e,negative=torch.empty(0,2))],'NEG0').route(e['key'])
    except ValueError as ex:assert str(ex)=='UNSUPPORTED_NEGATIVE_PROTO'
    else:raise AssertionError('Missing negative must not fall back')
    state=torch.get_rng_state();ScopeRouter([c],'SAFE').diagnostic(e['key']);assert torch.equal(state,torch.get_rng_state())
if __name__=='__main__':selfcheck();print('PASS: strict radius, native, tau0, unsupported, RNG')
