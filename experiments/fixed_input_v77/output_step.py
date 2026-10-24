"""Train G1/G2 while preserving the G3/G4 input factor exactly."""
import torch
from path_v18 import proposal

def reframe_output(r,state):
 core=state['G2'].detach().cpu().double();matrix=core.flatten(1)
 frame,_=r.mm.canonical_frame(matrix);transfer=matrix@frame.T
 root=torch.einsum('aib,bc->aic',state['G1'].detach().cpu().double(),transfer)
 return dict(G1=root.to(state['G1']),G2=frame.reshape_as(core).to(state['G2']),G3=state['G3'].clone(),G4=state['G4'].clone())

def output_step(r,before,gradient,scalar):
 assert list(before)==['G1','G2','G3','G4'] and gradient.numel()==7168 and torch.isfinite(gradient).all()
 masked=torch.zeros_like(gradient);n=before['G1'].numel()+before['G2'].numel();assert n==2560
 masked[:n]=gradient[:n]
 raw,updated,_=proposal(r,before,masked,scalar,list(before))
 result=reframe_output(r,raw) if masked[:n].count_nonzero() else {k:v.clone() for k,v in before.items()}
 assert all(torch.equal(result[k],before[k]) for k in ('G3','G4'))
 return result,updated

def selfcheck():
 import module_math as mm
 from alignment import TT,snapshot
 from types import SimpleNamespace
 r=SimpleNamespace(mm=mm);initial=mm.canonicalize(snapshot(TT(225034559)))[0]
 g=torch.arange(7168,dtype=torch.float64)/7168;s=dict(step=0,second_moments=[0.,0.]);a,sa=output_step(r,initial,g,s)
 gg=g.clone();gg[2560:]=1000;b,sb=output_step(r,initial,gg,s)
 assert sa==sb and all(torch.equal(a[k],b[k]) for k in a) and sa['second_moments'][1]==0 and s['step']==0
 c,sc=output_step(r,a,g,sa);assert sc['step']==2 and all(torch.equal(c[k],initial[k]) for k in ('G3','G4'))
 masked=g.clone();masked[2560:]=0;raw,_,_=proposal(r,a,masked,sa,list(a));rr=reframe_output(r,raw)
 factor=lambda x:torch.einsum('ia,ajb->ijb',x['G1'][0].double(),x['G2'].double()).reshape(4096,4)
 torch.testing.assert_close(factor(rr),factor(raw),rtol=2e-6,atol=1e-7)
 z,_=output_step(r,initial,torch.zeros_like(g),s);assert all(torch.equal(z[k],initial[k]) for k in z)
 assert c['G2'].shape==initial['G2'].shape and not torch.equal(c['G2'],initial['G2'])
 return dict(status='PASS',active_output2560=True,input_cores_bitwise_fixed=True,ignored_input_gradients=True,output_factor_reframe_equivalence=True,second_RMS_group_zero=True,zero_step_exact=True,scalar_no_alias=True,real_model_calls=0)
