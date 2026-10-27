"""Compare retained source initializers with the actual random right-canonical recipe."""
import torch
from alignment import TT,snapshot
import module_math as mm

def audit(initializers):
 original=mm.canonicalize(snapshot(TT(225034559)))[0];rows=[]
 for owner,state in sorted(initializers.items()):
  a0=mm.factors({k:v.double() for k,v in original.items()})[1]
  a=mm.factors({k:v.double() for k,v in state.items()})[1]
  assert state['G1'].count_nonzero()==0
  errors={}
  for name,s in [('random',original),('supervised',state)]:
   core=s['G2'].double().reshape(8,256);gram=core@core.T
   errors[name]=float((gram-torch.eye(8,dtype=torch.float64)).abs().max())
   torch.testing.assert_close(gram,torch.eye(8,dtype=torch.float64),rtol=1e-6,atol=1e-6)
  torch.testing.assert_close(a,a0,rtol=1e-5,atol=1e-7)
  rows.append(dict(owner=owner,G2_row_gram_max_abs_error=errors,G2_frobenius={name:float(s['G2'].double().norm()) for name,s in [('random',original),('supervised',state)]},input_factor_max_abs_difference=float((a-a0).abs().max()),input_cores_bitwise_equal={k:torch.equal(state[k],original[k]) for k in ('G3','G4')}))
 assert len(rows)==10
 return dict(status='PASS',same_shapes_and_row_isometry=True,input_factor_numerically_matched=True,rows=rows,decision='REUSE_V64B_NO_DUPLICATE_RANDOM_ARM',limitation='A fixed seed isolates this implementation contrast, not seed-averaged superiority; FP32 recanonicalization may differ by rounding.',model_calls=0)
