"""Synthetic shared-missing interaction and qualification checks, no formal scores."""
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import sys

RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
spec=importlib.util.spec_from_file_location('structured_metrics',RUN/'private/tools/reporting.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
assert len(r.ARMS)==6 and len(r.PAIRS)==7
assert r.gate_status([([1,1],1),([12,12],10)])=='SUPPORTED_ON_EXPOSED_PANEL'
assert r.gate_status([([0,1],1)])=='INCONCLUSIVE'
assert r.gate_status([(None,0)])=='INCONCLUSIVE'
assert r.gate_status([([0,0],1)])=='NOT_SUPPORTED'
# (TK_H-TK_NO)-(CP_H-CP_NO): identical shared missing must cancel exactly.
scores={'x':None,'y':1,'z':0};groups={'e1':'same','e2':'same'}
a={'e1':{'x':Fraction(1)},'e2':{'y':Fraction(1)}}
b={'e1':{'x':Fraction(1)},'e2':{'z':Fraction(1)}}
c={'e1':{'x':Fraction(1)},'e2':{'z':Fraction(1)}}
d={'e1':{'x':Fraction(1)},'e2':{'z':Fraction(1)}}
lhs={e:r.combine([a[e],d[e]]) for e in a};rhs={e:r.combine([b[e],c[e]]) for e in a}
out=r.paired(lhs,rhs,scores,groups)
assert out['macro_delta_bounds']==[.5,.5] and out['source_cluster_ci95_envelope']==[.5,.5]
zero=r.paired(lhs,lhs,scores,groups);assert zero['macro_delta_bounds']==[0,0]
# Different keys must retain uncertainty; never use successful-only subsets.
scores['u']=None
out=r.paired({'e1':{'x':Fraction(1)}},{'e1':{'u':Fraction(1)}},scores,groups)
assert out['macro_delta_bounds']==[-1,1]
print(json.dumps({'status':'PASS','checks':['six arms seven preregistered contrasts','three-valued gate including empty denominator','four-arm interaction shared missing cancellation','10000 paired edit/source bootstrap','different missing identities retain full bounds'],'formal_scores_read':0,'Judge_attempts':0,'GPU_loads':0}))
