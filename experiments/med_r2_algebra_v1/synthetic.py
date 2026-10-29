"""Synthetic Frobenius-space checks for a fixed-energy two-direction penalty."""
import json
import math
import os
from pathlib import Path
import random

rng = random.Random(20261029)
dim=12

def dot(a,b):
    return sum(x*y for x,y in zip(a,b))
def normalize(x):
    n=math.sqrt(dot(x,x))
    if n < 1e-10:
        raise ValueError('degenerate direction')
    return [v/n for v in x]
def construct(u,v,E):
    a=normalize(u)
    coefficient=dot(a,v)
    b=normalize([x-coefficient*y for x,y in zip(v,a)])
    scale=math.sqrt(E/2)
    return [x*scale for x in a],[x*scale for x in b]

assert dim==12
try:
    construct([1.0]+[0.0]*(dim-1),[2.0]+[0.0]*(dim-1),1.0)
    raise AssertionError('collinear candidate accepted')
except ValueError:
    pass
max_error=0.0
for _ in range(64):
    u=[rng.gauss(0,1) for _ in range(dim)];v=[rng.gauss(0,1) for _ in range(dim)]
    energy=1.0+rng.random()
    a,b=construct(u,v,energy)
    assert abs(dot(a,b))<1e-10 and abs(dot(a,a)+dot(b,b)-energy)<1e-10
    assert abs(dot(a,a)-energy/2)<1e-10 and abs(dot(b,b)-energy/2)<1e-10
    matrix=[[a[i]*a[j]+b[i]*b[j] for j in range(dim)] for i in range(dim)]
    for _ in range(4):
        delta=[rng.gauss(0,1) for _ in range(dim)]
        factored=dot(a,delta)**2+dot(b,delta)**2
        dense=sum(delta[i]*matrix[i][j]*delta[j] for i in range(dim) for j in range(dim))
        max_error=max(max_error,abs(factored-dense))
        assert dense >= -1e-10 and abs(factored-dense)<1e-9
out=Path(os.environ['AUDIT_OUTPUT']);assert not out.exists()
out.write_text(json.dumps({'status':'MECHANICAL_PASS','seed':20261029,'cases':64,'updates_per_case':4,'dimension':dim,'max_dense_difference':max_error,'trace_equal':True,'collinear_rejected':True,'clinical_or_model_efficacy':False},indent=2)+'\n')
