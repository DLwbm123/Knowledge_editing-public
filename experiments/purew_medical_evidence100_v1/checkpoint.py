"""Lossless compressed atomic state; live statistics retain their original precision."""
import gzip
import os
from pathlib import Path
import torch
import common as c


def read(path):
    p=Path(path)
    with p.open('rb') as f:compressed=f.read(2)==b'\x1f\x8b'
    if compressed:
        with gzip.open(p,'rb') as f:return torch.load(f,map_location='cpu',weights_only=True)
    return torch.load(p,map_location='cpu',weights_only=True)


def write(path,state):
    c.budget();p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.'+str(os.getpid())+'.tmp')
    with gzip.open(temp,'wb',compresslevel=1) as f:torch.save(state,f)
    size=sum(x.stat().st_size for x in (c.RUN/'private').rglob('*.pt') if not x.is_symlink())+temp.stat().st_size
    if size>=c.read(c.RUN/'RUN_MANIFEST.json')['owned_weight_limit_bytes']:
        temp.unlink();raise OSError('Compressed checkpoint + atomic temporary exceed owned cap')
    with temp.open('rb') as f:os.fsync(f.fileno())
    temp.replace(p)
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    with c.resources() as r:r['weights_observed_peak_bytes']=max(r.get('weights_observed_peak_bytes',0),size)


def exact(a,b):
    if isinstance(a,torch.Tensor):return a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b
