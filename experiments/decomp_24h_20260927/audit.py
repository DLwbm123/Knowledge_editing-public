"""Small protocol and role audits; private rows never leave the execution root."""
import os,sys,json,importlib,hashlib
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(ROOT),str(ROOT/'source_patch'),str(ROOT/'source')]
from resources import read,write

def main():
 import torch,transformers
 from methods.medtrace import AsymmetricCPExpert
 from structures import TuckerC4,LR4,convert
 cp=AsymmetricCPExpert(14336,4096,4);tk=TuckerC4()
 with torch.no_grad():
  cp.rho.copy_(torch.arange(1,5)/10);tk.u1.copy_(cp.u_out);tk.u2.copy_(cp.v_out);tk.u3.copy_(cp.u_in);tk.u4.copy_(cp.v_in);tk.L.zero_();tk.R.zero_()
  for i in range(4):tk.L[i*4+i,i]=cp.rho[i]*cp.beta/2;tk.R[i*4+i,i]=1
  torch.manual_seed(20260927);h=torch.randn(8,14336);a=cp.residual(h);b=tk.residual(h);c=convert(tk,1).residual(h);error=float((a-b).abs().max());error2=float((b-c).abs().max());assert error<1e-5 and error2<1e-5
 write(ROOT/'public/CONVERSION_PARITY.json',dict(CP_to_Tucker_to_LR4_max_abs=[error,error2],status='PASS',scope='algebraic identity test, not a performance method'))
 write(ROOT/'public/ARCHITECTURE_AND_PARAM_AUDIT.json',dict(CP4=sum(p.numel() for p in cp.parameters()),TK_C4=sum(p.numel() for p in tk.parameters()),free_LR4=sum(p.numel() for p in LR4().parameters()),matrix_rank_upper_bound=4,shape=[64,64,112,128],normalization='h/(RMS(h)+epsilon)',standard_LoRA=False,all_stage_equal_parameters=False))
 paths={n:importlib.import_module(n).__file__ for n in ['scripts.medtrace.stage15','methods.medtrace','methods.medtrace.selective_write','freshstart.runtime','worker_v3','router_r3','m3bench_repro.editors.llava_runtime']}
 write(ROOT/'MODULE_PATHS.json',paths)
 write(ROOT/'VERSION_LOCK.json',dict(torch=str(torch.__version__),transformers=transformers.__version__,cuda=torch.version.cuda,modules={k:dict(path=v,sha256=hashlib.sha256(Path(v).read_bytes()).hexdigest()) for k,v in paths.items()},phase_source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}))
 tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];g={x['order']:x for x in read(ROOT/'private/G_SUPPORTS.json')};roles=[];private=[]
 def key(row):return (row.get('image_sha256',row.get('image_path')),row['question'].strip().lower())
 for t in tasks:
  old={key(x) for x in t['U_fit']};new={key(x) for x in t['U_new']};ev={key(x) for x in t['evaluation']};assert not (old|new)&ev,'U/evaluation overlap'
  private.append(dict(order=t['order'],Uold=len(old),Unew=len(new),U_intersection=len(old&new),P_fit=len(set(t['semantic_fit_questions'])),G_fit=len(g[t['order']]['G_fit']),G_check=len(g[t['order']]['G_check']),exposure='Method training/selection/final evaluation in earlier rounds; not independent CONFIRM'))
 for panel,ts in [('DEV24',tasks[:24]),('REG24',tasks[24:])]:
  bytask=defaultdict(list)
  for t in ts:
   for row in t['evaluation']:bytask[row['task']].append((t,row))
  masks={(r['edit'],r['task'],r['query_id']):r['base_correct'] for r in read(ROOT/'private/BASE_MASKS.json')['rows']}
  for task,rs in bytask.items():
   labels=[masks.get((t['canonical_edit_id'],task,r['query_id']),True if task=='T2L_PRESSURE' else None) for t,r in rs]
   roles.append(dict(panel=panel,task=task,consumers=len(rs),unique_inputs=len({key(r) for _,r in rs}),edits=len({t['canonical_edit_id'] for t,_ in rs}),source_groups=len({r['source_group'] for _,r in rs}),Base_correct=sum(x is True for x in labels),Base_wrong=sum(x is False for x in labels),unknown_Base=sum(x is None for x in labels),modality_counts=dict(Counter('image_text' if r.get('image_path') else 'text' for _,r in rs))))
 write(ROOT/'private/SUPPORT_AUDIT.json',private);write(ROOT/'public/DATA_ROLE_AUDIT.json',dict(existing_panel='EXPOSED_REGRESSION',independent_CONFIRM=False,roles=roles,support_count_histograms={k:dict(Counter(str(r[k]) for r in private)) for k in ['Uold','Unew','U_intersection','P_fit','G_fit','G_check']},new_panel='No new panel admitted without complete exposure/role and Base-mask qualification; current evidence does not establish an independent new cohort',raw_train_only_Base_exposed_not_automatically_excluded=True))
 print('PASS: conversion, parameters, actual imports, support isolation and frozen-role audit')
if __name__=='__main__':main()
