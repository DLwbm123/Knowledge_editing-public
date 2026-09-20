"""Inference-only responsibility diagnostic; every prefix sees only its existing experts."""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage19_fasttrack_budget import read,write,check
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_score import query_id

def run(cfg):
 from scripts.medtrace.stage17_single import setup
 setup(cfg)
 import torch,shutil
 from dataclasses import replace,asdict
 from scripts.medtrace.stage19_fasttrack import load,record_for
 from scripts.medtrace import stage15
 from scripts.medtrace.stage18_cfact import assert_base_off
 from methods.medtrace import AsymmetricCPExpert
 from methods.medtrace.selective_write import LowRankExpert
 from methods.medtrace.hsic import bound_hook
 from m3bench_repro.editors.routing import MemoryRouter,euclidean_distances
 root=Path(cfg['run']);p=root/'private';pub=root/'public';common=Path(cfg['common_run']);stream=read(common/'private/STREAM.json');tasks=stream['tasks'];assert digest(stream)==cfg['stream_binding']
 prior=torch.load(Path(cfg['stage20_run'])/'private/BANKS.pt',map_location='cpu',weights_only=False);ids=[t['canonical_edit_id'] for t in tasks];pos={e:i+1 for i,e in enumerate(ids)}
 banks={(19,a):torch.load(common/'private'/f'BANKS_{a}.pt',map_location='cpu',weights_only=False) for a in ('E0','E2')};banks[19,'R_H']=torch.load('/root/rivermind-data/job-524c/run/private/BANKS_R_H.pt',map_location='cpu',weights_only=False)
 banks.update({(45,a):torch.load(common/'private'/f'BANKS_23_{a}.pt',map_location='cpu',weights_only=False) for a in ('E0','E2')})
 # Actual per-writer sparse training terms, rather than subtracting gradient norms.
 diag=[]
 for t in tasks[:19]:
  s=banks[19,'R_H']['experts'][t['canonical_edit_id']];v=torch.load(s['path'],map_location='cpu',weights_only=True);assert v['step']==320
  for c in v['curve']:
   if c['step'] in (1,80,160,320):diag.append(dict(position=t['order'],step=c['step'],training_terms={k:{x:y for x,y in z.items() if x!='sample'} for k,z in c['terms'].items()},total_preclip_gradient=c['gradient_norm'],regularization={k:z for k,z in c['regularization'].items() if k!='masks'}))
 write(pub/'ALL_WRITER_DIAGNOSTICS.json',dict(rows=diag,n_writers=19,n_steps=76,gradient_total_includes_regularizer=True))
 for (n,a),bank in banks.items():
  assert bank['inserted']==ids[:n]
  for t in tasks[:n]:
   s=bank['experts'][t['canonical_edit_id']];v=torch.load(s['path'],map_location='cpu',weights_only=True)
   assert v['step']==320 and digest(v['binding']['task'])==digest(t) and v['binding']['W0']==s['W0'] and v['binding']['code']==s['code'] and s['layer_id']==30
 write(pub/'BANK_ACCEPTANCE.json',dict(banks=[dict(N=n,arm=a,experts=len(b['experts'])) for (n,a),b in banks.items()],exact_task_W0_code_step=True))
 torch.use_deterministic_algorithms(True);rt=load(cfg);frozen=[(v,v._version) for v in rt.model.parameters()]
 def guard():
  check(cfg);assert_base_off(rt);assert all(v._version==n and not v.requires_grad for v,n in frozen)
  assert shutil.disk_usage(root).free>=8*1024**3
  if (root/'STOP').exists():raise RuntimeError('Stop requested')
 record=record_for(tasks[0]);base_hist={r['query_id']:r['output'] for r in read(common/'private/HISTORICAL_BASE.json')['records']}
 for i in (1,19,45):
  guard();t=tasks[i-1];raw,_,b=stage15.prepared(rt,t['native'],record);o=stage15.generate(rt,raw,b);assert o['raw_token_ids']==base_hist[query_id(t['native'])]['raw_token_ids']
 write(pub/'ENVIRONMENT_ACCEPTANCE.json',dict(Base_OFF_tokens_exact=[1,19,45],gpu_uuid=cfg['gpu_uuid'],training=False))
 rows={query_id(r):r for t in tasks for role in ('H_fit','U_fit') for r in t[role]};features=torch.load(common/'private/H_COVERAGE_FEATURES.pt',map_location='cpu',weights_only=True)['features'];keys={};base={}
 for q,r in rows.items():
  guard()
  if q in features:keys[q]=features[q]['key']
  else:
   with torch.inference_mode():keys[q]=rt.extract_layer_input_key(rt.build_question_batch(replace(record,question=r['question'],target='',image_path=Path(r['image_path']))),module_path='model.layers.31.mlp.up_proj',pooling='mean').cpu()
  raw,_,b=stage15.prepared(rt,r,record);base[q]=stage15.generate(rt,raw,b)
 write(p/'BASE.json',dict(records=[dict(query_id=q,source=rows[q],output=o) for q,o in base.items()]))
 routers={n:MemoryRouter.from_state(dict(distance='euclidean',entries=[prior['routes'][e] for e in ids[:n]]),device=rt.device) for n in range(1,46)}
 ownership=[];history=[];expert=LowRankExpert(AsymmetricCPExpert(14336,4096,4),20260912,rank=4).to(rt.device).requires_grad_(False);count=0
 def emit(n,arm,role,q,mode,selected,owner=None):
  nonlocal count
  guard();row=rows[q];out=base[q]
  if selected is not None:
   s=banks[n,arm]['experts'][selected];saved=torch.load(s['path'],map_location='cpu',weights_only=True);expert.load_state_dict(saved['expert']);raw,_,binding=stage15.prepared(rt,row,record)
   with bound_hook(rt,expert,dict(layer_id=30,expert_id=selected,W0_id=s['W0'])) as hook:out=stage15.generate(rt,raw,binding,hook)
  r=dict(N=n,prefix=n,arm=arm,role=role,mode=mode,query_id=q,source=row,output=out,Base=base[q],selected_position=pos.get(selected,0),owner_position=owner,diagnostic_only=True,supervision_present=not (role=='H_fit' and arm=='E0'),code=cfg['code_commit'])
  with (p/'OUTPUTS.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  count+=1
 for n in (19,45):
  for role in ('H_fit','U_fit'):
   owners={}
   for t in tasks[:n]:
    for r in t[role]:owners.setdefault(query_id(r),[]).append(t['order'])
   for q,own in owners.items():
    d=routers[n].route(keys[q].to(rt.device));winner=pos.get(d.logical_edit_id,0);insertion={}
    for k in range(min(own),n+1):
     dk=routers[k].route(keys[q].to(rt.device));known=[x for x in own if x<=k];h=dict(N=n,role=role,query_id=q,prefix=k,known_owners=known,winner=pos.get(dk.logical_edit_id,0),winner_is_any_owner=pos.get(dk.logical_edit_id,0) in known);history.append(h);insertion[k]=h['winner']
    for owner in own:
     route=prior['routes'][ids[owner-1]];ownership.append(dict(N=n,role=role,query_id=q,source_group=rows[q]['source_group'],owner_position=owner,owners=own,winner=winner,winner_is_this_owner=winner==owner,winner_is_any_owner=winner in own,own_radius=float(euclidean_distances(route['key'],keys[q])[0])<=route['radius'],winner_at_insertion=insertion[owner],switched=insertion[owner]!=winner))
    for arm in (('E0','E2','R_H') if n==19 else ('E0','E2')):
     emit(n,arm,role,q,'natural',d.logical_edit_id)
     for owner in own:emit(n,arm,role,q,'forced',ids[owner-1],owner)
    write(pub/'PROGRESS.json',dict(N=n,role=role,generated=count))
 write(p/'OWNERSHIP.json',dict(slots=ownership,history=history))
 write(pub/'GENERATED.json',dict(status='COMPLETE_NOT_SCORED',outputs=count,base_queries=len(base),no_training=True,no_24E_25=True,peak_allocated=torch.cuda.max_memory_allocated()))
if __name__=='__main__':run(read(os.environ['JOB_CONFIG']))
