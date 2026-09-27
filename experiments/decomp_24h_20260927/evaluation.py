"""R3 real-bank generation retained with current-phase execution bindings."""
import time,hashlib
from dataclasses import asdict
from resources import ROOT,read,write
import worker_v3 as old
from freshstart import runtime as rt
SEED=20260927
def evaluate(runtime,tasks,bank,router,label,mode,prefix,folder,bindings,protocol,forced=False):
 from scripts.medtrace import stage15
 from methods.medtrace import MedTraceLayerHook
 consumers=[]
 router_entries=[dict(edit=edit,radius=radius,key_sha256=hashlib.sha256(key.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()) for edit,key,radius in zip(router.logical_ids,router.keys,router.radii,strict=True)]
 science=read(ROOT/'SCIENCE_LOCK.json')['id']
 for t in tasks:
  for row in t['evaluation']:
   rt.check_budget()
   raw,_,ib=stage15.prepared(runtime,row,old.record(t))
   # The router only receives the Base feature. Associated IDs are diagnostics below.
   decision,extra=router.diagnostic(old.key(runtime,row,old.record(t)));decision=asdict(decision)
   expert=bank.get(decision['logical_edit_id'])
   if forced:
    assert len(bank)==1;expert=next(iter(bank.values()))
   binding=dict(input=ib,model=protocol['model'],precision=protocol['precision'],backend=protocol['backend'],science_id=science,bank=bindings,router=dict(kappa=router.kappa,mu=router.mu,entries=router_entries),prefix=prefix,forced_diagnostic=forced)
   ident=old.digest(binding);path=ROOT/'private/generations'/f'{ident}.json'
   if path.exists():
    saved=read(path);assert saved['execution_binding']==binding;out=saved['output']
   else:
    if time.time()>=rt.epoch(read(ROOT/'RUN_MANIFEST.json')['no_new_generation_after']):raise TimeoutError('Hour 11: no new generation expansion')
    hook=MedTraceLayerHook(runtime.get_module(rt.LAYER),expert) if expert is not None else None
    if hook:hook.attach()
    try:out=stage15.generate(runtime,raw,ib,hook)
    finally:
     if hook:hook.detach()
    write(path,dict(execution_binding=binding,output=out))
   base,bj=old.base(runtime,row,old.record(t));assert base['binding']==ib,'Base cache input/protocol mismatch'
   consumers.append(dict(arm=label,seed=SEED,mode=mode,prefix=prefix,edit=t['canonical_edit_id'],order=t['order'],task=row['task'],query_id=row['query_id'],source_group=row['source_group'],input_id=old.input_id(row),execution_id=ident,judge_key=old.request(row,out),base_judge_key=bj,route=decision,route_diagnostics=extra,output=out,exact_Base_token_consistency=out['raw_token_ids']==base['raw_token_ids']))
 write(folder/'CONSUMERS.json',consumers)
 return consumers
