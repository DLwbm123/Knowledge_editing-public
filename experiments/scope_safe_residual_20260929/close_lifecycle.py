"""Delete only the completed scope run's enumerated expendable checkpoints."""
import json,os,time
from pathlib import Path

def candidates(root):
 paths=[p for g in ['initializers','checkpoints','adapters/s20260929/AK','adapters/s20260929/AU','adapters/s20260929/AHS_001'] for p in (root/g).rglob('*.pt')]
 for p in paths:
  assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
  assert all(not q.is_symlink() for q in p.parents if q.is_relative_to(root))
 return sorted(paths)

def main():
 from resources import ROOT,read,write
 from storage import Store
 root=ROOT.resolve();assert root==Path('/data/bmw/Knowledge_editing/outputs/scope-safe-residual-20260929/run')
 assert not (root/'STOP').exists();assert (root/'SCORER_DONE').exists()
 assert all(x['status']=='COMPLETE' for n in ['QUEUE.json','HOLDOUT_QUEUE.json'] for x in read(root/n))
 ledger=read(root/'RESOURCE_LEDGER.json');assert not any(not x.get('ended_epoch') for x in ledger['gpu_sessions']);assert all(x['status'] in ['FORMAT_VALID','FAILED_NO_RETRY'] for x in ledger['judge_attempts'])
 assert read(root/'public/HOLDOUT_RESULTS.json')['status']=='COMPLETE' and read(root/'public/P5_DECISION.json')['status']=='NOT_ADMITTED'
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   assert p.joinpath('cwd').resolve()!=root,'Active process still uses this run'
  except (FileNotFoundError,PermissionError):pass
 files=[p for g in ['jobs','auxiliary/holdout'] for p in (root/g).rglob('CONSUMERS.json')]
 for p in files:
  for row in read(p):
   assert (root/'private/generations'/f"{row['execution_id']}.json").is_file()
   assert 'raw_token_ids' in row['output'] and 'raw_answer' in row['output']
 paths=candidates(root);manifest=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size) for p in paths]
 assert len(paths)==110,'Unexpected lifecycle set; inspect before deleting'
 store=Store(root)
 with store.lock() as d:
  before={m['path']:d['artifacts'].get(m['path']) for m in manifest}
  for a in before.values():
   if a:assert a['status']=='READY' and not a['readers'] and not a['consumers']
 write(root/'private/LIFECYCLE_DELETE_MANIFEST.json',dict(epoch=time.time(),files=manifest,registered_before=before,dependencies='P1/P2/P4 single/CHECK/banks and frozen holdout all complete; P5 not admitted',root=str(root)))
 for m in manifest:
  rel=m['path'];p=root/rel
  if before[rel]:
   with store.lock() as d:
    a=d['artifacts'][rel];assert a['status']=='READY' and not a['readers'] and not a['consumers'];a['pin']=False;a['unpin_reason']='all declared binary consumers complete; no training resume required'
   store.delete(rel)
  else:
   assert rel.startswith('initializers/') and p.stat().st_size==m['bytes'];p.unlink()
  m['deleted']=True
  write(root/'private/LIFECYCLE_DELETE_PROGRESS.json',manifest)
 retained=[p for p in (root/'adapters').rglob('*.pt')];assert len(retained)==192
 out=dict(status='COMPLETE',deleted_files=len(manifest),deleted_bytes=sum(x['bytes'] for x in manifest),deleted_categories=['48 independent initializer copies','6 inactive rolling optimizer checkpoints','24 AK + 24 AU + 8 unused-lambda adapters'],retained_adapters=len(retained),retained_bytes=sum(p.stat().st_size for p in retained),retained_methods=['E_orig','A0','AH','AHS_01'],retained_reason='Final paired baseline/simple-control/candidate banks for DEV24 and REG24, as execution plan requires',retained_key_features=True,private_outputs_scores_bindings_preserved=True,historical_files_deleted=False,restoration='Deleted adapter/optimizer states require deterministic recomputation; initializers can be copied from retained predecessor sources',raw_manifest_private=True)
 write(root/'public/CHECKPOINT_LIFECYCLE.json',out);print(json.dumps(out))
if __name__=='__main__':main()
