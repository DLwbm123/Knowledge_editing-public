"""Bounded read-only identity audit; no new inference or scoring."""
import pathlib,json,sqlite3,hashlib,collections,time,os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
old=pathlib.Path(os.environ['PAPER_PARENT']);run=pathlib.Path(os.environ['RUN_ROOT'])
read=lambda p:json.loads(p.read_text())
digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
original=read(old/'private/BENCHMARK146_QUEUE.json')['tasks'];packet=read(run/'private/M13_SOURCE_PACKET.json')
assert len(original)==len(packet['original_tasks'])==146
fields=('edit_id','order','native','fit_questions','seed','events')
assert all(all(a[k]==b[k] for k in fields) for a,b in zip(original,packet['original_tasks']))
ledger=read(old/'private/EVAL_LEDGER.json');assert all(ledger['Base_correctness'][k]==v for k,v in packet['base_mask'].items())
routers={}
for t in original:
 path=old/'private/edits'/t['anonymous_edit']/'ROUTER.pt'
 saved=torch.load(path,map_location='cpu',weights_only=True)
 assert len(saved['entries'])==1 and saved['entries'][0]['logical_edit_id']==t['edit_id']
 e=saved['entries'][0];key=torch.as_tensor(e['key']);assert torch.isfinite(key).all() and float(e['radius'])>0
 routers[str(t['order'])]=str(path)
recovery=old/'private/judge_astra_medium_recovery1'
assert read(recovery/'REPORT_COMPLETE.json')['status']=='COMPLETE_NO_MISSING_AND_PUBLISHED'
with sqlite3.connect('file:'+str(recovery/'queue.sqlite')+'?mode=ro',uri=True) as db:
 fixed={key:(json.loads(binding),status,score) for key,binding,status,score in db.execute('SELECT key,binding,status,correct FROM payload')}
assert len(fixed)==50 and all(v[1]=='FORMAT_VALID' for v in fixed.values())
summary={};index=[]
for method,queue in [('TT88_W0_RETRO146','judge_astra_medium'),('LORA_R1_W0_RETRO146','judge_lora146_astra_medium')]:
 q=old/'private'/queue;assert (q/'ALL_WORKERS_COMPLETE.json').exists()
 with sqlite3.connect('file:'+str(q/'queue.sqlite')+'?mode=ro',uri=True) as db:
  assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
  payloads={key:(json.loads(binding),status,correct) for key,binding,status,correct in db.execute('SELECT key,binding,status,correct FROM payload')}
  rows=db.execute('SELECT c.mode,c.prefix,c.edit_order,c.query_id,c.path,c.output_binding,c.payload_key FROM consumer c WHERE c.method=?',(method,)).fetchall()
 assert len(rows)==5722
 for key,(binding,status,correct) in payloads.items():
  assert digest(binding)==key
  assert binding['judge']['model']=='gpt-6-astra' and binding['judge']['reasoning_effort']=='medium'
 counts=collections.Counter();recovered=0;sampled=set()
 for mode,prefix,owner,qid,path,outkey,key in rows:
  binding,status,correct=payloads[key]
  assert binding['query_id']==qid and pathlib.Path(path).is_file()
  sample_key=(mode,prefix if mode=='bank_R0' else 0)
  if sample_key not in sampled:
   d=read(pathlib.Path(path));assert digest(d)==outkey
   inp,ji=d['binding']['input'],d['binding']['judge_input']
   expected=dict(query_id=inp['query_id'],question=inp['question'],reference=inp['reference'],image_sha256=inp['image_sha256'],image_path=ji['image_path'],prompt_ids=ji['prompt_ids'],attention_mask=ji['attention_mask'],runtime=ji['runtime'],generation=ji['generation'],output=d['R0'],judge=binding['judge'])
   assert binding==expected;sampled.add(sample_key)
  inherited_recovery=False;original_status=status
  if status=='MISSING' and key in fixed:
   bb,ss,score=fixed[key];assert bb==binding and ss=='FORMAT_VALID'
   status,correct=ss,score;recovered+=1;inherited_recovery=True
  assert (status=='FORMAT_VALID' and correct in (0,1)) or (status=='MISSING' and correct is None)
  counts[status]+=1
  index.append(dict(method=method,mode=mode,prefix=prefix,owner=owner,query_id=qid,path=path,output_binding=outkey,payload_key=key,status=status,correct=correct,original_status=original_status,authorized_historical_recovery=inherited_recovery))
 summary[method]=dict(consumers=len(rows),consumer_status=dict(counts),authorized_recovery_consumers=recovered,
    stored_full_payload_identity_verified=True,all_referenced_output_paths_present=True,
    freshly_checked_output_files=len(sampled),freshly_checked_output_files_scope='ONE_PER_MODE_AND_BANK_PREFIX',
    all_output_files_freshly_reverified=False,original_payloads_not_rejudged=True,
    comparison_type='HISTORICAL_DIFFERENT_TRAINING_RECIPE',matched_native_FIT160_control=False,
    parameters=7168 if method.startswith('TT88') else 18432)
receipt=dict(status='PASS_READONLY_HISTORICAL_REUSE',original146_tasks_seed_events_and_Base_mask_exact=True,
    original_router_entries=146,methods=summary,new_GPU=0,new_VLM=0,new_Judge=0,
    original_missing_preserved=True,only_existing_explicitly_authorized_recovery_overlay=True,
    ready_for_historical_comparison=True,full_output_file_rescan_performed=False,new_output_score_alias_admitted=False,full146_execution_admitted=False,epoch=time.time())
(run/'private/HISTORICAL_146_REUSE_AUDIT1.json').write_text(json.dumps(receipt,indent=2))
(run/'private/HISTORICAL_146_SCORE_INDEX1.json').write_text(json.dumps(index))
(run/'private/HISTORICAL_R0_BINDINGS146.json').write_text(json.dumps(routers))
print(json.dumps(receipt))
