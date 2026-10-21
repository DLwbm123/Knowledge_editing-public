"""Actual146 metadata through panel selection, queue ingestion and report; fake scores only."""
import json,os,shutil,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
payload=json.loads(sys.stdin.readline());started=time.monotonic()
payload['protocol']['cache_roots']=[]  # Fixture does not mutate or reinterpret real historical scores.
root=Path(tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp'));tools=root/'private/tools';tools.mkdir(parents=True)
parent=Path('/data/bmw/Knowledge_editing/outputs/purew-objective-scope-v1-20261015/run/private')
for p in (parent/'tools').glob('*.py'):shutil.copy2(p,tools/p.name)
for n,s in payload['files'].items():(tools/n).write_text(s)
os.environ.update(RUN_ROOT=str(root),ARM='HE',ACTION='CPU_REPORT_CHECK',CUDA_VISIBLE_DEVICES='')
sys.path[:0]=[str(tools),str(parent/'source')]
for name in ('BENCHMARK146_QUEUE.json','EVAL_LEDGER.json','EVAL_BINDINGS.json'):shutil.copy2(parent/name,root/'private'/name)
(root/'private/FROZEN_PROTOCOL.json').write_text(json.dumps(payload['protocol']))
import queue_revision as queue,report_revision
from configuration import ARMS,PREFIXES,ELIGIBLE,source_gate
from panels import expected_ids
q=queue.q
q.write(root/'private/GPU_SOURCE_VERSION.json',dict(commit='CPU_FIXTURE'))
q.write(root/'RESOURCE_LEDGER.json',dict(gpu_sessions=[],gpu_seconds_used=0,Judge_attempts=0,Astra_purew146_batches=[]))
tasks=q.read(root/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=q.read(root/'private/EVAL_LEDGER.json')
bindings=q.read(root/'private/EVAL_BINDINGS.json')
# Production queue must not ingest or score before both trajectories terminate.
q.ROOT.mkdir(parents=True);queue.ensure_manifest();db=q.connect();queue.ingest(db)
assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0
counts={}
for arm in ARMS:
 plans=[('insertion_R0',i,i) for i in range(1,147)]+[(m,0,n) for m in ('bank_R0','protect146_R0') for n in PREFIXES]
 counts[arm]=0
 for mode,owner,prefix in plans:
  phase=dict(arm=arm,mode=mode,owner=owner,prefix=prefix,protocol=q.digest(payload['protocol']),code='CPU_FIXTURE')
  folder=root/'private/outputs'/arm/mode/str(owner)/str(prefix);ids=expected_ids(tasks,ledger,mode,owner,prefix)
  counts[arm]+=len(ids)
  for key in ids:
   query=ledger['queries'][key];judge=bindings[query['opaque_Base_id']]
   q.write(folder/(key+'.json'),dict(binding=dict(input=query,judge_input=judge,phase=phase),R0=dict(raw_answer='CPU_fixture_'+arm,raw_token_ids=[1,2])))
  q.write(folder/'COMPLETE.json',dict(phase=phase,outputs=len(ids)))
 for owner in range(1,147):
  detail=dict(values=[dict(view=k//2,concept=k%2,margin=0.,full_logp=-1.,removed_logp=-1.) for k in range(10)]) if owner in ELIGIBLE else None
  q.write(root/'public'/('SOURCE_'+arm+'_'+str(owner)+'.json'),dict(final=dict(weighted_all_FIT_CE=.01),evidence_final=detail,evidence_initial=detail,gate=source_gate(.01,[0.]*10 if detail else [],bool(detail)),accepted=19,slots=20))
assert counts==dict(HM=4123,HE=4123)
q.write(root/'private/ARM_TERMINALS.json',{a:dict(consumers=4123) for a in ARMS})
q.write(root/'private/GENERATION_COMPLETE.json',dict(new_consumers=8246));q.write(root/'private/FINAL_CONSUMED.json',dict(status='PASS'))
queue.ingest(db);assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==8246
assert not queue.terminal_allowed(db)
db.execute("UPDATE payload SET status='FORMAT_VALID',correct=1");db.commit()
assert queue.terminal_allowed(db)
q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='CPU_FIXTURE'))
# No model/Judge is invoked: aggregation uses synthetic binary outcomes only.
import report_fs
with patch.object(report_fs,'audit_requests',lambda *a:None):report_revision.main()
result=q.read(root/'public/RESULTS.json')
assert result['coverage']=={a:{'FORMAT_VALID':4123} for a in ARMS}
assert not result['scientific_success'] and result['paired_no_degradation']
assert all(x['delta_bounds_pp']==[0.,0.] for x in result['contrasts'] if x['prefix']==146)
assert all(not result['gates'][a]['all_eligible_margins'] for a in ARMS)
# A shared payload missing in both arms cancels in a paired difference, but fails full scoring.
row=db.execute("SELECT payload_key FROM consumer WHERE method='HE' AND mode='bank_R0' AND prefix=146 LIMIT 1").fetchone()
db.execute("UPDATE payload SET status='MISSING',correct=NULL WHERE key=?",(row[0],));db.commit()
with patch.object(report_fs,'audit_requests',lambda *a:None):report_revision.main()
result=q.read(root/'public/RESULTS.json');assert result['status']=='COMPLETE_WITH_UNKNOWN' and not result['scientific_success']
shutil.rmtree(root)
print(json.dumps(dict(status='PASS',actual_metadata=True,consumers=8246,full_report=True,
 generation_barrier=True,scientific_failure_not_hidden=True,missing_denominator_retained=True,
 Judge_receipt_verification='NOT_EXERCISED_FAKE_SCORES',model_calls=0,GPU_calls=0,Judge_calls=0,CPU_seconds=time.monotonic()-started)))
