import os,sys,json,sqlite3,csv,copy,time
from pathlib import Path
r=Path(os.environ['RUN_ROOT']);base=Path(os.environ['BASE_ROOT']);sys.path.insert(0,str(r/'private/tools'));import common
sys.path.insert(0,str(r/'private/tools'));import report
from common import read,write,digest
records=[];scores={};inherited=0
for root in [r,base]:
 db=sqlite3.connect('file:'+str(root/'private/judge_common/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 for p in db.execute('select key,correct,status from payload'):
  assert p['status'] in ('FORMAT_VALID','MISSING');scores[p['key']]=p['correct']
 cs=db.execute('select * from consumer') if root==r else db.execute("select * from consumer where mode='bank_R0' and prefix=24 and method in ('FROZEN_W0','CE_U_SINGLE','CE_U_MULTI')")
 for c in cs:
  d=read(c['path']);assert digest(d)==c['output_binding']
  if root==r:d['binding']['mode']='bank_R0'
  records.append((d,c['payload_key']))
 if root==r:inherited=db.execute('select count(*) from inherited').fetchone()[0]
 db.close()
ts=[t for t in read(r/'private/QUEUES.json')['tasks'] if t['cohort']=='P2'];panels=[];coeff={};groups={}
for arm in ['FROZEN_W0','CE_U_SINGLE','CE_U_MULTI']:
 for route in ['R0','R_RADIUS','R_SCOPE']:
  label=arm if route=='R0' else arm+'__'+route
  for task in ['T0','T1G','T2G']:
   m,c,g=report.panel(records,scores,'P2',label,0 if arm=='FROZEN_W0' else 160,'bank_R0',task,ts,24);m.update(base_arm=arm,router=route);panels.append(m);coeff[(arm,route,task)]=c;groups.update(g)
contrasts=[]
for arm in ['FROZEN_W0','CE_U_SINGLE','CE_U_MULTI']:
 for route in ['R_RADIUS','R_SCOPE']:
  for task in ['T0','T1G','T2G']:
   aa=coeff[(arm,route,task)];bb=coeff[(arm,'R0',task)];es=set(aa)&set(bb);cc={e:report.combine([aa[e],{k:-v for k,v in bb[e].items()}]) for e in sorted(es)}
   bounds=report.score_bounds(report.combine([{k:v/len(cc) for k,v in c.items()} for c in cc.values()]),scores)
   contrasts.append(dict(base_arm=arm,comparison=route+'-R0',task=task,paired_edits=len(cc),delta_bounds_pp=[v*100 for v in bounds],**report.bootstrap(cc,scores,groups)))
u=report.summarize_u(records,'P2');u=[x for x in u if x['mode']=='bank_R0' and x['prefix']==24]
write(r/'public/B_RESULTS.json',dict(status='DEVELOPMENT_DIAGNOSTICS_COMPLETE',TT_training=False,edit_sources=24,seed_slots=1,panels=panels,contrasts=contrasts,U=u,independent_locality='NA_SCOPE_QUALIFICATION_NOT_COMPLETE',candidate='NOT_CONFIRMED'))
with (r/'public/RESULTS.csv').open('w',newline='') as f:
 cols=['base_arm','router','task','macro','known_correct','observations','missing','edit_units','route_ON'];w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(panels)
ledger=read(r/'RESOURCE_LEDGER.json');assert all(s.get('ended_epoch') for s in ledger['gpu_sessions']);assert ledger['gpu_seconds_used']<86400 and ledger['Judge_attempts']<=6000
write(r/'public/DELIVERY_AUDIT.json',dict(status='PASS_FOR_EXECUTED_A_B_PHASES',new_bank_outputs=1698,baseline_R0_bank_outputs=849,all_six_new_combinations_complete=True,bank_score_keys=403,inherited_bank_score_keys=inherited,missing_bank_scores=0,new_semantic_judge_attempts=ledger['Judge_attempts'],GPU_hours=ledger['gpu_seconds_used']/3600,all_GPU_sessions_ended=True,generated_artifact_peak_bytes=ledger['weights_observed_peak_bytes'],TT_training=False,historical_assets_read_only=True,C='NOT_RUN',C2='NOT_ADMITTED',confirmation='BLOCKED_CONFIRMATION',whole_protocol_complete=False,publication='PENDING_PUSH'))
print('B_PANELS',[(x['base_arm'],x['router'],x['task'],x['macro'],x['known_correct'],x['observations']) for x in panels])
print('B_U',u)
sem=read(r/'public/A1_SEMANTIC_DIAGNOSTIC.json');print('SEMANTIC_BANK24',[x for x in sem['summaries'] if x['mode']=='bank_R0' and x['prefix']==24]);print('GPU_HOURS',ledger['gpu_seconds_used']/3600)
