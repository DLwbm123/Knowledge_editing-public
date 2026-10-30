import json,os,sys
from pathlib import Path
w=Path(os.environ['FIXED_WORKTREE']); root=Path(os.environ['FIXED_SCORE_ROOT']); private=w/'.private/med_fixed_depth_v1_20261030'
sys.path.insert(0,str(w/'experiments/med_fixed_depth_v1'))
from report import aggregate,contrast
lock=json.loads((private/'AGREEMENT_DECISIONS_FROZEN.json').read_text())
if lock['status']!='FROZEN' or len(lock['rows'])!=508: raise ValueError('freeze missing')
choice={(r['owner'],r['query_id']):r['selected'] for r in lock['rows']}
old=json.loads((private/'BASE_M0_SCORE_BINDINGS.json').read_text())['rows']
packet=json.loads((root/'PACKET_PRIVATE.json').read_text())['consumers']
scored=json.loads((root/'DECISIONS.json').read_text())['decisions']
roles=json.loads((private/'EVALUATION_ROLES.json').read_text())
values={}
for r in old:
 for a in ('BASE','MED'):
  x=r[a];values[(r['owner'],r['query_id'],a)]={k:x[k] for k in ('core_correct','full_supported')}
for r in packet:
 key=(r['owner'],r['query_id'],r['arm']);x=scored.get(r['key'])
 values[key]={k:x.get(k) if x else None for k in ('core_correct','full_supported')}
 if r['arm']=='CDEEP' and r['prediction_reused_from'] and choice[(r['owner'],r['query_id'])]!='MED': raise ValueError('OFF not preserved')
if len(values)!=2032 or len(choice)!=508: raise ValueError('coverage')
for (o,q),a in choice.items(): values[(o,q,'GUARD')]=values[(o,q,a)]
events=[(r['owner'],r['query_id']) for r in old]
fields=('core_correct','full_supported','joint_supported')
for v in values.values():
 v['joint_supported']=None if None in (v['core_correct'],v['full_supported']) else v['core_correct'] and v['full_supported']
panels={}
for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
 es=[(int(o),q) for o,groups in roles.items() for q in groups[role]]
 for f in fields:
  panels[role+'_'+f]={'arms':{a:aggregate([(o,values[(o,q,a)][f]) for o,q in es]) for a in ('BASE','MED','C21','CDEEP','GUARD')},'guard_vs_med':contrast([(o,values[(o,q,'MED')][f],values[(o,q,'GUARD')][f]) for o,q in es]),'guard_vs_c21':contrast([(o,values[(o,q,'C21')][f],values[(o,q,'GUARD')][f]) for o,q in es])}
f='core_correct';t2=panels['T2G_'+f]; safety=[panels[role+'_'+f]['guard_vs_med'] for role in ('T0','FIT','T2L')]
common=[(int(o),q) for o,groups in roles.items() for q in groups['T2L'] if values[(int(o),q,'BASE')][f] is True and values[(int(o),q,'MED')][f] is True]
protection=contrast([(o,True,values[(o,q,'GUARD')][f]) for o,q in common])
full=contrast([(o,values[(o,q,'MED')]['full_supported'],values[(o,q,'GUARD')]['full_supported']) for o,q in events])
valid=all(v is not None for x in values.values() for v in x.values())
a=t2['guard_vs_med']['macro_delta_pp'];b=t2['guard_vs_c21']['macro_delta_pp']
passed=valid and a is not None and b is not None and a>=2 and b>=0 and all(x['events'] and x['losses']==0 for x in safety) and protection['events'] and protection['losses']==0 and full['losses']==0
out={'status':'DEV_SCREEN_PASS' if passed else 'DEV_SCREEN_NOT_PASSED' if valid else 'INCOMPLETE','events':508,'owners':16,'choices':{a:sum(x==a for x in choice.values()) for a in ('MED','CDEEP')},'panels':panels,'common_t2l_core':protection,'all_events_full_vs_med':full,'new_training':0,'new_generations':0,'new_judge':0,'limitations':['post hoc exposed DEV','full payload key agreement proxy, not verified semantic agreement','precomputed dual arm outputs; runtime incurs multiple model predictions','original protection curvature absent in deep arm','patient independence and clinical validity unknown']}
p=w/'reports/med_agreement_guard_v1_20261031/RESULTS.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':out['status'],'choices':out['choices'],'t2g_med':a,'t2g_c21':b,'safety':safety,'protection':protection,'full':full},ensure_ascii=False))
