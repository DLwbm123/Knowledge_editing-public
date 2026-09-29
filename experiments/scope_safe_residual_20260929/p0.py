"""Read completed artifacts only; write independent private audit and public aggregates."""
import json,hashlib,collections
from pathlib import Path

def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def bucket(a,b):
 x,y=a['activated'],b['activated']
 return 'OFF_OFF' if not x and not y else 'OFF_ON' if not x else 'ON_OFF' if not y else 'ON_ON_SAME' if a['logical_edit_id']==b['logical_edit_id'] else 'ON_ON_DIFFERENT'
def outcome(a,b):return 'missing' if a is None or b is None else 'both_correct' if a and b else 'correct_to_error' if a else 'error_to_correct' if b else 'both_error'
def main(parent,appendix,out):
 all_agg=[];mismatches=[];unmatched=0;parity=collections.Counter();imports=[]
 for phase,root in [('parent',parent),('appendix_completed_core',appendix)]:
  orders={t['canonical_edit_id']:t['order'] for t in read(root/'private/TASKS_R2_LOCKED.json')['tasks']};queue=read(root/'QUEUE.json');selected=[j for j in queue if j['status']=='COMPLETE' and (phase=='parent' or j.get('block') in ['E2_REG24','E1_E2_DEV24'] or j['id'].startswith('E1-'))]
  paths=[p for j in selected for p in (root/'jobs'/j['id']).rglob('CONSUMERS.json')]
  rows=[x for p in paths for x in read(p) if x['mode'] in ['single','sequential'] and (x['mode']=='single' or x['prefix'] in [12,24])]
  needed={x[k] for x in rows for k in ['judge_key','base_judge_key']};scores={k:read(root/'private/judge/scores'/f'{k}.json')['is_correct'] for k in needed if (root/'private/judge/scores'/f'{k}.json').exists()}
  imports.append(dict(phase=phase,completed_jobs=len(selected),consumer_files=len(paths),rows=len(rows),required_keys=len(needed),scored=len(scores)))
  def identity(x):return (x['seed'],'DEV24' if x['order']<=24 else 'REG24',x['arm'],x['edit'],x['task'],x['query_id'],x['input_id'])
  singles={identity(x):x for x in rows if x['mode']=='single'};groups={}
  for b in rows:
   if b['mode']!='sequential':continue
   a=singles.get(identity(b))
   if a is None:unmatched+=1;continue
   ga=read(root/'private/generations'/f'{a["execution_id"]}.json');gb=read(root/'private/generations'/f'{b["execution_id"]}.json');ba,bb=ga['execution_binding'],gb['execution_binding']
   common=['input','model','precision','backend','science_id','forced_diagnostic']
   compatible=all(ba.get(k)==bb.get(k) for k in common)
   if not compatible:
    parity['incompatible_binding']+=1
    for k in common:
     if ba.get(k)!=bb.get(k):parity['incompatible_'+phase+'_'+k]+=1
    continue
   # Resolve executed expert from the frozen router entry index, not associated edit.
   def effective(x,bind):
    if not x['route']['activated']:return 'BASE'
    order=orders[x['route']['logical_edit_id']];items=[a for a in bind['bank'] if Path(a['adapter']).name==f'e{order:03d}.pt'];assert len(items)==1;return digest(items[0])
   ea,eb=effective(a,ba),effective(b,bb)
   if ea==eb:
    same=ga['output']['raw_token_ids']==gb['output']['raw_token_ids'];parity['same_effective_execution']+=1;parity['token_equal' if same else 'token_different']+=1
    if not same:mismatches.append(dict(phase=phase,single_execution=a['execution_id'],bank_execution=b['execution_id'],effective=ea,binding_fields_checked=common))
   role='scope_positive' if b['task'] in ['T0','T1G','T2G'] else 'scope_negative' if b['task'] in ['T1L','T2L','T2L_PRESSURE'] else 'scope_unknown'
   base=scores.get(b['base_judge_key']);ka=(b['seed'],identity(b)[1],b['arm'],b['prefix'],role,b['task'],str(base),bucket(a['route'],b['route']))
   g=groups.setdefault(ka,dict(counts=collections.Counter(),inputs=set(),edits=set(),sources=set(),eligible=0,bank_active_base_correct_error=0,oracle_correct=0,oracle_missing=0))
   av,bv=scores.get(a['judge_key']),scores.get(b['judge_key']);g['counts'][outcome(av,bv)]+=1;g['inputs'].add(b['input_id']);g['edits'].add(b['edit']);g['sources'].add(b['source_group'])
   g['eligible']+=int(base is (True if role=='scope_negative' else False)) if role!='scope_unknown' else 0
   g['bank_active_base_correct_error']+=int(b['route']['activated'] and base is True and bv is False)
   ov=base if role=='scope_negative' else bv;g['oracle_missing']+=ov is None;g['oracle_correct']+=ov is True
  for key,g in groups.items():all_agg.append(dict(phase=phase,seed=key[0],panel=key[1],method=key[2],prefix=key[3],scope_role=key[4],task=key[5],Base_correct=key[6],transition=key[7],consumer_count=sum(g['counts'].values()),unique_inputs=len(g['inputs']),edits=len(g['edits']),source_clusters=len(g['sources']),patient_clusters=None,patient_status='source proxy only; consumer has no patient id',eligible=g['eligible'],outcomes=dict(g['counts']),bank_active_base_correct_error=g['bank_active_base_correct_error'],oracle_reject=dict(correct=g['oracle_correct'],missing=g['oracle_missing'],label='diagnostic only, not deployable or overall theoretical upper bound')))
 write(out/'public/FAILURE_DECOMPOSITION.json',dict(imports=imports,unmatched=unmatched,groups=all_agg,legacy_negative_activation_damage='bank activated AND Base correct AND bank wrong; not necessarily new error versus single'))
 write(out/'public/IDENTICAL_ROUTE_PARITY.json',dict(counts=dict(parity),status='REQUIRES_DIAGNOSIS' if mismatches else 'PASS_OBSERVED_PATHS',scope='Exact input/model/precision/backend/science/forced setting plus executed adapter binding; token comparison, no reuse authorization',mismatches=len(mismatches)))
 write(out/'private/PARITY_MISMATCHES.json',mismatches)
 counts=collections.Counter()
 for g in all_agg:
  for k,v in g['outcomes'].items():counts[g['transition']+' / '+k]+=v
 (out/'public/P0_FINDINGS.md').write_text('# P0 故障分解\n\n只读取已完成父与E1/E2，排除E3/E4。按seed/panel/method/edit/task/query/input及实际生成绑定配对；QA/前缀重复不作为独立病例。\n\n'+json.dumps(dict(counts),ensure_ascii=False,indent=2)+'\n\n患者ID在消费者记录中缺失，来源聚类只作代理。相同有效执行的输出差异见IDENTICAL_ROUTE_PARITY.json；有差异时不直接授权反事实缓存复用。oracle仅诊断，不是可部署结果。\n')
if __name__=='__main__':
 import sys
 main(*(Path(x) for x in sys.argv[1:]))
