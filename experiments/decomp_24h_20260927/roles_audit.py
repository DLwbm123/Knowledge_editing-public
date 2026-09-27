"""Global full-input role audit against the retained official query inventory."""
import json,hashlib
from pathlib import Path
from collections import defaultdict

def run(root,inventory,candidate,g_file="private/G_SUPPORTS.json",output="fix/pr5_v1/ROLE_AUDIT_PRIVATE.json"):
 r=Path(root);read=lambda p:json.loads(Path(p).read_text());tasks=read(r/'private/TASKS_R2_LOCKED.json')['tasks'];gs={x['order']:x for x in read(r/g_file)};pool=read(candidate);required=set(pool['required_query_ids']);official={};found=set();candidate_inputs={}
 def identity(row):return (row.get('image_sha256') or row.get('image_path') or 'TEXT_ONLY',' '.join(row['question'].split()).casefold())
 for line in Path(inventory).open():
  x=json.loads(line)
  if x['query_id'] in required:
   candidate_inputs[identity(x)]=x['query_id'];found.add(x['query_id'])
   if x.get('role')=='frozen_formal_probe' or x.get('source_task','').startswith('T2'):official[identity(x)]=x['query_id']
 for t in tasks:
  for x in t['evaluation']+t['official_evaluation_full']:official[identity(x)]=x['query_id']
 roles=defaultdict(list);sources=defaultdict(set)
 for t in tasks:
  o=t['order'];native=t['native']
  for role,questions in [('P_fit',t['semantic_fit_questions']),('G_fit',gs[o]['G_fit']),('G_check',gs[o]['G_check'])]:
   for question in questions:roles[identity(dict(native,question=question))].append((o,role));sources[native['source_group']].add(role)
  for role in ['U_fit','U_new']:
   for x in t[role]:roles[identity(x)].append((o,role));sources[x['source_group']].add(role)
 collisions=[];candidate_overlap=[]
 for key,entries in roles.items():
  role_set={role for _,role in entries};forbidden=[]
  if key in candidate_inputs and key not in official:candidate_overlap.append(dict(entries=entries,query_id=candidate_inputs[key]))
  if 'G_fit' in role_set and 'G_check' in role_set:forbidden.append('G_fit_G_check')
  if role_set&{'G_fit','G_check'} and 'P_fit' in role_set:forbidden.append('G_P_fit')
  if key in official and role_set&{'G_fit','G_check','U_fit','U_new'}:forbidden.append('support_official_full')
  if role_set&{'U_fit','U_new'} and role_set&{'G_fit','G_check','P_fit'}:forbidden.append('positive_negative_role_conflict')
  if forbidden:collisions.append(dict(input_hash=hashlib.sha256(json.dumps(key).encode()).hexdigest(),entries=entries,reasons=forbidden,official_query_id=official.get(key)))
 affected=sorted({o for x in collisions for o,role in x['entries'] if role in ['G_fit','G_check','U_fit','U_new']})
 result=dict(status='PASS' if not collisions and required<=found else 'COLLISIONS' if collisions else 'INCOMPLETE_OFFICIAL_INVENTORY',official_required=len(required),official_found=len(found),official_missing=sorted(required-found),full_input_collisions=collisions,affected_orders=affected,candidate_native_only_overlap_count=len(candidate_overlap),candidate_overlap_note='Unused candidate source QA is not automatically a formal evaluation probe; recorded separately',source_role_overlap_groups=sum(len(v)>1 for v in sources.values()),source_overlap_note='Native/P/G may share a positive source by design; cross-role full-input collisions are separately blocked',G_content_hash=hashlib.sha256((r/g_file).read_bytes()).hexdigest())
 from storage import Store
 Store(r).write(output,json.dumps(result,indent=2).encode(),pin=True);print(json.dumps({k:v for k,v in result.items() if k not in ['full_input_collisions','official_missing']}));return result
