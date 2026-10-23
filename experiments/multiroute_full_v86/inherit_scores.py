from pathlib import Path
import json,time
import argparse
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('previous_root',type=Path);args=p.parse_args()
r=args.root;old=args.previous_root;read=lambda p:json.loads(p.read_text())
def binding(d):return (d['binding']['image_sha256'],d['query']['question'],d['query']['reference'],d['raw_answer'],d['EOS'],d['at_cap'],tuple(d['raw_token_ids']))
labels={x['id']:x for x in read(old/'EVALUATION_LABELS_FROZEN.json')['items']};index={}
for key,paths in read(old/'BLIND_EVALUATION_MAPPING.json').items():
 for path in paths:
  d=read(old/path);b=binding(d)
  if b in index:assert index[b]['accepted']==labels[key]['accepted']
  index[b]=labels[key]
packet=read(r/'BLIND_EVALUATION.json');mapping=read(r/'BLIND_EVALUATION_MAPPING.json');inherited={};fresh=[]
for item in packet:
 bindings=[binding(read(r/path)) for path in mapping[item['id']]]
 if all(b in index for b in bindings):
  values=[index[b] for b in bindings];assert all(x['accepted']==values[0]['accepted'] and x['label']==values[0]['label'] for x in values)
  inherited[item['id']]={**values[0],'id':item['id'],'inherited_exact_binding':True}
 else:fresh.append(item)
(r/'INHERITED_LABELS_BEFORE_MAPPING.json').write_text(json.dumps(dict(inherited=inherited,fresh_ids=[x['id'] for x in fresh],time=time.time(),key_fields=['image','question','reference','full_response','EOS','at_cap','tokens']),indent=2))
(r/'FRESH_BLIND_REVIEW.json').write_text(json.dumps(fresh,ensure_ascii=False,indent=2));print('inherited',len(inherited),'fresh',len(fresh))
for row in fresh:print(json.dumps(row,ensure_ascii=False))
