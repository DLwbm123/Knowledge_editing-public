#!/usr/bin/env python3
"""Recompute the public Stage24E-R2 decomposition from deidentified item scores."""
import collections,json,sys

def rate(xs): return {'correct':sum(xs),'total':len(xs),'rate':sum(xs)/len(xs) if xs else None}
def main(path):
 rows=[json.loads(x) for x in open(path) if x.strip()]
 out={}
 for arm in ('B0','B1','B2'):
  p19=[r for r in rows if r['arm']==arm and r['prefix']==19]
  roles={}
  for role in ('native','H_eval','U_eval','native_text_extension','positive_image'):
   rr=[r for r in p19 if r['role']==role]; base=[r['base_correct'] for r in rr if r['base_correct'] is not None]
   roles[role]={'overall':rate([r['current_correct'] for r in rr]),'base_correct':rate(base),'keep':{'correct':sum(r['current_correct'] and r['base_correct'] is True for r in rr),'total':len(rr),'rate':sum(r['current_correct'] and r['base_correct'] is True for r in rr)/len(rr) if rr else None},'repair':{'correct':sum(r['current_correct'] and r['base_correct'] is False for r in rr),'total':len(rr),'rate':sum(r['current_correct'] and r['base_correct'] is False for r in rr)/len(rr) if rr else None}}
  out[arm]=roles
 print(json.dumps({'rows':len(rows),'arms':out},ensure_ascii=False,indent=2))
if __name__=='__main__': main(sys.argv[1])
