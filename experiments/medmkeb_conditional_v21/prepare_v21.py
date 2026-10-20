"""Bounded new PMC groups; preserve all exposed V20 candidates in exclusions."""
import csv,json,collections,re,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'medmkeb_scope_matrix_v20'))
from upstream_v20 import visual as old_visual
from prepare_v20 import norm,read,write
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'medmkeb_independent_admission_v15'))
from inventory import article
csv.field_size_limit(sys.maxsize)
TARGETS=('kidney','left lung','right upper lobe','duodenum')
def freeze(data,repo,out):
 assert not (out/'CANDIDATES_FROZEN.json').exists()
 old=read(repo/'.private/medmkeb_expanded_v18_20261019/CANDIDATES_FROZEN.json');banned=set(old['historical_PMC'])|set(old['PMC_roles'])
 for sub in ('','upstream1'):
  c=read(repo/'.private/medmkeb_scope_matrix_v20_20261019'/sub/'CANDIDATES_FROZEN.json');banned.update(x['article'] for x in c['items'])
 pool={t:collections.OrderedDict() for t in TARGETS};n=0
 for name in ('train.csv','train_2.csv','test.csv','test_2.csv','test_clean.csv'):
  with (data/'sources/PMC-VQA'/name).open() as f:
   for i,row in enumerate(csv.DictReader(f)):
    n+=1;a=article(row.get('Figure_path',''));answer=row.get('Answer','').strip()
    if answer in 'ABCD' and len(answer)==1:answer=re.sub(r'^\s*[A-D]\s*[:.)]\s*','',row.get('Choice '+answer,'')).strip()
    key=norm(answer)
    if not a or a in banned or key not in pool:continue
    pool[key].setdefault(a,dict(source_csv=name,source_row=i,figure=row['Figure_path'],question=row['Question'],reference=answer,original_choices={k:row.get(k) for k in ['Choice A','Choice B','Choice C','Choice D']}))
 items=[];taken=set()
 for ci,t in enumerate(TARGETS,1):
  for i,(a,row) in enumerate(list(pool[t].items())[:32]):
   if a in taken:continue
   taken.add(a);role=('WRITE_TRAIN','ROUTER_DEV','CONFIRM','CONFIRM')[i%4];items.append(dict(id='N%02d_%02d'%(ci,i+1),concept='N%02d'%ci,reference_key=t,role=role,article=a,**row))
 out.mkdir(parents=True,exist_ok=True);d=dict(status='FROZEN_BEFORE_PIXELS',items=items,metadata_rows=n,historical_PMC_excluded=len(banned),available={t:len(pool[t]) for t in TARGETS},new_models=0,time=time.time());write(out/'CANDIDATES_FROZEN.json',d);print({k:v for k,v in d.items() if k!='items'})
def visual(data,out):
 # Existing bounded member reader reused with per-V21 zero-based pixel accounting.
 import inspect
 source=inspect.getsource(old_visual).replace('def visual(data,out):','def new_visual(data,out):').replace('bytes_total=810196','bytes_total=0').replace('<=1024**3','<=256*1024**2').replace('total_V20_pixel_bytes','total_V21_pixel_bytes').replace('total_V20_bytes','total_V21_bytes')
 import upstream_v20
 namespace=dict(upstream_v20.__dict__);exec(source,namespace);return namespace['new_visual'](data,out)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--data',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(freeze(a.data,a.repo,a.out) if a.mode=='freeze' else visual(a.data,a.out))
