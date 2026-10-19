"""Bounded fresh PMC scope candidates, frozen before pixels and model outputs."""
import json,re,csv,sys,collections,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'medmkeb_independent_admission_v15'))
from inventory import article,image_key
csv.field_size_limit(sys.maxsize)
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def norm(s):return ' '.join(str(s).casefold().strip().rstrip('.;:').split())
def clinical_candidate(answer):
    s=norm(answer)
    if not s or len(s.split())>8 or re.fullmatch(r'[\d\W]+',s):return False
    if re.fullmatch(r'(yes|no|unknown|normal|abnormal|none|not available|not applicable)',s):return False
    terms=r'(ct|computed tomography|mri|magnetic resonance imaging|x[- ]?ray|radiograph|ultrasound|us|pet|spect|angiography|axial|coronal|sagittal|frontal|transverse|2d|3d|two[- ]dimensional|three[- ]dimensional|black|white|blue|red|green|gray|grey|grayscale|black and white)'
    if re.fullmatch(r'(the |a |an )?'+terms+r'( scan| image| view| projection| plane| imaging| images| scans| reconstruction)?',s):return False
    return True

def freeze(data,prior,out):
    assert not (out/'CANDIDATES_FROZEN.json').exists();old=read(prior/'CANDIDATES_FROZEN.json');banned=set(old['historical_PMC'])|set(old['PMC_roles']);old_images=set(old['historical_image_keys'])|set(old['actual_locality_image_roles']);pool=collections.OrderedDict();total=0
    for name in ('train_data.json','eval_data_threehop_final.json'):
        for i,row in enumerate(read(data/'raw'/name)):
            total+=1;a=article(row['m_loc']);key=image_key(row['m_loc']);answer=norm(row['m_loc_a'])
            if not a or a in banned or key in old_images or not clinical_candidate(answer):continue
            pool.setdefault(answer,collections.OrderedDict()).setdefault(a,dict(source_raw=name,index=i,original=row))
    concepts=[(a,list(rows.items())) for a,rows in pool.items() if len(rows)>=6][:12];items=[];taken=set();stats=[]
    for ci,(answer,rows) in enumerate(concepts):
        stat=dict(concept='C%02d'%(ci+1),reference_key=answer,available_sources=len(rows),selected=0)
        for index,(a,c) in enumerate(rows[:32]):
            if a in taken:continue
            taken.add(a);role=('SOURCE_CANDIDATE' if index%4==0 else 'ROUTER_TRAIN' if index%4==1 else 'HELDOUT');v=c['original'];ident='C%02d_%02d'%(ci+1,index+1);items.append(dict(id=ident,concept=stat['concept'],reference_key=answer,role=role,article=a,source_image=str(data/v['m_loc']),original=v,source_raw=c['source_raw'],source_index=c['index']));stat['selected']+=1
        stats.append(stat)
    assert total==6987 and len(items)<=384;wanted={(Path(x['source_image']).name,norm(x['original']['m_loc_q'])):x for x in items}
    for x in items:x['upstream_matches']=[]
    for name in ('train.csv','train_2.csv','test.csv','test_2.csv','test_clean.csv'):
        with (data/'sources/PMC-VQA'/name).open() as f:
            for row in csv.DictReader(f):
                key=(Path(row.get('Figure_path','')).name,norm(row.get('Question','')))
                if key not in wanted:continue
                answer=row.get('Answer','').strip()
                if answer in 'ABCD' and len(answer)==1:answer=re.sub(r'^\s*[A-D]\s*[:.)]\s*','',row.get('Choice '+answer,'')).strip()
                if norm(answer)==norm(wanted[key]['original']['m_loc_a']):wanted[key]['upstream_matches'].append(dict(question=row['Question'],reference=answer,source_csv=name,figure=row['Figure_path']))
    result=dict(status='FROZEN_BEFORE_PIXELS',items=items,concepts=stats,raw_records=total,historical_PMC_excluded=len(banned),representatives=len(items),exact_source_QA=sum(bool(x['upstream_matches']) for x in items),patient_ID_hard_gate=False,new_model_calls=0,pixel_reads=0,time=time.time());out.mkdir(exist_ok=True,parents=True);write(out/'CANDIDATES_FROZEN.json',result);return {k:result[k] for k in ['status','raw_records','historical_PMC_excluded','representatives','exact_source_QA','concepts']}
def visual(out):
    from PIL import Image,ImageDraw
    m=read(out/'CANDIDATES_FROZEN.json');rows=[];size=0
    for x in m['items']:
        p=Path(x['source_image']);j={k:x[k] for k in ('id','concept','role','article')}
        try:
            n=p.stat().st_size;assert size+n<=1024**3
            with Image.open(p) as im:im.load();j.update(size=im.size,status='READABLE',bytes=n)
            size+=n
        except Exception as e:j.update(status='UNREADABLE',error=repr(e))
        rows.append(j)
    write(out/'VISUAL_ITEMS.json',dict(items=rows,total_bytes=size))
    for start in range(0,len(rows),24):
        canvas=Image.new('RGB',(1600,1800),'white');d=ImageDraw.Draw(canvas)
        for i,r in enumerate(rows[start:start+24]):
            xx=i%4*400;yy=i//4*300;d.text((xx+5,yy+5),r['id'],fill='black')
            if r['status']=='READABLE':
                with Image.open(m['items'][start+i]['source_image']) as im:im=im.convert('RGB');im.thumbnail((390,265));canvas.paste(im,(xx+5,yy+30))
        canvas.save(out/('VISUAL_%02d.jpg'%(start//24)),quality=92)
    return dict(images=len(rows),readable=sum(x['status']=='READABLE' for x in rows),bytes=size)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','visual']);p.add_argument('--data',type=Path);p.add_argument('--prior',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(freeze(a.data,a.prior,a.out) if a.mode=='freeze' else visual(a.out))
