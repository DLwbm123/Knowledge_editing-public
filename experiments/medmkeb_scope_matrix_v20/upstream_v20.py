"""Select only frozen members from existing upstream PMC archives; never unpack all."""
import csv,json,collections,re,zipfile,time,sys
from pathlib import Path
from prepare_v20 import read,write,norm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'medmkeb_independent_admission_v15'))
from inventory import article
csv.field_size_limit(sys.maxsize)
CLINICAL={'liver','kidney','left kidney','right kidney','spleen','pancreas','lung','lungs','right lung','left lung','left upper lobe','right upper lobe','left lower lobe','right lower lobe','heart','aorta','brain','cerebellum','frontal lobe','temporal lobe','parietal lobe','occipital lobe','thoracic vertebrae','lumbar vertebrae','scapula','femur','humerus','tibia','fibula','pelvis','rectum','colon','stomach','duodenum','esophagus','thyroid','adrenal gland','lymph node','pulmonary embolism','pleural effusion','pneumonia','consolidation','pneumothorax','hemorrhage','intracranial hemorrhage','subarachnoid hemorrhage','aortic dissection','hydronephrosis','cholelithiasis','appendicitis','pancreatitis','glioblastoma','meningioma'}
def freeze(data,prior,out,first):
    assert not (out/'CANDIDATES_FROZEN.json').exists();old=read(prior/'CANDIDATES_FROZEN.json');banned=set(old['historical_PMC'])|set(old['PMC_roles'])|{x['article'] for x in read(first/'CANDIDATES_FROZEN.json')['items']};pool=collections.OrderedDict();rows=0
    for name in ('train.csv','train_2.csv','test.csv','test_2.csv','test_clean.csv'):
        with (data/'sources/PMC-VQA'/name).open() as f:
            for i,row in enumerate(csv.DictReader(f)):
                rows+=1;image=row.get('Figure_path','');a=article(image);answer=row.get('Answer','').strip()
                if answer in 'ABCD' and len(answer)==1:answer=re.sub(r'^\s*[A-D]\s*[:.)]\s*','',row.get('Choice '+answer,'')).strip()
                key=norm(answer)
                if not a or a in banned or key not in CLINICAL:continue
                pool.setdefault(key,collections.OrderedDict()).setdefault(a,dict(source_csv=name,source_row=i,figure=image,question=row['Question'],reference=answer,original_choices={k:row.get(k) for k in ['Choice A','Choice B','Choice C','Choice D']}))
    concepts=[(k,list(v.items())) for k,v in pool.items() if len(v)>=6][:12];items=[];taken=set();stats=[]
    for ci,(target,source) in enumerate(concepts):
        c='U%02d'%(ci+1);stats.append(dict(concept=c,reference_key=target,available_sources=len(source)))
        for i,(a,row) in enumerate(source[:32]):
            if a in taken or len(items)>=330:continue
            taken.add(a);role='SOURCE_CANDIDATE' if i%4==0 else 'ROUTER_TRAIN' if i%4==1 else 'HELDOUT';items.append(dict(id=c+'_%02d'%(i+1),concept=c,reference_key=target,role=role,article=a,**row))
    out.mkdir(exist_ok=True,parents=True);j=dict(status='FROZEN_BEFORE_PIXELS',items=items,concepts=stats,metadata_rows=rows,historical_PMC_excluded=len(banned),representatives=len(items),pixel_reads=0,new_model_calls=0,time=time.time());write(out/'CANDIDATES_FROZEN.json',j);return {k:j[k] for k in ['status','representatives','metadata_rows','concepts']}
def visual(data,out):
    from PIL import Image,ImageDraw
    import io
    m=read(out/'CANDIDATES_FROZEN.json');wanted={Path(x['figure']).name for x in m['items']};archives=[data/'sources/PMC-VQA'/x for x in ('images.zip','images_2.zip')];indexes=[];handles=[];bytes_total=810196;images=out/'images';images.mkdir(exist_ok=True)
    for path in archives:
        z=zipfile.ZipFile(path);handles.append(z);idx=collections.defaultdict(list)
        for info in z.infolist():
            if Path(info.filename).name in wanted and not info.is_dir():idx[Path(info.filename).name].append(info)
        indexes.append(idx)
    results=[]
    try:
        for x in m['items']:
            key=Path(x['figure']).name;priority=(1,0) if '_2.' in x['source_csv'] else (0,1);found=next(((i,indexes[i][key][0]) for i in priority if len(indexes[i].get(key,[]))==1),None);entry={k:x[k] for k in ['id','concept','role','article']}
            try:
                assert found is not None,'No unique source member';i,info=found;assert 0<info.file_size<=64*1024**2 and bytes_total+info.file_size<=1024**3
                data_bytes=handles[i].read(info)
                with Image.open(io.BytesIO(data_bytes)) as im:im.load();dimensions=im.size
                target=images/(x['id']+Path(key).suffix.lower());assert not target.exists();target.write_bytes(data_bytes);bytes_total+=len(data_bytes);entry.update(status='READABLE',path=str(target.resolve()),bytes=len(data_bytes),dimensions=dimensions,archive=archives[i].name,member=info.filename)
            except Exception as e:entry.update(status='UNREADABLE',error=repr(e))
            results.append(entry)
    finally:
        for z in handles:z.close()
    write(out/'VISUAL_ITEMS.json',dict(items=results,total_V20_pixel_bytes=bytes_total))
    for start in range(0,len(results),24):
        canvas=Image.new('RGB',(1600,1800),'white');d=ImageDraw.Draw(canvas)
        for i,x in enumerate(results[start:start+24]):
            xx=i%4*400;yy=i//4*300;d.text((xx+5,yy+5),x['id'],fill='black')
            if x['status']=='READABLE':
                with Image.open(x['path']) as im:im=im.convert('RGB');im.thumbnail((390,265));canvas.paste(im,(xx+5,yy+30))
        canvas.save(out/('VISUAL_%02d.jpg'%(start//24)),quality=92)
    return dict(images=len(results),readable=sum(x['status']=='READABLE' for x in results),total_V20_bytes=bytes_total)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','visual']);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--prior',type=Path);p.add_argument('--first',type=Path);a=p.parse_args();print(freeze(a.data,a.prior,a.out,a.first) if a.mode=='freeze' else visual(a.data,a.out))
