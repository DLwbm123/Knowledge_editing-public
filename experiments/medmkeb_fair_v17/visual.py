"""Only the80 previously frozen representatives; labels without question/answer hints."""
import json,shutil,time
from pathlib import Path
from PIL import Image,ImageDraw

def prepare(data,private):
    m=json.loads((private/'CANDIDATES_FROZEN.json').read_text());(private/'images').mkdir(exist_ok=True);assert not (private/'VISUAL_ITEMS.json').exists();items=[];n={'EDIT_EVAL':0,'RETENTION_TRAIN':0,'RETENTION_EVAL':0};total=0
    for c in m['candidates']:
        role=c['role'];n[role]+=1;v=c['representative']['original'];fields=('image','image_rephrase') if role=='EDIT_EVAL' else ('m_loc',)
        for field in fields:
            label=('E'+str(n[role]).zfill(2)+('N' if field=='image' else 'C')) if role=='EDIT_EVAL' else ('T' if role=='RETENTION_TRAIN' else 'V')+str(n[role]).zfill(2)
            src=data/v[field];record=dict(id=label,role=role,index=n[role],field=field,source_path=str(src))
            try:
                size=src.stat().st_size;assert total+size<=512*1024**2;dest=private/'images'/(label+src.suffix.lower());shutil.copyfile(src,dest)
                with Image.open(dest) as im:im.load();dims=im.size
                total+=size;record.update(path=str(dest.resolve()),bytes=size,dimensions=dims,status='READABLE')
            except Exception as e:record.update(status='UNREADABLE',error=repr(e))
            items.append(record)
    assert len(items)==80
    (private/'VISUAL_ITEMS.json').write_text(json.dumps(dict(items=items,total_bytes=total,unique_images=80,time=time.time()),indent=2)+'\n')
    for start in range(0,80,20):
        canvas=Image.new('RGB',(1600,1800),'white');draw=ImageDraw.Draw(canvas)
        for i,item in enumerate(items[start:start+20]):
            x=(i%4)*400;y=(i//4)*360;draw.text((x+5,y+5),item['id'],fill='black')
            if item['status']=='READABLE':
                with Image.open(item['path']) as im:im=im.convert('RGB');im.thumbnail((390,325));canvas.paste(im,(x+5,y+30))
            else:draw.text((x+5,y+50),'UNREADABLE',fill='red')
        canvas.save(private/('VISUAL_'+str(start//20)+'.jpg'),quality=92)
    return dict(images=len(items),bytes=total,readable=sum(x['status']=='READABLE' for x in items))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--private',type=Path,required=True);a=p.parse_args();print(prepare(a.data,a.private))
