"""Freeze remaining actual-image components and independent PMC-source roles."""
import collections,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'medmkeb_independent_admission_v15'))
from inventory import Groups,image_key,article
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n')

def history_exposure(history):
    old=read(history/'medmkeb_independent_admission_v15_20261019/CANDIDATES_FROZEN.json');images=set();articles=set()
    def expose(value):
        images.add(image_key(value));a=article(value)
        if a:articles.add(a)
    for path in old['history_sources']:
        h=read(path)
        if 'files_with_image_references' in h:
            assert not h['errors']
            for f in h['files_with_image_references']:
                if 'EXCLUSION_IDS' not in f['path']:
                    for k in f['image_keys']:expose(k)
        else:
            for c in h['candidates']:
                for k in ('image','image_rephrase','m_loc'):
                    if k in c.get('original',{}):expose(c['original'][k])
                for k in c.get('component_paths',[]):expose(k)
    for c in old['candidates']:
        for k in ('image','image_rephrase','m_loc'):expose(c['original'][k])
    # V17 is exactly V16; exclude entire selected member lists, not just representatives.
    for c in read(history/'medmkeb_provenance_v16_20261019/CANDIDATES_FROZEN.json')['candidates']:
        for row in c['members']:
            for k in ('image','image_rephrase','m_loc'):expose(row['original'][k])
    return images,articles

def freeze(data,private,history):
    assert not (private/'CANDIDATES_FROZEN.json').exists();start=time.time();rows=[]
    for split,name in [('official_train','train_data.json'),('official_eval','eval_data_threehop_final.json')]:
        path=data/'raw'/name;assert path.stat().st_size<64*1024**2
        rows.extend(dict(split=split,index=i,original=v) for i,v in enumerate(read(path)))
    assert len(rows)==6987;images,articles=history_exposure(history);g=Groups()
    for c in rows:g.union(image_key(c['original']['image']),image_key(c['original']['image_rephrase']))
    banned={g.root(k) for k in images if k in g.parent};families=collections.defaultdict(list)
    for c in rows:families[g.root(image_key(c['original']['image']))].append(c)
    candidates=[];roles={};selected_images=set()
    for group,family in families.items():
        if group in banned:continue
        c=next((c for c in family if image_key(c['original']['image'])!=image_key(c['original']['image_rephrase'])),None)
        if c is None:continue
        assert len(candidates)<43;roles[group]='EDIT_EVAL'
        candidates.append(dict(role='EDIT_EVAL',group=group,representative=c,members=family))
        for k in ('image','image_rephrase'):selected_images.add(image_key(c['original'][k]))
    pmc=collections.OrderedDict()
    for c in rows:
        v=c['original'];a=article(v['m_loc']);key=image_key(v['m_loc'])
        if not a or a in articles or key in images or key in selected_images:continue
        pmc.setdefault(a,[]).append(c)
    pmc_roles={};loc_roles={};counts=collections.Counter()
    for a,family in list(pmc.items())[:512]:
        target='RETENTION_TRAIN' if len(pmc_roles)%2==0 else 'RETENTION_EVAL';pmc_roles[a]=target;counts[target]+=1
        c=family[0];key=image_key(c['original']['m_loc']);assert key not in loc_roles;loc_roles[key]=target
        candidates.append(dict(role=target,article=a,representative=c,members=family))
    assert all(g.root(image_key(c['representative']['original']['image'])) not in banned for c in candidates if c['role']=='EDIT_EVAL')
    assert not set(pmc_roles)&articles and not set(loc_roles)&images
    summary=dict(status='COHORT_FROZEN_BEFORE_PIXELS',raw_records=len(rows),native_components=len(families),historically_excluded_components=len(banned),edit_candidates=counts.get('EDIT_EVAL',sum(c['role']=='EDIT_EVAL' for c in candidates)),retention_train=counts['RETENTION_TRAIN'],retention_eval=counts['RETENTION_EVAL'],fresh_PMC_available=len(pmc),representative_images=sum(2 if c['role']=='EDIT_EVAL' else 1 for c in candidates),pixel_reads=0,model_calls=0,patient_ID_hard_gate=False,elapsed_seconds=time.time()-start)
    assert summary['representative_images']<=598
    private.mkdir(exist_ok=True,parents=True)
    write(private/'CANDIDATES_FROZEN.json',dict(summary=summary,candidates=candidates,native_component_roles=roles,PMC_roles=pmc_roles,actual_locality_image_roles=loc_roles,historical_image_keys=sorted(images),historical_PMC=sorted(articles),excluded_native_components=sorted(banned)))
    write(private/'STATE.json',dict(**summary,protocol='reports/medmkeb_expanded_v18_20261019/PREREG_ZH.md',GPU_workers=0))
    return summary

def visual(data,private):
    from PIL import Image,ImageDraw
    import shutil
    assert not (private/'VISUAL_ITEMS.json').exists();m=read(private/'CANDIDATES_FROZEN.json');items=[];counts=collections.Counter();total=0;(private/'images').mkdir(exist_ok=True)
    for c in m['candidates']:
        role=c['role'];counts[role]+=1;v=c['representative']['original'];fields=('image','image_rephrase') if role=='EDIT_EVAL' else ('m_loc',)
        for field in fields:
            ident=('E%03d%s'%(counts[role],'N' if field=='image' else 'C')) if role=='EDIT_EVAL' else ('T' if role=='RETENTION_TRAIN' else 'V')+'%03d'%counts[role]
            src=data/v[field];dst=private/'images'/(ident+src.suffix.lower());x=dict(id=ident,role=role,index=counts[role],field=field,source_path=str(src),path=str(dst.resolve()))
            try:
                size=src.stat().st_size;assert total+size<=1024**3;shutil.copyfile(src,dst)
                with Image.open(dst) as im:im.load();dims=im.size
                total+=size;x.update(status='READABLE',bytes=size,dimensions=dims)
            except Exception as e:x.update(status='UNREADABLE',error=repr(e))
            items.append(x)
    assert len(items)==m['summary']['representative_images'];write(private/'VISUAL_ITEMS.json',dict(items=items,total_bytes=total))
    for start in range(0,len(items),20):
        canvas=Image.new('RGB',(1600,1800),'white');draw=ImageDraw.Draw(canvas)
        for i,x in enumerate(items[start:start+20]):
            xx=i%4*400;yy=i//4*360;draw.text((xx+5,yy+5),x['id'],fill='black')
            if x['status']=='READABLE':
                with Image.open(x['path']) as im:im=im.convert('RGB');im.thumbnail((390,325));canvas.paste(im,(xx+5,yy+30))
        canvas.save(private/('VISUAL_%02d.jpg'%(start//20)),quality=92)
    return dict(images=len(items),readable=sum(x['status']=='READABLE' for x in items),bytes=total)

def selfcheck():
    g=Groups();g.union('a','b');g.union('b','c');assert g.root('a')==g.root('c')
    assert article('PMC123.jpg')=='PMC123' and image_key('part1_row_1_image.png')==image_key('prefix_part1_row_1_image.jpg')
    # Co-occurrence is not an image edge: independent PMC and GMAI nodes remain different.
    assert g.root('PMC123.jpg')!=g.root('a')
    return dict(status='PASS',complete_native_components=True,PMC_roles_separate_from_arbitrary_pairing=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','visual','selfcheck']);p.add_argument('--data',type=Path);p.add_argument('--private',type=Path);p.add_argument('--history',type=Path);a=p.parse_args()
    print(selfcheck() if a.mode=='selfcheck' else freeze(a.data,a.private,a.history) if a.mode=='freeze' else visual(a.data,a.private))
