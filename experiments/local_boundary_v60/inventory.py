"""Freeze local caption candidates and source-disjoint roles before model screening."""
import argparse,csv,json,re,time,zipfile,collections
from pathlib import Path

def freeze(data,history,out):
    assert not (out/'CANDIDATES_FROZEN.json').exists()
    old=json.loads((history/'medmkeb_independent_admission_v15_20261019/CANDIDATES_FROZEN.json').read_text())
    manifests=[Path(x) for x in old['history_sources']]
    for pattern in ['*/DATA_ADMISSION.json','*/stage/DATA_ADMISSION.json','*/completed/DATA_ADMISSION.json']:
        manifests+=list(history.glob(pattern))
    manifests += [history/'mechanism_v51_20261022/BOUNDARY_INVENTORY.json',history/'local_pool_v58_20261022/CANDIDATES_FROZEN.json']
    used={'PMC9414113'}
    for p in sorted(set(manifests)):
        if not p.exists():continue
        obj=json.loads(p.read_text())
        if 'files_with_image_references' in obj:
            obj=[f['image_keys'] for f in obj['files_with_image_references'] if 'EXCLUSION_IDS' not in f['path']]
        used.update(x.upper() for x in re.findall(r'PMC\d+',json.dumps(obj),re.I))
    local={p.name:str(p) for p in (data/'images').rglob('*') if p.is_file()}
    archives={}
    for name in ['images.zip','images_2.zip']:
        p=data/'sources/PMC-VQA'/name
        if not p.is_file():continue
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if not n.endswith('/'):archives.setdefault(Path(n).name,dict(archive=str(p),member=n))
    candidates=[];excluded=[];counts=collections.Counter();seen=set()
    for file in ['train.csv','train_2.csv','test.csv','test_2.csv']:
        with (data/'sources/PMC-VQA'/file).open(newline='',encoding='utf-8-sig') as f:
            for idx,r in enumerate(csv.DictReader(f)):
                counts['all_csv_records']+=1;caption=r.get('Caption','');name=r['Figure_path']
                if not re.search(r'\bCT\b|computed\s+tomograph',caption,re.I):continue
                if not re.search(r'lymph\s*node|lymphadenopath|esophag|oesophag',caption,re.I):continue
                counts['lexical_records']+=1
                match=re.search(r'PMC\d+',name,re.I)
                if not match:counts['no_source']+=1;continue
                source=match.group().upper()
                key=(name,r['Question'])
                if key in seen:counts['duplicate_record']+=1;continue
                seen.add(key)
                if source in used:
                    excluded.append(dict(source_group=source,image=name,reason='PREVIOUSLY_RECORDED_SOURCE'));continue
                loc={'path':local[name]} if name in local else archives.get(name)
                if not loc:counts['no_local_image']+=1;continue
                ans=r['Answer'].strip();reference=r.get('Choice '+ans,ans) if ans in 'ABCD' and len(ans)==1 else ans
                candidates.append(dict(source_group=source,image_name=name,local_image=loc,csv=file,csv_index=idx,question=r['Question'].strip(),reference=reference.strip(),caption=caption,choices={k:v for k,v in r.items() if k.startswith('Choice ')}))
    sources=sorted({r['source_group'] for r in candidates},key=lambda x:int(x[3:]))
    roles={s:'TRAIN_SUPPORT' if i%2==0 else 'EVAL' for i,s in enumerate(sources)}
    candidates.sort(key=lambda r:(int(r['source_group'][3:]),r['image_name'],r['csv'],r['csv_index']))
    for i,r in enumerate(candidates):r.update(candidate_id=f'N{i:03d}',role=roles[r['source_group']])
    assert not ({r['source_group'] for r in candidates}&used)
    assert not ({r['source_group'] for r in candidates if r['role']=='TRAIN_SUPPORT'}&{r['source_group'] for r in candidates if r['role']=='EVAL'})
    obj=dict(time=time.time(),local_only=True,model_calls=0,counts=dict(counts),history_source_count=len(used),source_roles=roles,candidates=candidates,excluded=excluded)
    out.mkdir(parents=True,exist_ok=True);(out/'CANDIDATES_FROZEN.json').write_text(json.dumps(obj,indent=2)+'\n')
    return dict(counts=counts,candidates=len(candidates),sources=len(sources),unique_images=len({r['image_name'] for r in candidates}),historical_excluded=len(excluded))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--history',type=Path,required=True);a.add_argument('--out',type=Path,required=True);v=a.parse_args();print(json.dumps(freeze(v.data,v.history,v.out)))
