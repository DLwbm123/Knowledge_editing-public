"""Freeze a finite, local-only candidate cohort before any Base outputs."""
import json,re,time,csv,collections
from pathlib import Path

def freeze(data,history,out):
    assert not (out/'CANDIDATES_FROZEN.json').exists()
    old=json.loads((history/'medmkeb_independent_admission_v15_20261019/CANDIDATES_FROZEN.json').read_text())
    manifests=[Path(x) for x in old['history_sources']]
    manifests+=list(history.glob('*/DATA_ADMISSION.json'))+list(history.glob('*/stage/DATA_ADMISSION.json'))+list(history.glob('*/completed/DATA_ADMISSION.json'))
    manifests+=[history/'mechanism_v51_20261022/BOUNDARY_INVENTORY.json']
    used=set()
    for p in sorted(set(manifests)):
        if not p.exists():continue
        payload=json.loads(p.read_text())
        if 'files_with_image_references' in payload:
            text=json.dumps([f['image_keys'] for f in payload['files_with_image_references'] if 'EXCLUSION_IDS' not in f['path']])
        else:text=json.dumps(payload)
        used.update(re.findall(r'PMC\d+',text,re.I))
    used={x.upper() for x in used}; seen=set();candidates=[];counts=collections.Counter();all_rows=[]
    for split,name in [('train','train_data.json'),('eval','eval_data_threehop_final.json')]:
        source=json.loads((data/'raw'/name).read_text())
        for order,v in enumerate(source):
            image=data/v['m_loc'];m=re.search(r'PMC\d+',str(image),re.I);article=m.group().upper() if m else None
            if not article:counts['no_article']+=1;continue
            exposure='PREVIOUSLY_RECORDED_SOURCE' if article in used else 'NO_MATCH_IN_LOCAL_HISTORY'
            all_rows.append(dict(split=split,index=order,id=v['id'],source_group=article,exposure=exposure,image=str(image)))
            if article in used:counts['historical_source']+=1;continue
            if article in seen:counts['duplicate_source']+=1;continue
            if not image.is_file():counts['missing_local_image']+=1;continue
            if len(candidates)>=64:continue
            seen.add(article);candidates.append(dict(query_id=f'L{len(candidates):02d}',original_split=split,source_index=order,source_id=v['id'],source_group=article,image_path=str(image),question=v['m_loc_q'],reference=v['m_loc_a'],role='RETENTION_CANDIDATE_EVAL_ONLY',patient_independence='UNKNOWN',exposure=exposure))
    wanted={Path(x['image_path']).name for x in candidates};captions={k:[] for k in wanted}
    for name in ['train.csv','train_2.csv','test.csv','test_2.csv']:
        p=data/'sources/PMC-VQA'/name
        with p.open(newline='',encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                key=row.get('Figure_path')
                if key in wanted:captions[key].append(dict(source_file=name,**row))
    for x in candidates:x['local_caption_rows']=captions[Path(x['image_path']).name]
    assert len({x['source_group'] for x in candidates})==len(candidates) and not ({x['source_group'] for x in candidates}&used)
    record=dict(time=time.time(),selection='First64 unique unrecorded PMC locality sources in fixed train then eval metadata order; no answer/performance filter; no replacement after review',local_only=True,network_fetches=0,model_calls=0,patient_independence='UNKNOWN',not_independent_confirmation=True,history_manifest_count=len(set(manifests)),historical_source_count=len(used),total_metadata_records=len(all_rows),counts=dict(counts),candidates=candidates)
    out.mkdir(exist_ok=True,parents=True)
    for name,x in [('CANDIDATES_FROZEN.json',record),('HISTORY_INVENTORY.json',all_rows)]: (out/name).write_text(json.dumps(x,indent=2)+'\n')
    return dict(candidates=len(candidates),local_caption_bound=sum(bool(x['local_caption_rows']) for x in candidates),historical_sources=len(used),metadata_records=len(all_rows))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path);p.add_argument('--history',type=Path);p.add_argument('--out',type=Path);a=p.parse_args();print(json.dumps(freeze(a.data,a.history,a.out)))
