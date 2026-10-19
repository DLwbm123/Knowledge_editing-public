"""Bounded metadata-only cohort freeze; no image pixels or model imports."""
import json,re,time,collections
from pathlib import Path

def image_key(value):
    value=str(value);m=re.search(r'part\d+_row_\d+_image',value,re.I)
    return m.group(0).lower() if m else Path(value).name.lower()

def article(value):
    m=re.search(r'PMC\d+',str(value),re.I)
    return m.group(0).upper() if m else None

class Groups:
    def __init__(self):self.parent={}
    def root(self,k):
        self.parent.setdefault(k,k)
        if self.parent[k]!=k:self.parent[k]=self.root(self.parent[k])
        return self.parent[k]
    def union(self,a,b):
        a,b=self.root(a),self.root(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)

def selfcheck():
    assert image_key('images/x/part2_row_12_image.png')==image_key('0001_image_part2_row_12_image.jpg')
    assert article('pMc123_Figure.png')=='PMC123' and article('part2_row_12_image.png') is None
    g=Groups();g.union('a','b');g.union('b','c');assert g.root('a')==g.root('c') and g.root('d')!=g.root('a')
    return dict(status='PASS',image_alias_normalization=True,transitive_group=True,PMC_not_patient=True,models=0,pixels=0)

def run(data,private,history):
    start=time.time();assert not (private/'CANDIDATES_FROZEN.json').exists();private.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text());paths=[data/'raw/train_data.json',data/'raw/eval_data_threehop_final.json'];assert all(p.stat().st_size<=64*1024**2 for p in paths)
    records={s:read(p) for s,p in zip(('train','eval'),paths)};assert sum(map(len,records.values()))<=100000
    assert all(isinstance(v,list) for v in records.values())
    seen_images=set();seen_articles=set();history_sources=[];index_only=[]
    def expose(value):
        if isinstance(value,str):
            seen_images.add(image_key(value));a=article(value)
            if a:seen_articles.add(a)
    old=history/'medmkeb_semantic6_v1_20261018'
    for name in ('HISTORY_MY_GPU.json','HISTORY_PRO5000.json'):
        path=old/name;h=read(path);assert not h['errors'];history_sources.append(str(path))
        for f in h['files_with_image_references']:
            if 'EXCLUSION_IDS' in f['path']:index_only.append(f['path']);continue
            for k in f['image_keys']:expose(k)
    manifests=[old/'CANDIDATES_FROZEN.json']
    for version in (2,3,4):
        d=history/f'medmkeb_locality6_v{version}_20261018'
        manifests.extend(d/n for n in ('CANDIDATES_FROZEN.json','CANDIDATES_BEFORE_VISUAL_REVIEW.json'))
    for folder in ('medmkeb_locality6_v2_20261018/schema_repair0','medmkeb_locality6_v3_20261018/path_repair0','medmkeb_locality6_v4_20261018/prebase_scope_amendment0'):
        d=history/folder;manifests.extend(p for p in (d/'CANDIDATES_FROZEN.json',d/'CANDIDATES_BEFORE_VISUAL_REVIEW.json') if p.exists())
    old_candidates=0
    for path in manifests:
        h=read(path);history_sources.append(str(path))
        for c in h['candidates']:
            old_candidates+=1;v=c.get('original',{})
            for k in ('image','image_rephrase','m_loc'):
                if k in v:expose(v[k])
            for k in c.get('component_paths',[]):expose(k)
            for row in c.get('rows',[])+([c['row']] if 'row' in c else []):
                for k in ('image_path','source_group'):
                    if k in row:expose(row[k])
    groups=Groups();split_articles={};split_groups={};schema=collections.Counter();field_counts=collections.Counter()
    for split,items in records.items():
        split_articles[split]=set()
        for v in items:
            assert all(k in v for k in ('image','image_rephrase','m_loc','m_loc_q','m_loc_a','id'))
            groups.union(image_key(v['image']),image_key(v['image_rephrase']))
            a=article(v['m_loc'])
            if a:split_articles[split].add(a)
            for k,value in v.items():
                schema[k]+=1
                if re.search(r'patient|subject|study|case',k,re.I) and value not in (None,'',[],{}):field_counts[k]+=1
    for split,items in records.items():split_groups[split]={groups.root(image_key(v['image'])) for v in items}
    shared_groups=split_groups['train']&split_groups['eval'];shared_articles=split_articles['train']&split_articles['eval'];exposed_groups={groups.root(k) for k in seen_images if k in groups.parent}
    candidates=[];exclusions=[];counts=collections.Counter();chosen_images=set();chosen_groups=set();chosen_articles=set()
    def reasons(v,split,role):
        keys=[image_key(v[k]) for k in ('image','image_rephrase','m_loc')];group=groups.root(keys[0]);a=article(v['m_loc']);why=[]
        if any(k in seen_images for k in keys):why.append('HISTORICAL_IMAGE')
        if group in exposed_groups:why.append('HISTORICAL_NATIVE_CROSS_COMPONENT')
        if a in seen_articles:why.append('HISTORICAL_PMC')
        if not a:why.append('NO_LOCALITY_PMC_BINDING')
        elif a in shared_articles:why.append('CROSS_SPLIT_PMC')
        if a in chosen_articles or group in chosen_groups or any(k in chosen_images for k in keys):why.append('NEW_COHORT_ROLE_OR_GROUP_OVERLAP')
        if role=='EDIT':
            if group in shared_groups:why.append('CROSS_SPLIT_EDIT_COMPONENT')
            if keys[0]==keys[1]:why.append('SAME_IMAGE_PAIR')
        return why,group,a,keys
    def choose(split,role,cap):
        selected=0
        for index,v in enumerate(records[split]):
            if role=='EDIT' and v.get('modality')!='CT':continue
            why,group,a,keys=reasons(v,split,role)
            if why:
                exclusions.append(dict(split=split,role=role,index=index,id=v['id'],reasons=why));counts.update(why);continue
            candidates.append(dict(role=role,split=split,index=index,original=v,article=a,edit_component=group,patient_independence='UNKNOWN',actual_image_modality='NOT_REVIEWED',source_QA_admission='NOT_REVIEWED'));chosen_articles.add(a);chosen_groups.add(group);chosen_images.update(keys);selected+=1
            if selected==cap:break
        return selected
    selected={}
    for split in ('train','eval'):selected['RETENTION_'+split]=choose(split,'RETENTION',32)
    selected['EDIT_eval']=choose('eval','EDIT',8)
    # Dataset record ids and PMC article ids are deliberately not promoted to patient ids.
    patient_fields={k:v for k,v in field_counts.items() if re.search(r'patient|subject',k,re.I)}
    summary=dict(status='METADATA_COHORT_FROZEN_PATIENT_INDEPENDENCE_UNRESOLVED',source_record_counts={k:len(v) for k,v in records.items()},historical_manifest_count=len(manifests),historical_candidate_entries_including_archive_duplicates=old_candidates,historical_image_keys=len(seen_images),historical_PMC_count=len(seen_articles),historical_edit_components=len(exposed_groups),cross_split_edit_components=len(shared_groups),cross_split_PMC_count=len(shared_articles),selected=selected,exclusion_reason_counts=dict(counts),patient_related_field_counts=patient_fields,other_case_study_field_counts=dict(field_counts),patient_independence='UNKNOWN',strict_patient_independent_admission=False,actual_modality_admission=False,source_QA_admission=False,Base_admission=False,model_calls=0,pixel_reads=0,GPU_hours=0,elapsed_seconds=time.time()-start,source_schema_fields=sorted(schema),not_a_clinical_or_protection_result=True)
    manifest=dict(summary=summary,source_files=[dict(path=str(p),bytes=p.stat().st_size,mtime=p.stat().st_mtime) for p in paths],history_sources=history_sources,index_only_files_not_actual_exposure=index_only,candidates=candidates,excluded=exclusions)
    for n,v in [('CANDIDATES_FROZEN.json',manifest),('RESULTS_PRIVATE.json',summary),('CPU_ADMISSION.json',selfcheck())]:
        text=json.dumps(v,indent=2);assert len(text.encode())<=128*1024**2;assert time.time()-start<2*3600;(private/n).write_text(text+'\n')
    return summary

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--history',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.data,a.private,a.history),indent=2))
