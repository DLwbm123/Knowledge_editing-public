"""Existing upstream schemas and deterministic whole-component repartition, no models."""
import csv,json,re,time,collections,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'medmkeb_independent_admission_v15'))
from inventory import Groups,image_key,article
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n')

def identity_keys(keys):return [k for k in keys if any(x in k.lower() for x in ('patient','subject','study','case'))]

def header(path,delimiter):
    with path.open('rb') as f:head=f.readline(65536)
    assert head.endswith(b'\n') or len(head)<65536
    columns=next(csv.reader([head.decode('utf-8-sig').rstrip('\r\n')],delimiter=delimiter))
    return dict(header_bytes=len(head),columns=columns,patient_or_study_columns=identity_keys(columns))

def metadata(path):
    assert path.stat().st_size<=64*1024**2;value=read(path);keys=collections.Counter();nodes=0
    def walk(v):
        nonlocal nodes
        nodes+=1;assert nodes<=2000000
        if isinstance(v,dict):
            for k,x in v.items():keys[k]+=1;walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(value)
    return dict(bytes=path.stat().st_size,root_type=type(value).__name__,root_keys=list(value) if isinstance(value,dict) else None,keys=dict(keys),patient_or_study_keys={k:keys[k] for k in identity_keys(keys)})

def upstream(data,private):
    start=time.time();rows=[]
    for i in range(1,12):
        p=data/'sources/GMAI-MMBench'/f'GMAI_mm_bench_TEST_part_{i}.tsv';rows.append(dict(kind='GMAI_TSV_HEADER',path=str(p),**header(p,'\t')))
    for name in ('train.csv','train_2.csv','test.csv','test_2.csv','test_clean.csv'):
        p=data/'sources/PMC-VQA'/name;rows.append(dict(kind='PMC_CSV_HEADER',path=str(p),**header(p,',')))
    paths=[data/'reports'/f'{n}.json' for n in ('repo_versions','json_schema_report','manifest_report','pmc_vqa_locality_extract_report','gmai_mmbench_extract_report','omnivqa_source_report','download_report')]+[data/'processed/medmkeb_vlkeb_manifest.json'];total=0
    for p in paths:
        total+=p.stat().st_size;assert total<=128*1024**2;rows.append(dict(kind='SOURCE_MAPPING_METADATA',path=str(p),**metadata(p)))
    assert len(rows)==24 and time.time()-start<7200
    result=dict(status='SOURCE_SCHEMA_AUDIT_COMPLETE',files=rows,source_mapping_JSON_bytes=total,elapsed_seconds=time.time()-start,pixel_reads=0,model_calls=0,header_patient_fields_present=any(x.get('patient_or_study_columns') for x in rows),mapping_patient_fields_present=any(x.get('patient_or_study_keys') for x in rows),patient_mapping_verified=False)
    write(private/'UPSTREAM_AUDIT_PRIVATE.json',result);return result

def repartition(data,private,history):
    start=time.time();assert not (private/'CANDIDATES_FROZEN.json').exists();paths=[data/'raw'/n for n in ('train_data.json','eval_data_threehop_final.json')];assert all(p.stat().st_size<=64*1024**2 for p in paths)
    records=[]
    for split,p in zip(('official_train','official_eval'),paths):
        for index,v in enumerate(read(p)):records.append(dict(split=split,index=index,original=v))
    assert len(records)<=100000
    v15=read(history/'medmkeb_independent_admission_v15_20261019/CANDIDATES_FROZEN.json');images=set();articles=set()
    def expose(s):
        images.add(image_key(s));a=article(s)
        if a:articles.add(a)
    for path in v15['history_sources']:
        h=read(path)
        if 'files_with_image_references' in h:
            assert not h['errors']
            for f in h['files_with_image_references']:
                if 'EXCLUSION_IDS' in f['path']:continue
                for k in f['image_keys']:expose(k)
        else:
            for c in h['candidates']:
                for k in ('image','image_rephrase','m_loc'):
                    if k in c.get('original',{}):expose(c['original'][k])
                for k in c.get('component_paths',[]):expose(k)
    for c in v15['candidates']:
        for k in ('image','image_rephrase','m_loc'):expose(c['original'][k])
    g=Groups()
    for c in records:g.union(image_key(c['original']['image']),image_key(c['original']['image_rephrase']))
    families=collections.defaultdict(list)
    for c in records:families[g.root(image_key(c['original']['image']))].append(c)
    exposed={g.root(k) for k in images if k in g.parent};role={};selected=[];excluded=[];used_PMC=set()
    for group,family in families.items():
        if group in exposed:
            excluded.append(dict(group=group,reason='HISTORICAL_OR_V15_EDIT_COMPONENT'));continue
        c=next((c for c in family if c['original'].get('modality')=='CT' and image_key(c['original']['image'])!=image_key(c['original']['image_rephrase'])),None)
        if c is None:continue
        # Retain the conservative original-record source exclusion from the frozen plan.
        v=c['original'];a=article(v['m_loc'])
        if a in articles or image_key(v['m_loc']) in images:
            excluded.append(dict(group=group,reason='REPRESENTATIVE_HISTORICAL_LOCALITY_SOURCE'));continue
        role[group]='EDIT_EVAL';selected.append(dict(role='EDIT_EVAL',group=group,representative=c,members=family,patient_independence='UNKNOWN',actual_modality='NOT_REVIEWED'))
        if a:used_PMC.add(a)
        if sum(x['role']=='EDIT_EVAL' for x in selected)==8:break
    by_PMC=collections.defaultdict(list)
    for c in records:
        v=c['original'];a=article(v['m_loc']);group=g.root(image_key(v['image']))
        if not a or a in articles or image_key(v['m_loc']) in images or group in exposed or role.get(group)=='EDIT_EVAL':continue
        by_PMC[a].append(c)
    counts=collections.Counter();assigned_PMC={};assigned_locality={};native_roles=dict(role)
    for a,family in by_PMC.items():
        if a in used_PMC:continue
        target='RETENTION_TRAIN' if counts['RETENTION_TRAIN']<=counts['RETENTION_EVAL'] else 'RETENTION_EVAL'
        if counts[target]>=32:target='RETENTION_EVAL' if target=='RETENTION_TRAIN' else 'RETENTION_TRAIN'
        if counts[target]>=32:break
        family_groups={g.root(image_key(c['original']['image'])) for c in family}
        if any(k in native_roles and native_roles[k]!=target for k in family_groups):
            excluded.append(dict(article=a,reason='WHOLE_GROUP_ROLE_CONFLICT'));continue
        assigned_PMC[a]=target;counts[target]+=1
        for k in family_groups:native_roles[k]=target
        for c in family:assigned_locality[image_key(c['original']['m_loc'])]=target
        selected.append(dict(role=target,article=a,representative=family[0],members=family,patient_independence='UNKNOWN',actual_modality='NOT_REVIEWED'))
    for x in selected:
        for c in x['members']:
            k=g.root(image_key(c['original']['image']));assert native_roles[k]==x['role']
            if x['role']!='EDIT_EVAL':assert assigned_PMC[article(c['original']['m_loc'])]==x['role']
    audit=read(private/'UPSTREAM_AUDIT_PRIVATE.json');assert not audit['patient_mapping_verified']
    summary=dict(status='COMPONENT_RESPLIT_FROZEN_STRICT_SOURCE_GATE_NOT_MET',raw_records=len(records),native_cross_components=len(families),historical_or_V15_components=len(exposed),selected=dict(collections.Counter(x['role'] for x in selected)),frozen_representative_images=sum(2 if x['role']=='EDIT_EVAL' else 1 for x in selected),complete_member_rows=sum(len(x['members']) for x in selected),component_and_PMC_role_conflicts=0,patient_mapping_verified=False,strict_independence_admission=False,actual_modality_admission=False,Base_started=False,fair_comparison_started=False,model_calls=0,pixel_reads=0,upstream_files=24,elapsed_seconds=time.time()-start,not_official_benchmark_split=True)
    write(private/'CANDIDATES_FROZEN.json',dict(summary=summary,candidates=selected,native_component_roles=native_roles,PMC_roles=assigned_PMC,excluded=excluded));write(private/'RESULTS_PRIVATE.json',summary);return summary

def selfcheck():
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'x.tsv';p.write_text('index\timage\tquestion\n1\tDO_NOT_READ_IMAGE_BODY\tq\n');h=header(p,'\t');assert h['columns']==['index','image','question'] and h['header_bytes']<p.stat().st_size and not h['patient_or_study_columns']
        p=Path(t)/'x.json';write(p,[dict(image_id='x',PMC='PMC1',patient_id='p')]);assert metadata(p)['patient_or_study_keys']=={'patient_id':1}
    g=Groups();g.union('a','b');g.union('b','c');roles={g.root('a'):'EDIT_EVAL'};assert roles[g.root('c')]=='EDIT_EVAL';assert not identity_keys(['id','index','PMC','category'])
    return dict(status='PASS',header_no_blob=True,patient_key_not_PMC_or_index=True,transitive_component_role=True,models=0,pixels=0)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['upstream','repartition','selfcheck']);p.add_argument('--data',type=Path);p.add_argument('--private',type=Path);p.add_argument('--history',type=Path);a=p.parse_args()
    if a.mode=='upstream':v=upstream(a.data,a.private);print({k:v[k] for k in ('status','header_patient_fields_present','mapping_patient_fields_present','patient_mapping_verified')})
    elif a.mode=='repartition':print(json.dumps(repartition(a.data,a.private,a.history),indent=2))
    else:print(selfcheck())
