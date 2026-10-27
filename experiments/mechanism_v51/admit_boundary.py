"""Bounded existing-source metadata admission; never invent medical scope truth."""
import json,re
from pathlib import Path

def identity(value):
    return Path(str(value)).name.lower()

def inventory(repo,data):
    frozen=repo/'.private/medmkeb_independent_admission_v15_20261019/CANDIDATES_FROZEN.json'
    candidates=json.loads(frozen.read_text())['candidates'];assert len(candidates)==64
    seen=set();seen_articles=set();inputs=[]
    # Prior admitted panels conservatively count as exposed, whether or not scored.
    for d in sorted((repo/'.private').iterdir()):
        if not d.is_dir() or not d.name.startswith(('medmkeb_','m3bench_','evidence_scope_')):continue
        choices=[d/'DATA_ADMISSION.json',d/'completed/DATA_ADMISSION.json',d/'stage/DATA_ADMISSION.json',d/'remote/DATA_ADMISSION.json',d/'remote_snapshot/DATA_ADMISSION.json']
        p=next((p for p in choices if p.exists()),None)
        if p is None:continue
        assert p.stat().st_size<64*1024**2
        def walk(x):
            if isinstance(x,dict):
                for k,v in x.items():
                    if isinstance(v,str):
                        seen_articles.update(re.findall(r'PMC[0-9]+',v.upper()))
                        if k in ('image_path','image','m_loc','image_rephrase'):seen.add(identity(v))
                    elif isinstance(v,(dict,list)):walk(v)
            elif isinstance(x,list):
                for v in x:walk(v)
        walk(json.loads(p.read_text()));inputs.append(str(p))
    records=[]
    for i,c in enumerate(candidates):
        v=c['original'];p=data/v['m_loc'];used=identity(p) in seen or c['article'] in seen_articles
        records.append(dict(pool_index=i,source_cluster=c['article'],patient_identity='UNKNOWN',original_split=c['split'],image=str(p),question=v['m_loc_q'],reference=v['m_loc_a'],local_image_available=p.is_file(),prior_admission_exposed=used,role='UNASSIGNED',scope_label='UNKNOWN',reference_status='DATASET_REFERENCE_ONLY_NOT_YET_SOURCE_VERIFIED',evidence_type='NO_VERIFIED_ROI',status='EXCLUDE_PREVIOUS_PANEL' if used else 'NEEDS_SOURCE_AND_IMAGE_REVIEW'))
    return dict(source_pool='64 previously metadata-only V15 candidates; current boundary development admission is separate from failed V15 independent-edit gate',source_manifest=str(frozen),history_inputs=inputs,records=records,summary=dict(pool=64,prior_panel_excluded=sum(x['prior_admission_exposed'] for x in records),local_images_available=sum(x['local_image_available'] for x in records),complete_task_packs=0,patient_independent=False,model_calls=0,pixels_reviewed=0,valid_negative_pairs=0))

if __name__=='__main__':
    import sys
    repo,data,out=map(Path,sys.argv[1:]);x=inventory(repo,data);out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(json.dumps(x['summary']))
