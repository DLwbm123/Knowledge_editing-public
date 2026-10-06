"""Bounded targeted acquisition from original JATS and publisher images.

The private plan must be locked before the new-source construction phase.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from audit import read,write,save

def main():
    root=Path(os.environ['V2_ROOT']);p=root/'private';plan=json.loads(Path(os.environ['ACQUISITION_PLAN']).read_text())
    assert len({x['paper'] for x in plan['papers']})<=40 and sum(len(x.get('figures',[])) for x in plan['papers'])<=60
    prior=read(p/'source_registry.jsonl');known_groups={s['source_group_id']:s['assigned_role'] for s in prior};known_hashes={s['image_sha256']:s for s in prior}
    excluded=set()
    native_hash_file=p/'exclusions/native_image_hashes.json'
    native_record=json.loads(native_hash_file.read_text()) # Missing exclusion evidence fails closed.
    assert native_record['all_future_native_count']==594
    native_hashes=set(native_record['native_image_hashes'])
    for f in (p/'exclusions').glob('*.EXCLUSION_IDS.json'):excluded.update(json.loads(f.read_text())['papers'])
    ledger=[];units=[]
    for item in plan['papers']:
        paper=item['paper'];folder=p/'new_primary';meta=folder/(paper+'.metadata.json')
        rec=dict(paper=paper,targeted_edit_positions=item.get('positions',[]),search_query=item.get('query'),acquisition='NEW_PRIMARY_CHECK',outbound_status='UNKNOWN',role_check='NOT_ASSESSED',figures_acquired=0,usable_units=0)
        if not meta.exists():rec['status']='ACCESS_FAILED';ledger.append(rec);continue
        data=json.loads(meta.read_text());rec['license']=data['license']
        lic=' '.join(data['license']).lower()
        allowed=bool(re.search(r'licenses/by/(?:[\d.]+)|license cc-by [\d.]',lic)) and not re.search(r'noncommercial|non-commercial|no.?derivatives|by-nc|by-nd|share.?alike',lic)
        if not allowed:
            rec.update(status='LICENSE_NOT_CONFIRMED_FOR_THIS_OUTBOUND_PROTOCOL',outbound_status='BLOCKED' if lic else 'UNKNOWN');ledger.append(rec);continue
        if paper in excluded:rec.update(status='WHOLE_PAPER_TEST_EXCLUSION',outbound_status='BLOCKED',role_check='BLOCKED_TEST');ledger.append(rec);continue
        group='PMC:'+paper;role=known_groups.get(group,'FIT') # All new pilot groups are FIT, before model construction; no new EVAL.
        rec.update(assigned_role=role,source_group_id=group,outbound_status='ALLOWED',role_check='PASS',status='LICENSE_ROLE_PASS')
        for requested in item.get('figures',[]):
            fid=requested['id'];f=next(x for x in data['figures'] if x['id']==fid);receipt=folder/(paper+'_'+fid+'.ACQUIRED.json')
            if not receipt.exists():rec.setdefault('unit_failures',[]).append(dict(figure=fid,reason='IMAGE_ACCESS_FAILED_OR_BINDING_NOT_CONFIRMED'));continue
            acquired=json.loads(receipt.read_text());image=Path(acquired['path'])
            rec['figures_acquired']+=1
            if not requested.get('image_inspection'):rec.setdefault('unit_failures',[]).append(dict(figure=fid,reason='IMAGE_CAPTION_BINDING_UNRESOLVED'));continue
            sha=hashlib.sha256(image.read_bytes()).hexdigest()
            if sha in native_hashes:
                rec.setdefault('unit_failures',[]).append(dict(figure=fid,reason='FUTURE_OR_FORMAL_NATIVE_IMAGE_HASH'));continue
            if sha in known_hashes:
                rec.setdefault('unit_failures',[]).append(dict(figure=fid,reason='DUPLICATE_KNOWN_IMAGE_GROUP'));continue
            sid=paper+':'+fid;caption=f['caption']
            if 'reproduced' in caption.lower():rec.setdefault('unit_failures',[]).append(dict(figure=fid,reason='THIRD_PARTY_FIGURE_RIGHTS_UNRESOLVED'));continue
            units.append(dict(source_id=sid,source_dataset='PMC_PRIMARY',source_group_id=group,image_id=image.name,image_path=str(image.resolve()),image_sha256=sha,assigned_role=role,source_role_check='PASS',source_text=caption,evidence_origin='ORIGINAL_ANNOTATION',evidence_ids=[sid+':caption'],image_binding=requested['image_inspection'],license=data['license'],primary_url='https://pmc.ncbi.nlm.nih.gov/articles/'+paper+'/',image_URL=acquired['url'],outbound_status='ALLOWED',outbound_reason='Explicit original CC-BY license; anonymous primary caption text only. No medical image bytes, case history, dates, or credentials sent.',acquisition='NEW_PRIMARY',targeted_edit_positions=requested['positions'],modality=requested['modality'],body_part=requested['body_part'],patient_independence='UNKNOWN',visual_label_leakage=requested.get('visual_label_leakage',False),panel_limitations=requested.get('panel_limitations',[])))
            rec['usable_units']+=1
        ledger.append(rec)
    assert len(units)<=60
    write(p/'new_sources_ledger.jsonl',ledger);write(p/'new_units.jsonl',units)
    save(root/'public/ACQUISITION_SUMMARY.json',dict(new_papers_checked=len(ledger),new_figures_acquired=sum(x['figures_acquired'] for x in ledger),new_usable_units=len(units),new_source_groups=len({x['source_group_id'] for x in units}),status_counts={s:sum(x['status']==s for x in ledger) for s in sorted({x['status'] for x in ledger})},new_roles_fixed_before_construction=True,new_eval_created=False,limits=dict(papers=40,units=60)))
    print(json.dumps(dict(new_papers=len(ledger),usable_units=len(units)),indent=2))

if __name__=='__main__':main()
