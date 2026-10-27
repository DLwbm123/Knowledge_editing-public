"""Apply recorded conservative source-only judgments; no model outputs."""
import collections
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.private/med150_source_review_v1_20261028'


def main():
    packet=json.loads((OUT/'SOURCE_PACKET_PRIVATE.json').read_text())['records']
    groups=json.loads((ROOT/'.private/med150_source_metadata_v1_20261028/CANDIDATES_FROZEN.json').read_text())['selections']
    manual=json.loads((OUT/'MANUAL_UNCERTAINTY.private.json').read_text())
    assert manual['source_only'] and not manual['model_outputs_seen'] and not manual['image_pixels_seen']
    reasons={q:reason for reason,ids in manual['reasons'].items() for q in ids};assert len(reasons)==7
    rows=[];group_totals=[]
    for group_index,g in enumerate(groups,1):
        seen={};count=collections.Counter();images=set()
        for q in g['qid_pool']:
            v=packet[str(q)];images.add(v['img_name']);key=(v['question'].strip().casefold(),v['answer'].strip().casefold())
            if key in seen:status='DUPLICATE_SOURCE_QA';detail='Exact question and reference repeat in the same image group'
            elif q in reasons:status='UNKNOWN';detail=reasons[q]
            else:status='ANNOTATED_CANDIDATE_PENDING_IMAGE_AND_BASE';detail='Source annotation only; no image truth, clinical or Base qualification'
            seen.setdefault(key,q);count[status]+=1;rows.append(dict(group_index=group_index,qid=q,status=status,reason=detail))
        assert len(images)==1
        group_totals.append(dict(group_index=group_index,rows=len(g['qid_pool']),unique_image_paths=1,counts=dict(count)))
    assert len(rows)==48 and set(reasons)<={x['qid'] for x in rows}
    c=collections.Counter(x['status'] for x in rows)
    assert c==dict(ANNOTATED_CANDIDATE_PENDING_IMAGE_AND_BASE=37,DUPLICATE_SOURCE_QA=4,UNKNOWN=7)
    value=dict(status='SOURCE_ANNOTATIONS_REVIEWED_NOT_ADMITTED',time=time.time(),rows=rows,group_totals=group_totals,
        counts=dict(c),distinct_image_groups=8,image_pixels_read=0,base_calls=0,train_calls=0,judge_calls=0,
        clinical_truth='UNKNOWN',patient_independence='UNKNOWN',base_eligibility='NOT_TESTED',additional_sources_used=False)
    target=OUT/'SOURCE_REVIEW_PRIVATE.json';assert not target.exists();target.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ('rows','group_totals')}))

if __name__=='__main__':main()
