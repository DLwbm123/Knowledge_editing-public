"""Join explicit private source-frame review with source modality evidence."""
import collections
import importlib.util
import json
import os
import time
from pathlib import Path

spec=importlib.util.spec_from_file_location('p0',os.environ['P0_ENTRY'])
p0=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)


def main():
    start,cpu=time.time(),time.process_time()
    root=Path(os.environ['RUN_ROOT'])
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    policy=json.loads((root/'RUN_MANIFEST.json').read_text())
    assert time.time()<policy['deadline_epoch'] and not (root/'STOP').exists()
    source=root/'private/source_supplement/final'
    packet=json.loads((source/'RELATION_REVIEW_PACKET.json').read_text())
    review=json.loads((source/'BODY_FRAME_REVIEW.json').read_text())
    by_id={x['review_id']:x for x in packet}
    assert len(by_id)==len(packet)
    fits=[x for x in packet if x['admitted_H_fit']]
    for note in review:
        item=by_id[note['review_id']]
        assert not item['admitted_H_fit'] and item['relation_label']=='UNKNOWN'
        assert item['template'] in {'location_single_lesion','location_single_mass','location_abnormality'}
        raw=item['source']['original_source_row']
        assert raw['evaluation']=='evaluated' and raw['phrase_type']=='freeform'
        assert raw['image_organ']==note['source_body']
        assert note['reviewer']=='Codex source and pixel audit'
        assert note['source_pixel_frame_checked'] is True and note['clinical_verification'] is False
        assert (note['source_body'],note['native_target_frame']) in {('HEAD','lung'),('HEAD','liver'),('ABD','intracranial')}
        assert note['source_answer_in_source_frame'] is True
        fits.append(dict(item,admitted_H_fit=True,relation_label='H_SOURCE_VERIFIED',evidence=note))
    assert len({x['review_id'] for x in fits})==len(fits)
    ledger=json.loads((root/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json').read_text())
    counts=collections.Counter(x['edit_id'] for x in fits)
    gaps=[dict(position=i,edit_id=eid,H_fit_relations=counts[eid],
               status='SOURCE_VERIFIED_H_SUPPORTED' if counts[eid] else 'MISSING_H_EVIDENCE')
          for i,eid in enumerate(ledger['main_T0'],1)]
    out=root/'private/H_SUPPORT_FINAL'
    p0.write_new(out,'SOURCE_VERIFIED_H_FIT.json',fits)
    p0.write_new(out,'FULL_COHORT_H_COVERAGE.json',gaps)
    p0.write_new(out,'UNKNOWN_REVIEW_QUEUE.json',[x for x in packet if x['review_id'] not in {y['review_id'] for y in fits}])
    summary=dict(status='P0_PARTIAL_H_SUPPORT_GPU_NOT_ADMITTED',N=len(gaps),
                 H_supported_edits=sum(x['H_fit_relations']>0 for x in gaps),H_missing_edits=sum(x['H_fit_relations']==0 for x in gaps),
                 H_source_verified_relations=len(fits),H_distinct_source_QA=len({(x['source']['dataset'],x['source']['qid']) for x in fits}),
                 H_distinct_source_images=len({(x['source']['dataset'],x['source']['image_id']) for x in fits}),
                 modality_supported_edits=len({x['edit_id'] for x in fits if x['template']=='modality'}),
                 source_frame_supported_edits=len({x['edit_id'] for x in fits if x['template']!='modality'}),
                 remaining_UNKNOWN_relations=len(packet)-len(fits),
                 verification='ORIGINAL_SOURCE_QA; MODALITY_CONTRAST_OR_EXPLICIT_SOURCE_FRAME_SCOPE_REVIEW',
                 clinical_verified_relations=0,patient_independence='UNKNOWN',
                 GPU_admitted=False,GPU_hours=0,training=0,generation=0,new_Judge=0,
                 original_cohort_and_order_preserved=True,smaller_cohort_substitution=False,
                 CPU_seconds=time.process_time()-cpu,task_wall_seconds=time.time()-start)
    p0.write_new(root,'public/H_SUPPORT_FINAL_SUMMARY.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':main()
