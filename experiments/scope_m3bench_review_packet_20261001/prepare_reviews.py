"""Prepare review-only proposals from the fixed source-role ledger."""
import collections
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ.get('RUN_ROOT', '/tmp'))
SOURCE = Path(os.environ.get('SOURCE_ROOT', '/tmp'))
ROLE = Path(os.environ.get('ROLE_ROOT', '/tmp'))
BLOCKED = {'FROZEN_FIT_SOURCE', 'FORMAL_OR_RESERVED_SOURCE', 'CAL_CHECK_SUPPORT_RESERVED', 'PROTECTED_SPLIT'}


def allowed(roles):
    return bool(roles) and not BLOCKED.intersection(roles)


def selfcheck():
    assert allowed(['BASE_QUERY_INVENTORY', 'BENCHMARK_METADATA'])
    assert not allowed([])
    for role in BLOCKED:
        assert not allowed([role, 'BASE_QUERY_INVENTORY'])


def main():
    started = time.time()
    selfcheck()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    argv = Path('/proc/self/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert not any(k in argv.lower() for k in ['wangbomin', 'knowledge_editing', 'scope'])
    sys.path.insert(0, str(ROOT))
    from candidate_rules import bucket
    inputs = [SOURCE/'private/SOURCE_QA_ROLE_AUDIT.json', SOURCE/'private/FACT_CARDS.json', ROLE/'private/IDENTITY_ROLE_LEDGER.json']
    assert sum(p.stat().st_size for p in inputs) < 64*1024**2
    rows, cards, ledger = [json.loads(p.read_text()) for p in inputs]
    roles = {(r['dataset'], r['image_id'], r['qid']): r['roles'] for r in ledger}
    pool = [r for r in rows if allowed(roles[(r['dataset'], r['image_id'], r['qid'])]) and r['image_exists'] and (r['dataset'] != 'VQA-RAD' or r['source_QA_evaluation'] == 'evaluated')]
    packet, summary = [], []
    for card in cards:
        anchor = card['anchor']
        groups = {k: [] for k in ['EXACT_Q_EXACT_A', 'EXACT_Q_DIFFERENT_A', 'ATTRIBUTE_CONTENT_OVERLAP']}
        for candidate in pool:
            if candidate['dataset'] == anchor['dataset'] and candidate['image_id'] != anchor['image_id']:
                key = bucket(anchor, candidate)
                if key:
                    groups[key].append(candidate)
        for key, values in groups.items():
            values.sort(key=lambda r: (r['image_id'], r['qid']))
            for index, candidate in enumerate(values[:4]):
                packet.append(dict(review_id=f'P{card["edit_order"]}-{key}-{index+1}', edit_order=card['edit_order'], anchor=anchor, candidate=candidate, source_roles=roles[(candidate['dataset'], candidate['image_id'], candidate['qid'])], retrieval_bucket=key, scope_label='UNKNOWN', data_role='REVIEW_ONLY_EXPOSURE_UNVERIFIED', admitted_CAL=False, patient_independence='UNKNOWN', reviewer_1=None, reviewer_2=None, evidence=None, adjudication=None))
        summary.append(dict(pilot_order=card['edit_order'], totals={k: len(v) for k,v in groups.items()}, selected=sum(min(len(v),4) for v in groups.values())))
    assert all(x['scope_label'] == 'UNKNOWN' and not x['admitted_CAL'] and allowed(x['source_roles']) for x in packet)
    (ROOT/'private').mkdir(exist_ok=True)
    (ROOT/'private/BLINDED_REVIEW_PACKET.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
    fields=['review_id','edit_order','anchor_question','anchor_reference','candidate_image_path','candidate_question','candidate_reference','source_annotation','source_roles','scope_label','reviewer_1','reviewer_2','evidence','adjudication']
    with (ROOT/'private/BLINDED_REVIEW.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for x in packet:
            a,c=x['anchor'],x['candidate'];writer.writerow(dict(review_id=x['review_id'],edit_order=x['edit_order'],anchor_question=a['question'],anchor_reference=a['answer'],candidate_image_path=c['image_path'],candidate_question=c['question'],candidate_reference=c['answer'],source_annotation=c['source_QA_evaluation'],source_roles='|'.join(x['source_roles']),scope_label='UNKNOWN'))
    public=dict(status='COMPLETE_WAITING_REAL_SCOPE_REVIEW',source_role_qualified_for_review=2070,source_QA_annotation_filtered=len(pool),filtered_by_dataset=dict(collections.Counter(r['dataset'] for r in pool)),proposals=len(packet),distinct_candidate_inputs=len({(x['candidate']['dataset'],x['candidate']['image_id'],x['candidate']['qid']) for x in packet}),distinct_candidate_images=len({(x['candidate']['dataset'],x['candidate']['image_id']) for x in packet}),per_edit=summary,scope_labels={'UNKNOWN':len(packet)},scope_verification_complete=False,independent_CAL=False,independent_CONFIRM=False,exposure_status='UNVERIFIED; existing Base/pool registrations retained',new_verified_CAL=0,GPU_hours=0,new_Judge=0,training_steps=0,new_generation=0)
    (ROOT/'public/REVIEW_PACKET_AGGREGATES.json').write_text(json.dumps(public,indent=2)+'\n')
    manifest=json.loads((ROOT/'RUN_MANIFEST.json').read_text())
    assert time.time()-started<60 and time.time()<manifest['deadline_epoch'] and not (ROOT/'STOP').exists()
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<16*1024**2
    (ROOT/'public/FINAL_EXECUTION_AUDIT.json').write_text(json.dumps(dict(status='COMPLETE',role_guard_selfcheck='PASS',all_labels_unknown='PASS',CPU_seconds=time.time()-started,input_bytes=sum(p.stat().st_size for p in inputs),GPU_hours=0,new_Judge=0,verified_scope=0)))
    (ROOT/'PROCESS_RECEIPT.json').write_text(json.dumps(dict(pid=os.getpid(),argv=argv,started_epoch=started,CUDA_VISIBLE_DEVICES='')))
    (ROOT/'RUN_STATUS.json').write_text(json.dumps(dict(status='COMPLETE',phase='CLOSED_WAITING_REAL_SCOPE_REVIEW',epoch=time.time())))
    print(json.dumps(public))


if __name__=='__main__':
    if '--selfcheck' in sys.argv:
        selfcheck();print('PASS: every protected source role is rejected')
    else:
        try:main()
        except Exception as e:
            (ROOT/'FAILURE.json').write_text(json.dumps(dict(error=str(e),traceback=traceback.format_exc())))
            (ROOT/'RUN_STATUS.json').write_text(json.dumps(dict(status='FAILED_PRESERVED')))
            raise
