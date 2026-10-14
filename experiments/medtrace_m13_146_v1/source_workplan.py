"""Original-question review queue; proposal and candidate pools never imply admission."""
import collections
import contract


def build(packet):
    gate=contract.source_admission(packet);rows=[]
    for task in packet['original_tasks']:
        owner=task['order'];used=len(packet['reviews']['owners'].get(str(owner),[]))
        if owner==8:used=max(used,len(packet['extra_review']['rows'])+6)
        cap=12 if owner==8 else 6
        state='QUALIFIED' if owner in gate['qualified_owners'] else ('WAIT_ORIGINAL_EVIDENCE_NO_THIRD_ROUND' if used>=cap else 'BOUNDED_REVIEW_REQUIRED')
        rows.append(dict(owner=owner,state=state,candidates_used=used,candidate_limit=cap,
            remaining_candidates=max(0,cap-used),required_independent_images=2,
            task_binding=contract.original.task_binding(task)))
    counts=collections.Counter(row['state'] for row in rows)
    return dict(private_rows=rows,public=dict(original_edits=146,qualified=counts['QUALIFIED'],
        bounded_review_required=counts['BOUNDED_REVIEW_REQUIRED'],exhausted_waiting_original_evidence=counts['WAIT_ORIGINAL_EVIDENCE_NO_THIRD_ROUND'],
        maximum_remaining_candidate_reviews=sum(row['remaining_candidates'] for row in rows if row['state']=='BOUNDED_REVIEW_REQUIRED'),
        original_questions_unchanged=True,all_146_required_before_GPU=True,training_ready=False))
