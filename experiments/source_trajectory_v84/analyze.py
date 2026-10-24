"""Apply frozen source labels before checking the registered ranking admission."""
import json
import os
from pathlib import Path
import time


def read(p):return json.loads(Path(p).read_text())
def analyze(root):
    labels=read(root/'LABELS_FROZEN.json')
    assert labels['status']=='FROZEN_BEFORE_OWNER_AND_RANK_MAPPING'
    mapping=read(root/'BLIND_TRAJECTORY_MAPPING.json');errors=[];per_edit={str(i):dict(total=0,correct=0,errors=0) for i in range(1,9)}
    for label in labels['items']:
        d=read(root/mapping[label['id']]);item=per_edit[str(d['owner'])];item['total']+=1;item['correct']+=int(label['accepted'])
        if not label['accepted'] and d['EOS'] and not d['at_cap']:
            item['errors']+=1;teacher=d['teacher_scores']
            errors.append(dict(owner=d['owner'],view=d['view'],teacher_prefers_all_correct=all(x['mean_log_probability']>teacher[3]['mean_log_probability'] for x in teacher[:3])))
    assert len(labels['items'])==40 and all(v['total']==5 for v in per_edit.values())
    owners={e['owner'] for e in errors};teacher_pass=all(e['teacher_prefers_all_correct'] for e in errors) if errors else None
    admitted=len(owners)>=2 and teacher_pass is True
    targets=read(root/'RESOURCE_TARGETS.json')['targets'];gpu_start=next(t['started_epoch'] for t in targets if t.get('gpu') is not None)
    result=dict(status='ADMITTED_SOURCE_SIGNAL' if admitted else 'CLOSED_NO_SOURCE_ERRORS' if not errors else 'CLOSED_SOURCE_OR_TEACHER_COVERAGE',
        samples=40,strict_correct=sum(x['accepted'] for x in labels['items']),actual_factual_errors=len(errors),error_edits=len(owners),
        teacher_advantage_on_errors=teacher_pass,teacher_advantage_note='Not testable when there is no erroneous source trajectory; never vacuous success.',
        per_edit=per_edit,counts=read(root/'EXIT.json')['counts'],
        gpu_process_upper_bound_seconds=read(root/'RESOURCE_GUARD_EXIT.json')['time']-gpu_start,
        time_accounting='Conservative bound from worker launch until guard observed no registered process, includes at most polling tail.',
        no_training=True,qualification_only=True,requires_new_evidence_before_rank_training=True,time=time.time())
    (root/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (root/'REVIEW_COMPLETE.json').write_text(json.dumps(dict(status='PASS',labels=40,time=time.time()),indent=2)+'\n')
    return result


if __name__=='__main__':analyze(Path(os.environ['OUTPUT_RUN']))
