"""Join frozen V86 scores to all 775 consumers; reuse fixed Base review explicitly."""
import collections
import json
import math
import os
from pathlib import Path
import time


def read(path):return json.loads(Path(path).read_text())
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def mean(values):return sum(values)/len(values) if values else None


def analyze(root):
    edited=read(root/'EVALUATION_LABELS_FROZEN.json');base_root=root.parent.parent/'source-supported8-v83-20261023/run1';base=read(base_root/'BASE_LABELS_FROZEN.json')
    assert edited['status']=='FROZEN_BEFORE_ARM_MAPPING'
    assert base['status']=='FROZEN_BEFORE_OWNER_MAPPING'
    labels={x['id']:x for x in edited['items']};mapping=read(root/'BLIND_EVALUATION_MAPPING.json')
    path_labels={path:labels[key] for key,paths in mapping.items() for path in paths}
    assert len(path_labels)==775
    base_labels={x['id']:x for x in base['items']};base_mapping=read(base_root/'BLIND_BASE_MAPPING.json')
    by_base={(v['owner'],v['query_id']):base_labels[key] for key,values in base_mapping.items() for v in values}
    tasks=read(root/'TASKS.json');task_roles={}
    for task in tasks:
        roles=collections.defaultdict(set)
        for event in task['events']:
            for qid in event['all_probe_query_ids']:roles[qid].add(event['task'])
        assert all(len(value)==1 for value in roles.values())
        task_roles[task['order']]={k:next(iter(v)) for k,v in roles.items()}
    parent=Path(read(root/'ENV.json')['PARENT_RUN'])
    legacy_base_mask=read(parent/'private/M13_SOURCE_PACKET.json')['base_mask']
    records=[]
    for plan in read(root/'EVAL_PLAN.json'):
        owner=plan['owner'];qid=plan['query']['query_id']
        role=task_roles[owner].get(qid)
        if role is None:
            assert plan['query']['audit_role']=='FIT';role='FIT'
        indices=[x for x in read(root/'EVAL_PLAN.json') if x['owner']==owner]
        index=next(i for i,x in enumerate(indices) if x['query']['query_id']==qid)
        for arm in ('S0','U','CAP','REL','TNC'):
            path=f'outputs/{arm}/{owner}/{index}.json';out=read(root/path);label=path_labels[path]
            records.append(dict(owner=owner,query_id=qid,role=role,arm=arm,path=path,
                strict_correct=label['accepted'],core_complete=label['label']=='C' and label['complete'],
                label=label['label'],extra_claims=label['extra_claims'],EOS=out['EOS'],at_cap=out['at_cap'],
                typo_tolerant=label['accepted'] or label.get('typo_tolerant_sensitivity_only',False),
                same_grader_Base_correct=by_base[(owner,qid)]['accepted'],
                original_Base_correct=legacy_base_mask.get(qid),
                raw_token_ids=out['raw_token_ids']))
    assert len(records)==775 and len({(r['arm'],r['owner'],r['query_id']) for r in records})==775
    per_arm={};per_edit={}
    for arm in ('S0','U','CAP','REL','TNC'):
        rows=[r for r in records if r['arm']==arm];panels={}
        for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
            selected=[r for r in rows if r['role']==role]
            by_owner={i:[r for r in selected if r['owner']==i] for i in range(1,9)}
            panels[role]=dict(correct=sum(r['strict_correct'] for r in selected),total=len(selected),
                micro=mean([r['strict_correct'] for r in selected]),
                macro_over_supported_edits=mean([mean([r['strict_correct'] for r in v]) for v in by_owner.values() if v]),
                supported_edits=sum(bool(v) for v in by_owner.values()))
        protected=[r for r in rows if r['role'] in ('T1L','T2L')]
        support=[r for r in protected if r['original_Base_correct'] is True]
        same_support=[r for r in protected if r['same_grader_Base_correct']]
        per_arm[arm]=dict(panels=panels,overall_correct=sum(r['strict_correct'] for r in rows),total=155,
            protected_total=len(protected),original_Base_correct_support=len(support),
            original_Base_correct_damage=sum(not r['strict_correct'] for r in support) if support else None,
            same_grader_Base_correct_support=len(same_support),
            same_grader_Base_correct_damage=sum(not r['strict_correct'] for r in same_support) if same_support else None,
            extra_claims=sum(r['extra_claims'] for r in rows),abstentions=sum(r['label']=='A' for r in rows),
            uncertainty=sum(r['label']=='U' for r in rows),EOS=sum(r['EOS'] for r in rows),at_cap=sum(r['at_cap'] for r in rows),
            typo_tolerant_correct=sum(r['typo_tolerant'] for r in rows))
        per_edit[arm]={}
        for owner in range(1,9):
            selected=[r for r in rows if r['owner']==owner]
            per_edit[arm][str(owner)]={role:dict(correct=sum(r['strict_correct'] for r in selected if r['role']==role),
                total=sum(r['role']==role for r in selected)) for role in ('T0','FIT','T1G','T2G','T1L','T2L')}
            protected_owner=[r for r in support if r['owner']==owner]
            per_edit[arm][str(owner)]['protection']=dict(support=len(protected_owner),
                damage=sum(not r['strict_correct'] for r in protected_owner) if protected_owner else None)
    paired={}
    for left,right in [('REL','S0'),('REL','CAP'),('TNC','S0'),('U','S0'),('CAP','S0')]:
        l={(r['owner'],r['query_id']):r for r in records if r['arm']==left}
        rr={(r['owner'],r['query_id']):r for r in records if r['arm']==right}
        panel={}
        for role in ('ALL','T0','FIT','T1G','T2G','T1L','T2L'):
            keys=[k for k,v in l.items() if role=='ALL' or v['role']==role]
            panel[role]=dict(total=len(keys),wins=sum(l[k]['strict_correct'] and not rr[k]['strict_correct'] for k in keys),
                losses=sum(rr[k]['strict_correct'] and not l[k]['strict_correct'] for k in keys),
                ties=sum(l[k]['strict_correct']==rr[k]['strict_correct'] for k in keys),
                exact_token_equal=sum(l[k]['raw_token_ids']==rr[k]['raw_token_ids'] for k in keys),
                typo_tolerant_wins=sum(l[k]['typo_tolerant'] and not rr[k]['typo_tolerant'] for k in keys),
                typo_tolerant_losses=sum(rr[k]['typo_tolerant'] and not l[k]['typo_tolerant'] for k in keys))
        paired[left+'-'+right]=panel
    resources=read(root/'RESOURCE_TARGETS.json')['targets']
    gpu_seconds=sum(t['ended_epoch']-t['started_epoch'] for t in resources if t.get('gpu') is not None)
    result=dict(status='SCORED_ALL_775',reviewed_unique_answers=175,reviewed_unique_Base_answers=112,exact_inherited_unique=156,new_hidden_arm_full_reviews=19,Base_review_reused_from='V83 fixed unedited Base same query/image bindings',
        arms=per_arm,per_edit=per_edit,paired=paired,all8_preserved=True,missing=0,
        counts=read(root/'MAIN_COMPLETE.json')['counts'],gpu_process_seconds=gpu_seconds,
        same_grader_protection_support_note='Report NA rather than success if no Base-correct protected examples.',
        original_protection_support_note='Frozen original Base mask retained; edited full-answer scores use current strict assistant rubric, so cross-rubric interpretation requires caution.',
        reviewer='single assistant, full responses hidden against arm before label freeze; source-reference grading, not clinical truth',
        scientific_scope='8 exposed development edits; no independent confirmation',time=time.time())
    write(root/'SCORED_CONSUMERS_PRIVATE.json',records)
    write(root/'RESULTS.json',result)
    write(root/'SCORING_COMPLETE.json',dict(status='PASS',consumers=775,edited_labels=175,Base_labels=112,time=time.time()))
    return result


if __name__=='__main__':analyze(Path(os.environ['OUTPUT_RUN']))
