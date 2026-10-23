"""Aggregate complete hidden-arm scores; Base is freshly generated in each route."""
import argparse,json,time
from pathlib import Path


def analyze(root,arms,pairs):
    read=lambda p:json.loads(p.read_text());write=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
    frozen=read(root/'EVALUATION_LABELS_FROZEN.json');assert frozen['status']=='FROZEN_BEFORE_ARM_MAPPING'
    labels={x['id']:x for x in frozen['items']};mapping=read(root/'BLIND_EVALUATION_MAPPING.json')
    by_path={path:labels[key] for key,paths in mapping.items() for path in paths}
    expected=155*len(arms);assert len(by_path)==expected==read(root/'MAIN_COMPLETE.json')['outputs']
    roles={}
    for task in read(root/'TASKS.json'):
        rows={}
        for event in task['events']:
            for qid in event['all_probe_query_ids']:
                if qid in rows:assert rows[qid]==event['task']
                rows[qid]=event['task']
        roles[task['order']]=rows
    parent=Path(read(root/'ENV.json')['PARENT_RUN']);mask=read(parent/'private/M13_SOURCE_PACKET.json')['base_mask']
    records=[];plan=read(root/'EVAL_PLAN.json')
    for owner in range(1,9):
        for index,p in enumerate(x for x in plan if x['owner']==owner):
            query=p['query'];qid=query['query_id'];role=roles[owner].get(qid)
            if role is None:assert query['audit_role']=='FIT';role='FIT'
            base=by_path[f'outputs/Base/{owner}/{index}.json']
            for arm in arms:
                path=f'outputs/{arm}/{owner}/{index}.json';out=read(root/path);l=by_path[path]
                assert out['query']==query
                records.append(dict(owner=owner,query_id=qid,role=role,arm=arm,strict_correct=l['accepted'],complete=l['complete'],
                    label=l['label'],extra_claims=l['extra_claims'],EOS=out['EOS'],at_cap=out['at_cap'],
                    original_Base_correct=mask.get(qid),same_grader_Base_correct=base['accepted'],
                    typo_tolerant=l['accepted'] or l.get('typo_tolerant_sensitivity_only',False),raw_token_ids=out['raw_token_ids']))
    panels={};per_edit={}
    def counts(rows):return dict(correct=sum(x['strict_correct'] for x in rows),total=len(rows))
    for arm in arms:
        rows=[x for x in records if x['arm']==arm];protected=[x for x in rows if x['role'] in ('T1L','T2L')]
        supports=[x for x in protected if x['original_Base_correct'] is True];fresh=[x for x in protected if x['same_grader_Base_correct']]
        panels[arm]=dict(panels={role:counts([x for x in rows if x['role']==role]) for role in ['T0','FIT','T1G','T2G','T1L','T2L']},overall=counts(rows),
            protected_total=len(protected),original_Base_correct_support=len(supports),original_Base_correct_damage=sum(not x['strict_correct'] for x in supports) if supports else None,
            same_grader_Base_correct_support=len(fresh),same_grader_Base_correct_damage=sum(not x['strict_correct'] for x in fresh) if fresh else None,
            extra_claims=sum(x['extra_claims'] for x in rows),EOS=sum(x['EOS'] for x in rows),at_cap=sum(x['at_cap'] for x in rows),
            typo_tolerant_correct=sum(x['typo_tolerant'] for x in rows))
        per_edit[arm]={str(i):{role:counts([x for x in rows if x['owner']==i and x['role']==role]) for role in ['T0','FIT','T1G','T2G','T1L','T2L']} for i in range(1,9)}
    paired={};gates={}
    for left,right in pairs:
        a={(x['owner'],x['query_id']):x for x in records if x['arm']==left};b={(x['owner'],x['query_id']):x for x in records if x['arm']==right}
        out={}
        for role in ['ALL','T0','FIT','T1G','T2G','T1L','T2L']:
            keys=[k for k in a if role=='ALL' or a[k]['role']==role]
            out[role]=dict(total=len(keys),wins=sum(a[k]['strict_correct'] and not b[k]['strict_correct'] for k in keys),
                losses=sum(b[k]['strict_correct'] and not a[k]['strict_correct'] for k in keys),
                exact_token_equal=sum(a[k]['raw_token_ids']==b[k]['raw_token_ids'] for k in keys))
        paired[left+'-'+right]=out
        deltas=[per_edit[left][str(i)]['T2G']['correct']-per_edit[right][str(i)]['T2G']['correct'] for i in range(1,9)]
        preservation=all(per_edit[left][str(i)][role]['correct']>=per_edit[right][str(i)][role]['correct'] for i in range(1,9) for role in ['T0','FIT','T1G'])
        gates[left+'-'+right]=dict(net_T2G=sum(deltas),improved_edits=sum(d>0 for d in deltas),source_FIT_T1G_preserved=preservation,
            preregistered_development_continue=sum(deltas)>=3 and sum(d>0 for d in deltas)>=2 and preservation)
    resources=read(root/'RESOURCE_TARGETS.json')['targets'];end=read(root/'RESOURCE_GUARD_EXIT.json')['time'];seconds=sum((t.get('ended_epoch') or end)-t['started_epoch'] for t in resources if t.get('gpu') is not None)
    result=dict(status='SCORED_ALL_'+str(expected),consumers=expected,unique_full_reviews=len(labels),exact_inherited=frozen['inherited_exact_full_binding'],fresh_full_reviews=frozen['fresh_full_answer_reviews'],
        arms=panels,per_edit=per_edit,paired=paired,development_gates=gates,all8_preserved=True,missing=0,counts=read(root/'MAIN_COMPLETE.json')['counts'],
        gpu_process_seconds_upper_bound=seconds,gpu_cost_note='Unended last-worker lease bounded by guard-confirmed exit; includes failure leases and their carried operation counts.',
        reviewer='Single research assistant full source-reference review; new arm hidden before label freeze, not independent clinician adjudication.',
        scientific_scope='8 exposed development edits, no independent confirmation; original protection support0 means damage NA.',time=time.time())
    write(root/'SCORED_CONSUMERS_PRIVATE.json',records);write(root/'RESULTS.json',result);write(root/'SCORING_COMPLETE.json',dict(status='PASS',consumers=expected,unique=len(labels),time=time.time()));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--arms',nargs='+',required=True);p.add_argument('--pairs',nargs='+',required=True);a=p.parse_args()
    analyze(a.root,a.arms,[x.split(':') for x in a.pairs])
