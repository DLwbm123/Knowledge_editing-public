"""Freeze CORE-only T5 eligibility and explicit temporal denominators."""
from collections import Counter
import json
from pathlib import Path
import statistics
import time


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def bounds(values):
    return dict(n=len(values),unknown=sum(v is None for v in values),
        lower_pct=100*sum(v is True for v in values)/len(values) if values else None,
        upper_pct=100*sum(v is not False for v in values)/len(values) if values else None)


def finalize(cfg,packet,decisions,unknown,charged):
    root=Path(cfg['root']);report=Path(cfg['report'])
    core=lambda key:decisions.get(key,{}).get('core_correct')
    if cfg['stage']=='t5_base':
        by_query={r['query_id']:core(r['key']) for r in packet['consumers']}
        source=json.loads((root/'EVENTS_FROZEN.private.json').read_text())
        events=[]
        for event in source['events']:
            prior=by_query[event['edit_query_id']];follow=by_query[event['followup_query_id']]
            status='NATIVE_UNKNOWN' if prior is None else 'NATIVE_BASE_CORRECT' if prior else 'FOLLOWUP_BASE_UNKNOWN' if follow is None else 'BASE_ELIGIBLE'
            events.append(dict(event,prior_core=prior,followup_core=follow,status=status))
        usable=[e for e in events if e['status']=='BASE_ELIGIBLE']
        result=dict(status='BASE_MASK_FROZEN' if not unknown else 'BASE_MASK_FROZEN_WITH_UNKNOWN',time=time.time(),
            candidate_cases=len(events),confirmed_usable_cases=len(usable),
            confirmed_sequential_cases=sum(e['sequential_role_safe'] for e in usable),
            exclusions_by_status=dict(Counter(e['status'] for e in events if e['status']!='BASE_ELIGIBLE')),
            followup_pre_correct=sum(e['followup_core'] is True for e in usable),
            followup_pre_wrong=sum(e['followup_core'] is False for e in usable),
            cases_by_presence_code=dict(Counter(e['presence_code'] for e in usable)),
            unique_patient_groups=len({e['patient_id'] for e in usable}),
            new_attempt_items=charged,reused_payloads=len(packet['known_valid']),unknown_payloads=len(set(unknown)),
            followup_pre_correct_not_a_gate=True,method_outputs_used_for_selection=False,independent_confirmation=False)
        write(root/'BASE_MASK_FROZEN.json',dict(result,events=events,query_core=by_query))
        write(report/'BASE_QUALIFICATION_RESULTS.json',result)
        write(root/'COMPLETE.json',result)
        return
    frozen=json.loads(Path(cfg['base_mask']).read_text());events={e['owner']:e for e in frozen['events'] if e['status']=='BASE_ELIGIBLE'}
    panels={}
    for mode in ('single','bank'):
        source=[r for r in packet['consumers'] if r['mode']==mode]
        prior=[core(r['key']) for r in source if r['role']=='T0']
        fit={}
        for r in source:
            if r['role']=='FIT':fit.setdefault(r['owner'],[]).append(core(r['key']))
        future=[r for r in source if r['role']=='T5']
        def temporal(rows):
            after=[core(r['key']) for r in rows];before=[events[r['owner']]['followup_core'] for r in rows]
            introduced_free=[True if not pre else post for pre,post in zip(before,after)]
            retention=[post for pre,post in zip(before,after) if pre]
            fix=[post for pre,post in zip(before,after) if not pre]
            result=dict(edits=len(rows),Base_followup_accuracy_pct=100*statistics.mean(before) if before else None,
                post_followup_accuracy=bounds(after),no_introduced_error_all_edits=bounds(introduced_free),
                retention_on_pre_correct=bounds(retention),fix_on_pre_wrong=bounds(fix),
                route_ON=sum(r.get('route_ON') is True for r in rows))
            post=result['post_followup_accuracy'];pre=result['Base_followup_accuracy_pct']
            result['delta_accuracy_pp']=[post['lower_pct']-pre,post['upper_pct']-pre] if rows else None
            return result
        panels[mode]=dict(T0=bounds(prior),FIT=dict(edits=len(fit),events=sum(map(len,fit.values())),
            unknown=sum(v is None for x in fit.values() for v in x),
            lower_pct=100*statistics.mean(sum(v is True for v in x)/len(x) for x in fit.values()) if fit else None,
            upper_pct=100*statistics.mean(sum(v is not False for v in x)/len(x) for x in fit.values()) if fit else None),
            T5=temporal(future),T5_by_presence_code={code:temporal([r for r in future if events[r['owner']]['presence_code']==code]) for code in ('11','01','10')})
    result=dict(status='COMPLETE' if not unknown else 'COMPLETE_WITH_UNKNOWN',time=time.time(),panels=panels,
        rubric='CORE',new_attempt_items=charged,reused_payloads=len(packet['known_valid']),unknown_payloads=len(set(unknown)),
        primary='1 - introduced-error fraction over all eligible edits; explicitly operationalized paper prose',
        secondary=['retention among pre-correct followups','fix among pre-wrong followups','Base/post accuracy','delta accuracy'],
        cohort='Separate source-bound PadChest-GR T5 development evaluation; not exact author cohort',
        main150_modified=False,independent_confirmation=False)
    write(report/'RESULTS.json',result);write(root/'COMPLETE.json',result)
