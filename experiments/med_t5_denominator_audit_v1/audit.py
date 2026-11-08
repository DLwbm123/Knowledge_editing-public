"""Re-express accepted T5 scores; never generate, judge, or change eligibility."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re


def read(path):
    return json.loads(Path(path).read_text())


def form(text):
    text = text.strip()
    if not text:
        return 'EMPTY'
    if re.match(r'yes\b', text, re.I):
        return 'YES_PREFIX'
    if re.match(r'no\b', text, re.I):
        return 'NO_PREFIX'
    if re.match(r'there is no evidence\b', text, re.I):
        return 'EXPLICIT_NO_STATEMENT'
    if re.search(r"(?:cannot|can't|unable to) (?:determine|interpret|diagnose)|as an ai|sorry", text, re.I):
        return 'POSSIBLE_REFUSAL'
    return 'OTHER'


def accuracy(values):
    n = len(values)
    return dict(n=n, correct=sum(v is True for v in values), incorrect=sum(v is False for v in values),
        unknown=sum(v is None for v in values),lower_pct=100*sum(v is True for v in values)/n if n else None,
        upper_pct=100*sum(v is not False for v in values)/n if n else None)


def distribution(events):
    codes = Counter(e['presence_code'] for e in events)
    return dict(n=len(events),presence_codes={k:codes[k] for k in ['11','01','10']},
        followup_reference={'Yes':codes['11']+codes['01'],'No':codes['10']})


def panel(rows, packet, decisions, events):
    consumers = {c['consumer_id']:c for c in packet['consumers']}
    def score(row):
        return decisions['decisions'].get(consumers[row['consumer_id']]['key'],{}).get('core_correct')
    future = [r for r in rows if r['role']=='T5']
    assert len(future)==len({r['owner'] for r in future})
    ee = [events[r['owner']] for r in future]
    for r,e in zip(future,ee):
        assert r['edit_id']==e['event_id'] and r['query_id']==e['followup_query_id']
    before=[e['followup_core'] for e in ee];after=[score(r) for r in future]
    symbols={False:'F',True:'T',None:'U'}
    cells={a+b:[] for a in 'FTU' for b in 'FTU'}
    for e,a,b in zip(ee,before,after):
        cells[symbols[a]+symbols[b]].append(e)
    pre,post=accuracy(before),accuracy(after)
    definite_damage=sum(a is True and b is False for a,b in zip(before,after))
    possible_damage=sum(a is not False and b is not True for a,b in zip(before,after))
    n=len(ee)
    names={'FF':'wrong_to_wrong','FT':'wrong_to_correct','TF':'correct_to_wrong','TT':'correct_to_correct',
        'UF':'unknown_to_wrong','UT':'unknown_to_correct','FU':'wrong_to_unknown','TU':'correct_to_unknown','UU':'unknown_to_unknown'}
    ret=[b for a,b in zip(before,after) if a is True]
    ans=Counter(form(r['answer']['raw_answer']) for r in future)
    return dict(n=n,confusion={names[k]:distribution(v) for k,v in cells.items()},
        denominator_distributions=dict(all=distribution(ee),known_Base_correct=distribution([e for e in ee if e['followup_core'] is True]),
            known_Base_wrong=distribution([e for e in ee if e['followup_core'] is False]),Base_unknown=distribution([e for e in ee if e['followup_core'] is None])),
        ACC_before=pre,ACC_after=post,delta_ACC_pp=[post['lower_pct']-pre['upper_pct'],post['upper_pct']-pre['lower_pct']],
        legacy_all_edit_no_damage_pct=[100*(1-possible_damage/n),100*(1-definite_damage/n)],
        retention_on_known_Base_correct=accuracy(ret),unknown_Base_membership=sum(v is None for v in before),
        raw_followup_answer_forms=dict(ans),EOS=sum(r.get('EOS') is True for r in future),at_cap=sum(r.get('at_cap') is True for r in future),
        T0=accuracy([score(r) for r in rows if r['role']=='T0']))


def base_behavior(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r['judge_input']['question']].append(r)
    multi=[rr for rr in groups.values() if len({r['judge_input']['image_sha256'] for r in rr})>1]
    return dict(n=len(rows),answer_forms=dict(Counter(form(r['answer']['raw_answer']) for r in rows)),
        EOS=sum(r.get('EOS') is True for r in rows),at_cap=sum(r.get('at_cap') is True for r in rows),
        questions=len(groups),questions_with_multiple_images=len(multi),
        multi_image_questions_with_identical_text=sum(len({r['answer']['raw_answer'] for r in rr})==1 for rr in multi),
        images=len({r['judge_input']['image_sha256'] for r in rows}),
        note='Lexical response forms, not new CORE labels or a clinical truth adjudication.')


def main():
    w=Path(os.environ['AUDIT_WORKTREE'])
    output=w/'reports/med_t5_denominator_audit_v1_20261108';output.mkdir(exist_ok=True)
    frozen=read(os.environ['AUDIT_BASE_MASK'])
    events={e['owner']:e for e in frozen['events']}
    material=read(w/'.private/med_t5_release_alignment_v2_20261108/BASE_MATERIAL.private.json')
    targets=[('legacy148_144','MDU',w/'.private/med_t5_core_v1_20261107/METHOD_OUTPUTS.private.json',w/'.private/med_t5_core_v1_20261107/method')]
    for method in ['BALANCEDIT','GRACE','LORA']:
        root=w/'.private/med_t45_comparators_v1_20261108'/method/'T5'
        targets.append(('legacy148_144',method,root/'OUTPUTS.private.json',root/'T5'))
    # The first two release159 methods had accepted scores at audit start.
    for method in ['MDU','LORA']:
        root=w/'.private/med_t5_release_alignment_v2_20261108'/method
        if (root/'DECISIONS.json').exists():
            targets.append(('release159',method,root/'OUTPUTS.private.json',root))
    result=dict(created_UTC=datetime.now(timezone.utc).isoformat(),status='OFFLINE_DENOMINATOR_AND_RESPONSE_AUDIT',
        changes=dict(model_calls=0,Judge_calls=0,retries=0,modified_labels=0,modified_eligibility=0),methods={})
    for cohort,method,outputs,folder in targets:
        values=read(outputs);packet=read(folder/'PACKET_PRIVATE.json');decisions=read(folder/'DECISIONS.json')
        result['methods'].setdefault(cohort,{})[method]={mode:panel([r for r in values if r['mode']==mode],packet,decisions,events) for mode in ['single','bank']}
    base=material['base']
    admitted={o for o,e in events.items() if e['prior_core'] is False}
    result['Base_behavior']={role:base_behavior([r for r in base if r['role']==role]) for role in ['T0','T5']}
    result['Base_behavior_prior_wrong159']={role:base_behavior([r for r in base if r['role']==role and r['owner'] in admitted]) for role in ['T0','T5']}
    result['controls']='Constant-answer controls remain analytical reference-label calculations; no literal Yes/No response was newly judged.'
    result['paper_identity']='Public-release delta_ACC is not proven to be paper-table Temporality. No formula selected to match published numbers.'
    for cohort,methods in result['methods'].items():
        for method,panels in methods.items():
            for mode,p in panels.items():
                assert sum(v['n'] for v in p['confusion'].values())==p['n']
                assert p['n']==(159 if cohort=='release159' else 148 if mode=='single' else 144)
                assert sum(p['raw_followup_answer_forms'].values())==p['n']
    (output/'DENOMINATOR_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for cohort,methods in result['methods'].items():
        for method,panels in methods.items():
            for mode,p in panels.items():
                print(cohort,method,mode,p['n'],{k:v['n'] for k,v in p['confusion'].items() if v['n']},p['ACC_after'],p['raw_followup_answer_forms'])
    print('Base_behavior',json.dumps(result['Base_behavior'],ensure_ascii=False))
    print('Base_behavior_prior_wrong159',json.dumps(result['Base_behavior_prior_wrong159'],ensure_ascii=False))


if __name__=='__main__':
    main()
