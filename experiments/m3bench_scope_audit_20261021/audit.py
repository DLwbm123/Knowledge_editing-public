"""Read-only frozen-answer audit. No model, fitting, or new adjudication."""
import json
import re
import signal
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = {'CORRECT', 'INCORRECT', 'UNCERTAIN', 'ABSTAIN'}
ARMS = ('B', 'C160', 'C_STOP')
BIND = ('image_sha256', 'question', 'reference', 'prompt_ids', 'attention_mask', 'runtime', 'generation')

def read(path):
    return json.loads(Path(path).read_text())

def norm(text):
    return re.sub(r'\s+', ' ', text.strip()).rstrip('.').casefold()

def require_verdict(value):
    assert value.get('label') in CLASSES and value.get('reason'), 'Missing frozen four-class verdict'
    return value['label']

def require_binding(output, row, reference=None):
    for key in ('image_sha256', 'question', 'reference'):
        assert output['binding'][key] == row[key], 'Input/reference binding mismatch'
    if reference is not None:
        assert all(output['binding'][k] == reference['binding'][k] for k in BIND), 'Cross-root binding mismatch'

def image_group(row):
    # Same identity rule as the frozen V22 route plan, including derived variants.
    match = re.search(r'xmlab\d+', row['image_path'])
    if match:
        return 'SLAKE:'+match[0]
    match = re.search(r'synpic\d+', row['image_path'], re.I)
    return 'VQA-RAD:'+match[0].lower() if match else 'IMAGE:'+row['image_sha256']

def run(v24, v25, v22, destination):
    signal.alarm(60)
    configs = [read(r/'DATA_ADMISSION.json') for r in (v24, v25)]
    assert configs[0]['tasks'] == configs[1]['tasks'], 'Task definitions differ'
    tasks = configs[0]['tasks']
    assert len(tasks) == 10
    maps = {r: read(r/'BLIND_MAPPING.json') for r in (v24, v25)}
    labels = {}
    bases = {}
    for root in (v24, v25):
        reuse = read(root/'BLIND_REUSE.json'); fresh = read(root/'BLIND_VERDICTS.json')
        assert not set(reuse).intersection(fresh), 'Conflicting verdict sources'
        values = dict(reuse, **fresh)
        assert set(values) == set(maps[root].values()), 'Incomplete frozen verdict coverage'
        labels[root] = {p: require_verdict(values[c]) for p, c in maps[root].items()}
        bases[root] = {p.stem: p for p in (root/'base').glob('*/*.json')}
        assert len(bases[root]) == 180
    rows = []; seen = set()
    for task in tasks:
        part = (task['order']-1) % 5
        for row in task['eval_rows']:
            qid = row['query_id']; identity = (task['key'], qid)
            assert identity not in seen, 'Duplicate consumer'
            seen.add(identity)
            off24 = read(bases[v24][qid]); off25 = read(bases[v25][qid])
            require_binding(off24, row); require_binding(off25, row, off24)
            assert (off24['raw_answer'], off24['raw_token_ids']) == (off25['raw_answer'], off25['raw_token_ids']), 'OFF changed'
            base_label = labels[v24][str(bases[v24][qid].relative_to(v24))]
            assert base_label == labels[v25][str(bases[v25][qid].relative_to(v25))]
            detail = dict(edit=task['key'], query_id=qid, roles=row['roles'], OFF=base_label,
                          source_reference_equal=norm(row['reference']) == norm(task['source']['reference']))
            for arm in ARMS:
                root = v24 if arm == 'B' else v25
                rel = f'{part}/{task["key"]}/outputs/{arm}/{qid}.json'
                output = read(root/rel); require_binding(output, row, off24)
                detail[arm] = dict(label=labels[root][rel], copy=norm(output['raw_answer']) == norm(task['source']['reference']),
                                   tokens=len(output['raw_token_ids']), at_cap=output['at_cap'], EOS=output['EOS'],
                                   decoded_empty=not output['raw_answer'].strip())
            rows.append(detail)
    assert len(rows) == 216 and len({r['query_id'] for r in rows}) == 180
    role_results = {}; edit_results = {}
    def summarize(subset):
        out = dict(consumers=len(subset), unique_inputs=len({r['query_id'] for r in subset}),
                   OFF=dict(Counter(r['OFF'] for r in subset)), arms={})
        for arm in ARMS:
            out['arms'][arm] = dict(classes=dict(Counter(r[arm]['label'] for r in subset)),
                expert_or_OFF_correct=sum(r[arm]['label']=='CORRECT' or r['OFF']=='CORRECT' for r in subset),
                OFF_only_rescues=sum(r[arm]['label']!='CORRECT' and r['OFF']=='CORRECT' for r in subset),
                neither_correct=sum(r[arm]['label']!='CORRECT' and r['OFF']!='CORRECT' for r in subset),
                copy_classes=dict(Counter(r[arm]['label'] for r in subset if r[arm]['copy'])),
                at_cap=sum(r[arm]['at_cap'] for r in subset), decoded_empty=sum(r[arm]['decoded_empty'] for r in subset))
        out['B_C_STOP_or_OFF_correct'] = sum(r['OFF']=='CORRECT' or any(r[a]['label']=='CORRECT' for a in ('B','C_STOP')) for r in subset)
        out['B_C_STOP_correct_sets_equal'] = all((r['B']['label']=='CORRECT') == (r['C_STOP']['label']=='CORRECT') for r in subset)
        out['three_writers_or_OFF_correct'] = sum(r['OFF']=='CORRECT' or any(r[a]['label']=='CORRECT' for a in ARMS) for r in subset)
        out['B_correct_C_STOP_noncorrect'] = sum(r['B']['label']=='CORRECT' and r['C_STOP']['label']!='CORRECT' for r in subset)
        out['C160_to_C_STOP_repairs'] = sum(r['C160']['label']!='CORRECT' and r['C_STOP']['label']=='CORRECT' for r in subset)
        out['C160_to_C_STOP_losses'] = sum(r['C160']['label']=='CORRECT' and r['C_STOP']['label']!='CORRECT' for r in subset)
        return out
    for role in sorted({role for r in rows for role in r['roles']}):
        subset = [r for r in rows if role in r['roles']]
        role_results[role] = summarize(subset)
        edit_results[role] = {t['key']:summarize([r for r in subset if r['edit']==t['key']]) for t in tasks}
    locality = [r for r in rows if set(r['roles']).intersection(('T1L','T2L')) and r['OFF']=='CORRECT']
    damage = {a:dict(base_correct=len(locality), lost=sum(r[a]['label']!='CORRECT' for r in locality), explicit_I=sum(r[a]['label']=='INCORRECT' for r in locality)) for a in ARMS}
    t1l = [r for r in rows if 'T1L' in r['roles']]
    sets = {a:{(r['edit'],r['query_id']) for r in t1l if r[a]['copy'] and r[a]['label']=='INCORRECT'} for a in ARMS}
    overlap = {f'{a}&{b}':len(sets[a]&sets[b]) for a,b in [('B','C160'),('B','C_STOP'),('C160','C_STOP')]}
    support = read(v22/'SUPPORT_FROZEN.json'); plan = read(v22/'ROUTE_PLAN_FROZEN.json')
    sq = {x['row']['query_id'] for x in support}; sg = {x['source_group'] for x in support}
    canonical = {image_group(x['row']) for x in support}
    sharing = {q for q,g in plan['query_groups'].items() if g in canonical}
    support_audit = dict(canonical_support_groups=len(canonical),
        canonical_overlap_route_groups=len(canonical & set(plan['query_groups'].values())),
        route_inputs_sharing_support_canonical_groups=len(sharing),
        canonical_DEV_overlap=sum(plan['query_roles'][q]=='DEV' for q in sharing),
        canonical_HELDOUT_overlap=sum(plan['query_roles'][q]=='HELDOUT_DEVELOPMENT' for q in sharing),
        sharing_input_roles=dict(Counter(role for q in sharing for role in plan['queries'][q]['roles'])),
        native_sources_in_overlap=sum(t['source']['query_id'] in sharing for t in plan['tasks']),
        stored_canonical_group_mismatches=sum(x['source_group']!=image_group(x['row']) for x in support),
        support_rows=len(support), stored_support_groups=len(sg),
        synthetic_negation_rows=sum(x['wrong_choice']=='This is not '+x['row']['reference'] for x in support),
        overlap_evaluation_query_ids=len(sq & {r['query_id'] for r in rows}),
        overlap_route_query_ids=len(sq & set(plan['queries'])),
        overlap_stored_group_strings=len(sg & set(plan['query_groups'].values())),
        nominal_relation_counts=dict(Counter(plan['relations'].values())),
        clinical_applicability_label_field_present=any(any(k in x for k in ('clinical_applicability','adjudication')) for x in support),
        independent_confirmation=plan['independent_confirmation'],
        boundary_support_admission='NOT_ESTABLISHED_IN_INSPECTED_FROZEN_ARTIFACTS',
        patient_independence='UNKNOWN_NOT_INFERRED_FROM_IMAGE_GROUPS',
        remaining_136='DEFERRED_NOT_OPENED',
        note='Stored group string overlap is not a patient-independence or derived-image grouping certificate; synthetic negation is not a clinical boundary negative.')
    lists = dict(B_correct_C_STOP_noncorrect=[r for r in rows if r['B']['label']=='CORRECT' and r['C_STOP']['label']!='CORRECT'],
                 C160_to_C_STOP_repairs=[r for r in rows if r['C160']['label']!='CORRECT' and r['C_STOP']['label']=='CORRECT'],
                 T1L_source_copy_explicit_I={a:[r for r in t1l if r[a]['copy'] and r[a]['label']=='INCORRECT'] for a in ARMS})
    result = dict(status='COMPLETE_ZERO_MODEL_FROZEN_LABEL_AUDIT', consumers=216, unique_inputs=180, edits=10,
        new_model_calls=0, new_judgments=0, roles=role_results, per_edit=edit_results, locality=damage,
        T1L_wrong_copy_intersections=overlap, T1L_source_reference_equal=sum(r['source_reference_equal'] for r in t1l),
        T1L_three_writer_correct_sets_identical=all(len({r[a]['label']=='CORRECT' for a in ARMS})==1 for r in t1l),
        support_admission=support_audit,
        interpretation='Oracle uses exposed frozen correctness, not a trained router or continuous-gate bound. Current judgments are fixed-reference assistant review, not independent clinical adjudication.')
    destination.mkdir(parents=True,exist_ok=False)
    for name,value in [('RESULTS',result),('PRIVATE_ROWS',rows),('PRIVATE_LISTS',lists)]:
        text = json.dumps(value,indent=2)+'\n'
        assert len(text.encode()) < 3_000_000, 'Audit output exceeds finite byte budget'
        (destination/(name+'.json')).write_text(text)
    signal.alarm(0)
    return result

def selfcheck():
    for value in ({'label':False,'reason':'x'},{'label':'INCORRECT'},{}):
        try: require_verdict(value)
        except AssertionError: pass
        else: raise AssertionError('Incomplete/binary verdict accepted')
    try: require_binding({'binding':{'image_sha256':'a','question':'q','reference':'r'}},{'image_sha256':'b','question':'q','reference':'r'})
    except AssertionError: pass
    else: raise AssertionError('Wrong binding accepted')
    return {'status':'PASS','binary_or_missing_verdict_rejected':True,'wrong_binding_rejected':True}
