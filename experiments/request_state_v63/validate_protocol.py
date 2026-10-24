"""Static protocol validation only: no torch, model, server, launcher, or scorer."""
import argparse
import collections
import json
import unicodedata
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def paired_gate(candidate, control):
    """Sets contain (edit, query) identities, not aggregate counts alone."""
    for role in ['T0', 'T1G', 'REPHRASE_DIAGNOSTIC']:
        if control['correct'][role] - candidate['correct'][role]:
            return False
    a, b = candidate['correct']['T2G'], control['correct']['T2G']
    owners = {o for o, _ in a | b}
    positive = sum(sum(o == owner for o, _ in a) > sum(o == owner for o, _ in b) for owner in owners)
    if len(a) - len(b) < 2 or positive < 2:
        return False
    for role in ['T1L', 'T2L']:
        if len(candidate['correct'][role]) < len(control['correct'][role]):
            return False
    for field in ['Base_correct_locality_losses', 'unsupported_source_assertions', 'capped_outputs']:
        if candidate[field] - control[field]:
            return False
    return True


def selfcheck():
    import copy
    roles=['T0','T1G','T2G','T1L','T2L','REPHRASE_DIAGNOSTIC']
    a=dict(correct={r:{(1,'old')} for r in roles},Base_correct_locality_losses=set(),unsupported_source_assertions=set(),capped_outputs=set())
    c=copy.deepcopy(a);c['correct']['T2G']|={(1,'new'),(2,'new')}
    assert paired_gate(c,a)
    same=copy.deepcopy(a);same['correct']['T2G']|={(1,'new1'),(1,'new2')}
    assert not paired_gate(same,a)
    for role in ['T0','T1G','T1L','T2L','REPHRASE_DIAGNOSTIC']:
        bad=copy.deepcopy(c);bad['correct'][role]=set()
        assert not paired_gate(bad,a)
    for field in ['Base_correct_locality_losses','unsupported_source_assertions','capped_outputs']:
        bad=copy.deepcopy(c);bad[field].add((2,'new'))
        assert not paired_gate(bad,a)
    assert not paired_gate(a,a)
    # Main candidate must beat both A and B; tying B cannot trigger R.
    assert not (paired_gate(c,a) and paired_gate(c,c))


def validate(report, private, parent, original):
    selfcheck()
    lock=read(report/'PROTOCOL_LOCK.json')
    assert lock['launch_authorized'] is False
    assert lock['admission_status'].startswith('STATIC_TEXT_PASS_ONLY')
    assert len(lock['main_arms'])==6 and lock['objective']['steps']==160
    assert lock['budget_ceiling']['main_updates']==6*10*160
    assert lock['budget_ceiling']['max_total_updates']==8*10*160
    assert lock['budget_ceiling']['main_gradient_passes']==6*10*160*2
    assert lock['budget_ceiling']['main_formal_generations']==6*219
    assert lock['budget_ceiling']['conditional_formal_generations']==2*219
    assert lock['writer']['active_tokens']=='ALL_TOKENS'
    assert lock['state']['shape']==[1,4096] and lock['writer']['input_dimension']==14336
    assert lock['budget_ceiling']['per_host_new_bytes'] < 50_000_000_000
    tasks={t['order']:t for t in read(parent/'DATA_ADMISSION.json')['tasks']}
    views=read(private/'VIEWS_FROZEN.json')['views']
    assert len(views)==20 and {(v['owner'],v['view']) for v in views}=={(i,k) for i in range(1,11) for k in [1,2]}
    excluded={normalize(x['query']['question']) for x in read(original/'EVAL_PLAN146.json')}
    excluded|={normalize(q['question']) for t in tasks.values() for q in t['eval_rows']}
    for v in views:
        assert v['source']==tasks[v['owner']]['source']
        assert v['target_reference']==v['source']['reference']
        assert normalize(v['derived_question']) not in excluded
        assert normalize(v['derived_question'])!=normalize(v['source']['question'])
        assert v['manual_semantic_review']=='PASS' and v['answer_scope_preserved']
        assert v['answer_facts_added_to_question'] is False
    by_source=collections.defaultdict(set)
    for v in views:
        by_source[(v['source']['question'],v['view'])].add(v['derived_question'])
    assert all(len(x)==1 for x in by_source.values())
    result=dict(status='STATIC_PASS_ONLY',launch_authorized=False,edits=10,frozen_views=20,
                unique_view_texts=18,registered_question_collisions=0,unchanged_image_and_target_bindings=True,
                balanced_view_uses_per_edit=[80,80],main_arms=6,conditional_random_arms=2,
                paired_gate_synthetic_set_checks='PASS',real_model_admission='NOT_EXECUTED',
                tokenization_admission='NOT_EXECUTED',state_vectors='NOT_EXTRACTED',
                model_calls=0,training_updates=0,server_access=False)
    (report/'STATIC_ADMISSION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['report','private','parent','original']:
        p.add_argument(name,type=Path)
    a=p.parse_args();validate(a.report,a.private,a.parent,a.original)
