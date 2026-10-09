"""Protect only previously qualified Base responses using their actual token paths."""
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
import initial_probe as initial

q, c, p, RUN = initial.q, initial.c, initial.p, initial.RUN
PARENT = Path(os.environ['ANSWER_PARENT'])


def split():
    qualification = c.read(PARENT/'private/SEMANTIC_QUALIFICATION.json')['rows']
    assert len(qualification) == 192 and all(x['status']=='FORMAT_VALID' for x in qualification)
    basis, held = [], []
    for item in qualification:
        row = dict(item['row'],semantic_correct=bool(item['correct']),response_path=item['path'])
        assert row['role']=='REPLAY_FIT'
        if row['audit_role']=='BASIS' and row['semantic_correct']:basis.append(row)
        elif row['audit_role']=='HELDOUT':held.append(row)
    # Preserve source order rather than SQLite's payload ordering.
    order = {r['query_id']:i for i,r in enumerate(c.read(PARENT/'private/ANSWER_ROWS.json'))}
    basis.sort(key=lambda r:order[r['query_id']]);held.sort(key=lambda r:order[r['query_id']])
    assert len(basis)==61 and len(held)==96 and sum(r['semantic_correct'] for r in held)==63
    assert not {r['source_group'] for r in basis} & {r['source_group'] for r in held}
    return basis, held


def protection_batch(runtime, row, task):
    saved = c.read(row['response_path'])
    assert saved['unedited_Base'] and saved['binding']['input']['query_id']==row['query_id']
    tokens = saved['R0']['raw_token_ids']
    assert 0<len(tokens)<=128 and tokens[-1]==runtime.adapter.tokenizer.eos_token_id
    kwargs, labels, mask, binding = p.tr.teacher_batch(runtime,row,tokens)
    assert binding['image']==saved['binding']['judge_input']['image_sha256']
    assert binding['prompt_tokens']==saved['binding']['judge_input']['prompt_ids']
    assert int(mask.sum())==len(tokens)
    kwargs=dict(kwargs,labels=labels)
    return SimpleNamespace(labels=labels,target_token_ids=tuple(tokens),forward_kwargs=lambda:kwargs)


q.split=split
q.protection_batch=protection_batch


def plan():
    import qualified_report
    qualified_report.selfcheck()
    initial.selfcheck()
    basis, held=split()
    parent=c.read(PARENT/'public/RESULTS.json')
    assert parent['decision']=='SEMANTIC_SUPPORT_AVAILABLE' and parent['payload_status']=={'FORMAT_VALID':192}
    assert len({r['source_group'] for r in basis})==31
    assert len({r['source_group'] for r in held if r['semantic_correct']})==29
    for row in basis+held:
        saved=c.read(row['response_path']);tokens=saved['R0']['raw_token_ids']
        assert not saved['audit']['at_generation_cap'] and 0<len(tokens)<=128
    lock=dict(experts=8,basis_questions=61,basis_groups=31,holdout_questions=96,holdout_groups=32,
        primary_heldout_questions=63,primary_heldout_groups=29,forward_calls=4552,backward_calls=504,
        candidate_optimizer_steps=8,new_generations=0,new_Judge=0,persistent_checkpoints=0,
        initial_state='ORIGINAL_SEEDED_ZERO_TT88',target='ACTUAL_SEMANTICALLY_CORRECT_BASE_RESPONSE_TOKENS',
        source_roles=['BASIS_SEMANTIC_CORRECT_ONLY'],heldout_geometry=False,CHECK_used=False,replay_CE_optimization=False,
        role_binding=c.digest([basis,held]),code=c.read(RUN/'private/GPU_SOURCE_VERSION.json'))
    c.write(RUN/'private/PROBE_LOCK.json',lock)
    c.write(RUN/'public/ADMISSION.json',dict(status='PASS',**{k:v for k,v in lock.items() if k not in ('role_binding','code')},
        projection_selfcheck=q.pm.selfcheck(),function_selfcheck=q.fm.selfcheck()))
    p.done('PLAN_COMPLETE')


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('qualified_probe.py','qualified_worker',g,i) for i,g in enumerate(q.GPUS)])
    pipeline.wait([pipeline.launch('qualified_report.py','qualified_report')])


if __name__=='__main__':
    try:
        {'qualified_plan':plan,'qualified_worker':q.worker,'qualified_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),
                dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
