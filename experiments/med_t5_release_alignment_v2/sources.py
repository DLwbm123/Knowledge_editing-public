"""Count/path adaptation only; preserve comparator and selected MDU recipes."""
from pathlib import Path
import sys


def once(source, old, new):
    assert source.count(old)==1,old
    return source.replace(old,new)


def build(w):
    w=Path(w)
    s=(w/'experiments/med_t45_comparators_v1/worker.py').read_text()
    for old,new in [("for cohort in ('T4','T5'):","for cohort in ('T5',):"),
        ("len(tasks) == (67 if cohort == 'T4' else 148)","len(tasks) == 159"),
        ("{'GRACE':42600, 'BALANCEDIT':10750, 'LORA':2130}","{'GRACE':31800, 'BALANCEDIT':7950, 'LORA':1590}"),
        ("counts['consumers'] <= 2696","counts['consumers'] <= 1908"),
        ("counts['consumers'] == 2696","counts['consumers'] == 1908"),
        ("all(binding[k] == expected[k] for k in binding)","all(binding[k] == expected[k] for k in expected)")]:s=once(s,old,new)
    score=(w/'experiments/med_t4_core_v1/score.py').read_text()
    branch="""    if cfg.get('stage') == 'release_t5':
        sys.path.insert(0,cfg['alignment_source'])
        from metrics import evaluate
        events=read(root/'EVENTS_FROZEN.private.json')['events']
        result=evaluate(events,packet['consumers'],decisions)
        result.update(status='COMPLETE_WITH_UNKNOWN' if unknown or any(e['followup_core'] is None for e in events) else 'COMPLETE',time=time.time(),
            unknown_payloads=len(set(unknown)),new_attempt_items=charged,protocol=PROTOCOL)
        write(Path(cfg['report'])/'RESULTS.json',result)
        write(root/'COMPLETE.json',result)
        return
"""
    score=once(score,"    if cfg.get('stage') in ('t5_base', 't5_method'):",branch+"    if cfg.get('stage') in ('t5_base', 't5_method'):")
    sources=dict(worker=s,score=score,cache_editor=(w/'experiments/med150_grace_correctness_v1/editor.py').read_text())
    sys.path.insert(0,str(w/'experiments/med_t5_core_v1'))
    # Avoid importing this module again under the generic sources name.
    import importlib.util
    spec=importlib.util.spec_from_file_location('frozen_t5_sources',w/'experiments/med_t5_core_v1/sources.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    mdu=module.build(w)
    marker="    for mode in ('single','bank'):"
    injection="""    reuse=json.loads((ROOT/'REUSE_SINGLE_OUTPUTS.private.json').read_text())
    prior={(v['edit_id'],v['role'],v['query_id']):v for v in reuse if v['mode']=='single'}
"""
    mdu['bank.py']=once(mdu['bank.py'],marker,injection+marker)
    marker="            key=r.c.digest(signature);reused=key in cache"
    injection="""            old=prior.get((task['edit_id'],row['role'],query['query_id'])) if mode=='single' else None
            if old:
                assert all(old['judge_input'][k]==item[4][k] for k in ['question','reference','image_sha256','prompt_ids','attention_mask'])
                assert old['state_hash']==signature['effective_state'] and old['effective_expert']==chosen
                cache[r.c.digest(signature)]=old['answer']
"""
    mdu['bank.py']=once(mdu['bank.py'],marker,injection+marker)
    for name,body in [*sources.items(),*mdu.items()]:compile(body,name,'exec')
    return sources,mdu
