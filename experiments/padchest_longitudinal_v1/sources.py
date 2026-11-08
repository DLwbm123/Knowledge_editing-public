"""Derive new-task runners without changing recipes or old running files."""
import importlib.util
from pathlib import Path


def once(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)


def build(w):
    w=Path(w);spec=importlib.util.spec_from_file_location('t5_frozen_sources',w/'experiments/med_t5_core_v1/sources.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);out=module.build(w)
    inject='    from pair_input import install\n    install()\n'
    for name in ['initialize.py','bank.py']:
        out[name]=once(out[name],'    import module_run as r\n','    import module_run as r\n'+inject)
    out['bank.py']=out['bank.py'].replace("assert state['arm']=='MDU'", "assert state['arm']=='MDU'")
    cov=out['coverage.py'];cov=once(cov,"'<image>\\n' + question","__import__('pair_input').HEADER + question")
    start=cov.index('    def encode(text):');end=cov.index('    counts = collections.Counter()',start)
    cov=cov[:start]+'''    def encode(text):
        parts=text.split('<image>')
        if len(parts)!=3:raise ValueError('Dual-image placeholder mismatch')
        ids=[];offsets=[];shift=0
        for index,part in enumerate(parts):
            item=tok(part,return_offsets_mapping=True)
            assert item.input_ids[0]==tok.bos_token_id
            keep=0 if index==0 else 1
            ids+=item.input_ids[keep:]
            offsets += [(a+shift,b+shift) if b>a else (0,0) for a,b in item.offset_mapping[keep:]]
            shift+=len(part)
            if index<2:ids.append(-200);offsets.append((0,0));shift+=len('<image>')
        return ids,offsets

'''+cov[end:];out['coverage.py']=cov
    s=(w/'experiments/med_t45_comparators_v1/worker.py').read_text()
    s=once(s,'    import module_run as r\n','    import module_run as r\n'+inject)
    s=once(s,"for cohort in ('T4','T5'):","for cohort in ('T5',):")
    s=once(s,"assert len(tasks) == (67 if cohort == 'T4' else 148)","assert 1<=len(tasks)<=6")
    s=once(s,"assert all(binding[k] == expected[k] for k in binding)","assert all(binding[k] == expected[k] for k in expected)")
    s=once(s,"assert counts['consumers'] == 2696","assert counts['consumers'] == 2*len(plan)")
    out['comparator.py']=s;out['cache_editor.py']=(w/'experiments/med150_grace_correctness_v1/editor.py').read_text()
    for name in ['pair_input.py','base.py','setup_train.py','guard.py','pixel_preflight.py']:
        out[name]=(w/'experiments/padchest_longitudinal_v1'/name).read_text()
    for name,source in out.items():compile(source,name,'exec')
    return out
