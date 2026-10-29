"""Prepare the two-arm worker from frozen V133 source; no launch occurs here."""
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('unexpected V133 source integration point')
    return text.replace(old, new, 1)


def build(source):
    text = replace_once(source, '    import med133_curvature as curvature\n',
        '    import med133_curvature as curvature\n'
        '    import fisher_slice, projected_direction as r2\n'
        '    from two_direction import build as build_r2\n')
    start = text.index('    # Same original optimizer and cache, versus the new factor-container dispatch.\n')
    end = text.index('    runtime = load_real_runtime(', start)
    text = text[:start] + "    assert json.loads((ROOT/'CPU_ADMISSION.json').read_text())['status']=='PASS'\n" + text[end:]
    text = replace_once(text, "for name in ('FULL_CTX','MED')}", "for name in ('FULL_CTX','CLEAN_MED1','CLEAN_MED2')}")
    old = """        legacy = r.p.load_state(PARENT/'private/OPERATOR_FISHER.pt')
        basis, _ = r.q.split()
        assert len(basis) == len(legacy['token_counts']) == 61
        xpieces, ypieces = list(legacy['x'].split(legacy['token_counts'])), list(legacy['y'].split(legacy['token_counts']))
        fallback = curvature.concatenate([dict(x=xpieces[v['index']],y=ypieces[v['index']])
                                          for v in inputs['unmapped']])
"""
    new = """        source_lock = inputs['source_identity_lock']
        assert source_lock['label']=='IDENTITY_LOCK_ONLY'
        fd = os.open(PARENT/'private/OPERATOR_FISHER.pt', os.O_RDONLY)
        try:
            parts = fisher_slice.extract_from_lock(fd, source_lock, (1514,14336,32,4096), include_mapped=True)
        finally:
            os.close(fd)
        xpieces = {v['index']:torch.from_numpy(v['x']) for v in parts}
        ypieces = {v['index']:torch.from_numpy(v['y']) for v in parts}
        assert len(parts)==59 and len(inputs['unmapped'])==29
        fallback = curvature.concatenate([dict(x=xpieces[v['index']],y=ypieces[v['index']])
                                          for v in inputs['unmapped']])
"""
    text = replace_once(text, old, new)
    text = replace_once(text, '        full_blocks, med_blocks, audits, actual_inputs = [], [], [], []\n',
                        '        full_blocks, med_blocks, r2_blocks, audits, actual_inputs = [], [], [], [], []\n')
    text = replace_once(text, '                assert row == basis[index]\n',
        "                allowed = [v for v in source_lock['keep'] if v['index']==index and v['kind']=='mapped']\n"
        "                assert len(allowed)==1 and row['query_id']==allowed[0]['query_id']\n")
    text = replace_once(text, '                    full_blocks.extend(full);med_blocks.append(med)\n',
        '                    full_blocks.extend(full);med_blocks.append(med)\n'
        '                    candidate,reason=r2.mapped_block(full,med,audit,curvature.gram,curvature.energy,build_r2)\n'
        '                    r2_blocks.append(candidate);audit["second_direction"]=reason\n')
    old = """            caches = {name:dict(parts=[fallback,curvature.concatenate(blocks)])
                      for name,blocks in [('FULL_CTX',full_blocks),('MED',med_blocks)]}
"""
    new = """            assert len(med_blocks)==len(r2_blocks)
            caches = {'FULL_CTX':dict(parts=[fallback,curvature.concatenate(full_blocks)]),
                      'CLEAN_MED1':dict(parts=[fallback,curvature.concatenate(med_blocks)]),
                      'CLEAN_MED2':dict(parts=[fallback,r2.pack_mapped(r2_blocks)])}
"""
    text = replace_once(text, old, new)
    text = replace_once(text, 'common_fallback=61-len(med_blocks)', 'common_fallback=59-len(med_blocks)')
    text = replace_once(text, 'del expert,legacy,xpieces,ypieces,full_blocks,med_blocks,paths',
                        'del expert,parts,xpieces,ypieces,full_blocks,med_blocks,r2_blocks,paths')
    text = replace_once(text, '''        if task['order'] in (47,117,93,87):
            prior=r.c.read(Path(os.environ['SMALL_RUN'])/'calibration'/f"{task['order']}.json")['coefficient']
            assert abs(coefficient-prior)<=1e-12+1e-4*abs(prior)
''', "        # The excluded sources change the shared denominator; compare arms, not old lambda.\n")
    text = replace_once(text, "    for task in inputs['tasks']:\n",
        "    assert LOCK['arms']==['CLEAN_MED1','CLEAN_MED2']\n"
        "    for task in [v for v in inputs['tasks'] if v['order'] in (47,117,93,87)]:\n")
    start = text.index('        final_path=ROOT/\'candidates\'/f"MED_{task[\'order\']}.pt"\n')
    end = text.index("        rows = inputs['source_rows'][str(task['order'])]\n", start)
    text = text[:start] + "        assert not recovering, 'automatic training recovery is not admitted'\n" + text[end:]
    text = replace_once(text, 'resume_path=ROOT/\'resume\'/f"MED_{task[\'order\']}.pt"',
                        'resume_path=ROOT/\'resume\'/f"{arm}_{task[\'order\']}.pt"')
    text = replace_once(text, "(ROOT/'training').glob('MED_*.json'))==23360\n    assert COUNTS['updates']>=23360",
        "(ROOT/'training').glob('CLEAN_MED*.json'))==1280\n    assert COUNTS['updates']==1280")
    text = replace_once(text, '    import med133_evaluate as evaluate\n', '    import med_clean_evaluate as evaluate\n')
    if "r.q.split()" in text or "r.p.load_state(PARENT/'private/OPERATOR_FISHER.pt')" in text:
        raise ValueError('unsafe legacy bulk reader remains')
    compile(text, 'clean-two-arm-worker', 'exec')
    return text


if __name__ == '__main__':
    import sys
    source, destination = map(Path, sys.argv[1:])
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(build(source.read_text()))
