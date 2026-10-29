"""Prepare a four-owner evaluator; source bindings are checked at execution."""
from pathlib import Path
from build_worker import replace_once


def build(source):
    text = replace_once(source, 'import json\n', 'import json\nimport os\n')
    text = replace_once(text,
        '    states={t[\'edit_id\']:r.p.load_state(root/\'candidates\'/f"MED_{t[\'order\']}.pt") for t in tasks}\n',
        '    states={t[\'edit_id\']:r.p.load_state(root/\'candidates\'/f"MED_{t[\'order\']}.pt") for t in tasks if t[\'order\'] in (47,117,93,87)}\n')
    text = replace_once(text, '    assert len(states)==146\n', '    assert len(states)==4\n')
    text = replace_once(text, "    package=root/'retained/MED146.pt'\n", "    package=root/'retained/MED_R2_DEV4.pt'\n")
    text = replace_once(text,
        "calibration={str(t['order']):r.c.read(root/'calibration'/f\"{t['order']}.json\") for t in tasks}),package.with_suffix('.tmp'))\n",
        "calibration={str(t['order']):r.c.read(root/'calibration'/f\"{t['order']}.json\") for t in tasks if t['order'] in (47,117,93,87)}),package.with_suffix('.tmp'))\n")
    text = replace_once(text,
        "    assert check['state_hashes']==hashes and len(check['states'])==len(check['R0_entries'])==146\n",
        "    assert check['state_hashes']==hashes and len(check['states'])==4 and len(check['R0_entries'])==146\n")
    text = replace_once(text, "dict(experts=146,states=hashes", "dict(experts=4,states=hashes")
    text = replace_once(text, "    rngpath=root/'GENERATION_RNG.pt'\n", "    rngpath=Path(os.environ['REFERENCE_MED_RUN'])/'GENERATION_RNG.pt'\n    assert rngpath.is_file()\n")
    text = replace_once(text, "    for stage in ('A','B'):\n", "    for stage in ('A',):\n")
    text = replace_once(text,
        "        rows=[row for row in inputs['plan'] if (row['mode']=='single')==(stage=='A')]\n",
        "        rows=[row for row in inputs['plan'] if row['mode']=='single' and row['owner'] in (47,117,93,87)]\n"
        "        assert len(rows)==lock['single_consumers']\n")
    needle="                item=r.s.prepare(runtime,bindings,task,[query])[0]\n"
    text = replace_once(text, needle, needle +
        "                    import contract\n"
        "                    cid=contract.consumer_id(row)\n"
        "                    prior=r.c.read(Path(os.environ['REFERENCE_MED_RUN'])/'outputs'/(r.c.digest(cid)+'.json'))\n"
        "                    old=prior['binding']['judge_input']; new=item[4]\n"
        "                    assert all(old[k]==new[k] for k in ('question','reference','image_sha256','prompt_ids','attention_mask','runtime','generation'))\n"
        "                    assert prior['effective_expert']==selected and prior['route']==decision_as_json(decision)\n")
    text = replace_once(text, "    assert len(list((root/'outputs').glob('*.json')))==7494\n",
                        "    assert len(list((root/'outputs').glob('*.json')))==lock['single_consumers']\n")
    text = replace_once(text, "consumers=7494,counts=counts,\n        single=2956,sequential=4538,",
                        "consumers=lock['single_consumers'],counts=counts,\n        single=lock['single_consumers'],sequential=0,")
    compile(text, 'candidate-evaluator', 'exec')
    return text


if __name__ == '__main__':
    import sys
    source, destination = map(Path, sys.argv[1:])
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(build(source.read_text()))
