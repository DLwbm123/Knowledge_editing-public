"""Reuse the selected training recipes; change only T5 admission/count plumbing."""
import importlib.util
from pathlib import Path


def build(worktree):
    w=Path(worktree)
    spec=importlib.util.spec_from_file_location('fixed_task_sources',w/'experiments/med_t4_core_v1/build_sources.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sources=module.build(w)
    for name,source in list(sources.items()):
        source=source.replace('<= 137','<= 322').replace('<=137','<=322').replace('T4_SOURCE_ADMITTED','T5_SOURCE_ADMITTED').replace('T4_C_MED_MDU_SHARD_COMPLETE','T5_C_MED_MDU_SHARD_COMPLETE')
        if name=='initialize.py':
            before="assert task['edit_id']==want['query_id']"
            assert source.count(before)==1
            source=source.replace(before,"assert task['native']['query_id']==want['query_id']")
        sources[name]=source
    sources['pipeline_guard.py']=(w/'experiments/med_t4_core_v1/pipeline_guard.py').read_text().replace('T4','T5')
    bank=(w/'experiments/med_t4_core_v1/bank.py').read_text().replace('T4','T5')
    bank=module.replace_once(bank,"ordered=[routers[t['edit_id']] for t in tasks]",
        "ordered=[routers[t['edit_id']] for t in tasks if t['sequential_role_safe']]")
    bank=module.replace_once(bank,"        for row in data['plan']:\n",
        "        for row in data['plan']:\n            if mode=='bank' and not byid[row['edit_id']]['sequential_role_safe']:continue\n")
    bank=bank.replace('2*len(data[\'plan\'])',"data['expected_consumers']").replace("COUNTS['generations']<=3000","COUNTS['generations']<=4000")
    marker='    features,cache,route_cache={},{},{}'
    bank=module.replace_once(bank,marker,marker+'''
    for filename in (ROOT/'base_outputs').glob('*.json'):
        value=json.loads(filename.read_text())
        key=r.c.digest(dict(input=value['judge_input'],effective_state='UNEDITED_BASE'))
        cache[key]=value['answer']
''')
    sources['bank.py']=bank
    for name in ('preprocess.py','base.py','base_guard.py'):
        sources[name]=(w/'experiments/med_t5_core_v1'/name).read_text()
    sources['stream_extract.py']=(w/'experiments/med150_source_review_v1/stream_extract.py').read_text()
    sources['projection.py']=(w/'experiments/med_t4_core_v1/projection.py').read_text()
    for name,source in sources.items():compile(source,name,'exec')
    return sources
