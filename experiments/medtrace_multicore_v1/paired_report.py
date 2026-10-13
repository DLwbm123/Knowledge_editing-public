"""Reuse the frozen full report separately for each arm, then join aggregates."""
import os
from pathlib import Path
import json
import subprocess
import tempfile
RUN=Path(os.environ['RUN_ROOT'])


def main():
    source=(RUN/'private/tools/full_report.py').read_text()
    results={};semantics=[]
    for arm in ('ROOT_FIX','MULTI'):
        code=source.replace('REFINE',arm).replace('import full_queue as queue','import paired_queue as queue')
        code=code.replace('FROM consumer c JOIN payload p ON c.payload_key=p.key\'','FROM consumer c JOIN payload p ON c.payload_key=p.key WHERE c.method IN ("'+arm+'","NATURAL_'+arm+'")\'')
        code=code.replace('assert len(current) == queue.CONSUMERS','assert len(current) == 1158')
        # The shared report's defaults must use this experiment's isolated queue.
        code=code.replace("q.ROOT = RUN / 'private/judge_full_astra_medium'", "q.ROOT = RUN / 'private/judge_multicore_astra_medium'")
        fd,path=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
        Path(path).write_text(code)
        subprocess.run([os.environ['TRAIN_PYTHON'],path],check=True)
        for folder,names in [('public',['FULL_RESULTS.json','FULL_REPORT_ZH.md']),
                             ('private',['FULL_RESULTS_WITH_OWNER_AUDIT.json'])]:
            for name in names:(RUN/folder/name).rename(RUN/folder/(arm+'_'+name))
        results[arm]=json.loads((RUN/'public'/f'{arm}_FULL_RESULTS.json').read_text())
        semantics+=json.loads((RUN/'private/MODULE_SEMANTIC.json').read_text())
    assert len(semantics)==2316
    (RUN/'private/MODULE_SEMANTIC.json').write_text(json.dumps(semantics)+'\n')
    summary=dict(status='COMPLETE',arms={k:dict(decision=v['decision'],compatibility=v['compatibility'],
        missing_consumers=v['missing_consumers'],panels=v['panels'][k],natural=v['panels']['NATURAL_'+k]) for k,v in results.items()},
        same_frozen_states=True,same_budget=True,same_gain_and_step_rules=True,independent_confirmation=False,
        resource=results['MULTI']['resource'])
    summary['complete_no_missing']=all(v['missing_consumers']==0 for v in results.values())
    summary['scientific_success']=summary['complete_no_missing'] and all(results['MULTI']['compatibility'].values())
    import scope_report as prior
    rows=[dict(data=json.loads(Path(v['path']).read_text()),score=v['correct']) for v in semantics]
    pairs={}
    for mode in ('forced','natural'):
        mapped={}
        for arm in ('ROOT_FIX','MULTI'):
            method=arm if mode=='forced' else 'NATURAL_'+arm
            mapped[arm]={(v['data']['expert_order'] if mode=='forced' else 0,
                          v['data']['binding']['input']['query_id']):v for v in rows if v['data']['arm']==method}
        assert mapped['ROOT_FIX'].keys()==mapped['MULTI'].keys()
        roles=sorted({v['data']['role'] for v in mapped['ROOT_FIX'].values()})
        pairs[mode]={role:prior.paired([(v['score'],mapped['MULTI'][key]['score'])
            for key,v in mapped['ROOT_FIX'].items() if v['data']['role']==role]) for role in roles}
    summary['root_multi_pairs']=pairs
    (RUN/'public/PAIRED_FULL_RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert all(v['status']=='COMPLETE' for v in results.values())


if __name__=='__main__':main()
