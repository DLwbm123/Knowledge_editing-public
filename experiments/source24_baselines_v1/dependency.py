"""One finite CPU dependency and candidate consumer; no recurring monitor."""
import os,sys,time,json,shutil,traceback
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=Path('/data/bmw/Knowledge_editing/outputs/purew-source-v5-20261011/run')
sys.path.insert(0,str(R/'private/tools'));import journal as j
def read(p):return json.loads(p.read_text())
def state(status,**kwargs):j.write(R/'private/DEPENDENCY_STATE.json',dict(status=status,epoch=time.time(),**kwargs))
def main():
    with j.exclusive(R/'private/locks/dependency.lock'):
        state('WAITING_FOR_PARENT_GPU_CONSUMERS')
        while True:
            assert time.time()<read(R/'RUN_MANIFEST.json')['deadline_epoch'],'Frozen new deadline reached'
            assert not list((P/'private/failures').glob('*.json')),'Parent stopped; preserve original failure'
            for arm in ('U','H_SOURCE'):
                dest=R/'private/parent_candidates'/f'{arm}.pt';receipt=R/'private'/f'CAPTURE_{arm}.json'
                if receipt.exists():assert dest.is_file();continue
                source=P/'private/candidates'/f'{arm}.pt';complete=P/'private'/f'TRAIN_COMPLETE_{arm}.json'
                if complete.exists():
                    assert read(complete)['lock']==read(R/'private/PARENT_BINDING.json')['scientific_lock']
                    dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_suffix('.tmp');shutil.copy2(source,temp);temp.replace(dest)
                    assert dest.stat().st_size==source.stat().st_size
                    j.write(receipt,dict(bytes=dest.stat().st_size,parent_receipt=read(complete),consumer='fixed end-to-end performance and native reload',epoch=time.time()))
            if (P/'private/GENERATION_COMPLETE.json').exists() and all(x.get('ended_epoch') for x in read(P/'RESOURCE_LEDGER.json')['gpu_sessions']):break
            time.sleep(10)
        assert all((R/'private/parent_candidates'/f'{a}.pt').is_file() for a in ('U','H_SOURCE'))
        state('PARENT_GPU_COMPLETE_ADMISSION_PENDING')
        import pipeline
        # The admission worker performs the UUID and >=60000MiB nonexclusive check.
        child=pipeline.launch('bench.py','baseline_controller',5,0)
        start=read(R/'private/START_CHAIN_baseline_controller_0.json');assert j.alive(start)
        state('BASELINE_GPU_CHAIN_RUNNING',start=start)
        assert child.wait()==0 and (R/'private/GENERATION_COMPLETE.json').is_file()
        state('GPU_COMPLETE_SCORING_AND_PUBLICATION_PENDING')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        state('STOPPED_WITH_RETAINED_EVIDENCE',error=repr(e),traceback=traceback.format_exc());j.write(R/'private/failures'/f'dependency_{time.time_ns()}.json',dict(error=repr(e)));raise
