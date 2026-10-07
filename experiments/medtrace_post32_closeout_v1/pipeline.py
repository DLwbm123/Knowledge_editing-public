"""Finite dependency chain for own workers; no monitoring/recurring task."""
import os,sys,time,json,subprocess,traceback,signal
from pathlib import Path
RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
import core
common=core.common

def wait_existing():
    starts=[common.read(RUN/'private'/('CORE_REPAIRED2_START_'+str(i)+'.json')) for i in [0,1]]
    while not all((RUN/'private'/('TRAIN_PART_'+str(i)+'.json')).exists() for i in [0,1]):
        common.budget()
        for i,x in enumerate(starts):
            stat=Path('/proc/'+str(x['pid'])+'/stat')
            if not (RUN/'private'/('TRAIN_PART_'+str(i)+'.json')).exists():assert stat.exists() and stat.read_text().split()[21]==x['start_ticks'] and stat.read_text().split()[2]!='Z','Own core worker stopped; retain evidence, no blind retry'
        time.sleep(5)

def launch(file,action,gpu=None,part=None,python=None):
    import tempfile
    fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd);Path(entry).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
    env=dict(os.environ,TASK_FILE=str(RUN/'private/tools'/file),ACTION=action)
    if gpu is not None:env.update(GPU=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu))
    if part is not None:env['PARTITION']=str(part)
    log=RUN/'logs'/('chain_'+action+'_'+str(part)+'.log');p=subprocess.Popen([python or os.environ['TRAIN_PYTHON'],'-u',entry],env=env,stdout=log.open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    common.write(RUN/'private'/('START_CHAIN_'+action+'_'+str(part)+'.json'),dict(pid=p.pid,start_ticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],entry=entry,log=str(log),epoch=time.time()));return p

def wait(jobs):
    while any(p.poll() is None for p in jobs):
        try:common.budget()
        except BaseException:
            for p in jobs:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            raise
        if any(p.poll() is not None and p.returncode!=0 for p in jobs):
            for p in jobs:
                if p.poll() is None:p.wait()
            raise RuntimeError('Own finite stage failed; do not automatically retry')
        time.sleep(5)
    assert all(p.returncode==0 for p in jobs)

def score():
    common.write(RUN/'private/GENERATION_COMPLETE.json',dict(status='PHASE_GENERATED',epoch=time.time()))
    wait([launch('qwen_scorer.py','score',python=os.environ['QWEN_PYTHON'])])

def main():
    wait_existing();common.write(RUN/'public/PROGRESS.json',dict(status='CORE_TRAINING_COMPLETE_BANK_EVAL_RUNNING',whole_task_complete=False))
    wait([launch('evaluate.py','core_eval',5,0),launch('evaluate.py','core_eval',6,1)]);score();wait([launch('semantic.py','semantic_prepare')]);wait([launch('semantic.py','A1_JUDGE',7,0,python=os.environ['QWEN_PYTHON'])]);wait([launch('results.py','core_report')])
    assert common.used()<=8*3600
    common.write(RUN/'STAGE_CAP.json',dict(stage='FINAL_COMPARISON',GPU_seconds_limit=24*3600,development_used_seconds=common.used(),minimum_final_reserve=16*3600))
    common.write(RUN/'public/PROGRESS.json',dict(status='CORE_EVALUATION_COMPLETE_FINAL_STRUCTURAL_COMPARISON_RUNNING',whole_task_complete=False))
    wait([launch('lora.py','lora_train',5,0),launch('lora.py','lora_train',6,1)]);wait([launch('lora.py','lora_eval',7,0)]);score();wait([launch('final_report.py','final_report')])
    common.write(RUN/'public/PROGRESS.json',dict(status='BOUNDED_RUN_ENDED_REPORT_REVIEW_PUBLICATION_PENDING',training_complete=True,evaluation_complete=True,development_candidate_locked=True,independent_confirmation_complete=False,benchmark146_complete=False,publication_complete=False,whole_task_complete=False,no_next_round=True))
    common.write(RUN/'private/CONTROLLER_COMPLETE.json',dict(epoch=time.time(),all_executed_GPU_sessions_ended=all(x.get('ended_epoch') for x in common.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions']),scientific_and_publication_review_pending=True))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        common.write(RUN/'private/PIPELINE_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time(),retry=False));common.write(RUN/'public/PROGRESS.json',dict(status='STOPPED_WITH_RETAINED_EVIDENCE',training_complete=all((RUN/'private'/('TRAIN_PART_'+str(i)+'.json')).exists() for i in [0,1]),whole_task_complete=False));raise
