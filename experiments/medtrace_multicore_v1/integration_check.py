"""Temporary CPU fixture: original bindings, fabricated answers/scores, no API."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    run=Path(os.environ['RUN_ROOT']);parent=Path(os.environ['REFINE_PARENT'])
    fixture=Path(tempfile.mkdtemp(prefix='CPU_FIXTURE_ONLY_',dir=run/'private'))
    read=lambda p:json.loads(p.read_text())
    def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v)+'\n')
    def execute(entry,env):
        result=subprocess.run([os.environ['TRAIN_PYTHON'],entry],env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stderr[-5000:]
    try:
        (fixture/'private').mkdir()
        (fixture/'private/tools').symlink_to(run/'private/tools',target_is_directory=True)
        (fixture/'private/source').symlink_to(run/'private/source',target_is_directory=True)
        for name in ['LAUNCH_ENV.json','SCOPE_LOCK.json','HELD_BASE.json','REFERENCE_EXACT.json','INHERITED_COST.json']:
            shutil.copy2(run/'private'/name,fixture/'private'/name)
        for name in ['RUN_MANIFEST.json','RESOURCE_LEDGER.json']:
            shutil.copy2(run/name,fixture/name)
        ledger=read(fixture/'RESOURCE_LEDGER.json')
        ledger['gpu_sessions']=[dict(CPU_FIXTURE_ONLY=True,started_epoch=1.,ended_epoch=1.,resident_seconds=0.) for _ in range(11)]
        write(fixture/'RESOURCE_LEDGER.json',ledger)
        (fixture/'public').mkdir()
        shutil.copy2(parent/'public/FISHER_ADMISSION.json',fixture/'public/FISHER_ADMISSION.json')
        (fixture/'private/results/C').mkdir(parents=True)
        for i in range(1,9):
            (fixture/f'private/results/C/{i}.json').symlink_to((run/f'private/results/C/{i}.json').resolve())
        for arm in ['ROOT_FIX','MULTI']:
            for i in range(1,9):
                value=read(parent/f'private/results/REFINE/{i}.json');value['arm']=arm
                write(fixture/f'private/results/{arm}/{i}.json',value)
            for p in (parent/'private/outputs').glob('*/*/*.json'):
                data=read(p);relative=p.relative_to(parent/'private/outputs')
                method=arm if data['arm']=='REFINE' else 'NATURAL_'+arm
                data.update(arm=method,CPU_FIXTURE_ONLY=True)
                data.pop('alias_of',None)
                data['R0']['raw_answer']='CPU_FIXTURE_ONLY_'+method+'_'+str(relative)
                dest=fixture/'private/outputs'/method/relative.parts[1]/relative.parts[2]
                write(dest,data)
        assert len(list((fixture/'private/outputs').glob('*/*/*.json')))==2316
        write(fixture/'private/MODULE_GENERATION_COMPLETE.json',dict(CPU_FIXTURE_ONLY=True,
              counts=[dict(updates=0,backwards=0,operator_backwards=0,local_solves=0,non_generation_forwards=0,generation_attempts=0,forwards=0)]))
        fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
        Path(entry).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
        env=dict(os.environ,RUN_ROOT=str(fixture),CUDA_VISIBLE_DEVICES='',TASK_FILE=str(run/'private/tools/paired_queue.py'))
        execute(entry,env)
        import sqlite3
        queue=fixture/'private/judge_multicore_astra_medium'
        db=sqlite3.connect(queue/'queue.sqlite')
        assert db.execute('select count(*) from consumer').fetchone()[0]==2316
        db.execute("update payload set status='FORMAT_VALID',correct=1");db.commit()
        write(queue/'ALL_WORKERS_COMPLETE.json',dict(CPU_FIXTURE_ONLY=True))
        env['TASK_FILE']=str(run/'private/tools/paired_report.py')
        execute(entry,env)
        assert read(fixture/'public/PAIRED_FULL_RESULTS.json')['scientific_success']
        key=db.execute("select payload_key from consumer where method='MULTI' and folder='NATIVE' limit 1").fetchone()[0]
        db.execute('update payload set correct=0 where key=?',(key,));db.commit()
        execute(entry,env)
        assert not read(fixture/'public/PAIRED_FULL_RESULTS.json')['scientific_success']
        db.execute("update payload set status='MISSING',correct=NULL where key=?",(key,));db.commit()
        execute(entry,env)
        result=read(fixture/'public/PAIRED_FULL_RESULTS.json')
        assert not result['complete_no_missing'] and not result['scientific_success']
        db.close()
    finally:
        assert fixture.parent==run/'private' and fixture.name.startswith('CPU_FIXTURE_ONLY_')
        shutil.rmtree(fixture)
    print(json.dumps(dict(status='PASS',synthetic_fixture_only=True,consumers=2316,
        two_arm_queue_and_report=True,regression_and_missing_rejected=True,new_model_calls=0,new_Judge=0)))


if __name__=='__main__':main()
