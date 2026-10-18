"""Regression: string environment path and terminal GPU receipts never relaunch GPU work."""
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
from four_gpu_check import function

def main():
    counts=dict(updates=236520,backwards=411724,operator_backwards=78840,generation_attempts=8868)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'private/queue').mkdir(parents=True)
        (root/'private/queue/ALL_WORKERS_COMPLETE.json').touch()
        calls=[];writes={}
        def read(path):
            assert isinstance(path,Path),'Environment paths must be normalized before read'
            if path.name=='plan.json':return dict(prior_counts={},diagnostic_counts={},abandoned_counts={})
            if path.name.startswith('EXIT_'):return dict(exit_code=0)
            if path.name.startswith('DELETION_'):return dict(status='DELETED')
            if path.name=='CURRICULUM_CPU_CHECK.json':return dict(status='PASS')
            return dict(status='COMPLETE',save_load_exact=True,curve=[None]*540,counts=counts if path.name=='COMPLETE_curriculum_train_C_0.json' else {})
        def stage(*args):raise AssertionError('Completed GPU stage was relaunched')
        def launch(file,*args):calls.append(file);return SimpleNamespace(wait=lambda:0)
        scope=dict(os=os,Path=Path,RUN=root,read=read,write=lambda p,d:writes.update({p.name:d}),stage=stage,time=SimpleNamespace(time=lambda:0),shared=SimpleNamespace(merge=lambda _:None,launch=launch),f=SimpleNamespace(admit=lambda _:None,CAPS=counts,METHODS=('C','M1','M13P'),CONSUMERS=0,QUEUE='queue'))
        run=function('curriculum_controller.py','main',scope)
        with patch.dict(os.environ,CURRICULUM_ADOPT_TRAINING='1',CURRICULUM_RECOVERY_PLAN=str(root/'plan.json'),CURRICULUM_FINISH_ONLY='1'):run()
        assert calls==['curriculum_report.py']
        assert writes['CURRICULUM_GENERATION_COMPLETE.json']['counts']==counts
        assert 'RECOVERY_COST_AUDIT.json' in writes and 'CURRICULUM_COMPLETE.json' in writes
    print('PASS: environment path normalized; completed GPU stages skipped; aggregation/report reached')
if __name__=='__main__':main()
