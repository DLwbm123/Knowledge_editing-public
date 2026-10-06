"""Finite inference shards, original scoring/report/audit, original clock and caps."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback
from common import RUN,read,write,budget,used,digest
import sys
sys.path.insert(0,str(RUN/'private/tools'))
import controller


def main():
    with (RUN/'private/CONTROLLER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        auth=read(RUN/'private/ACCELERATION_AUTHORIZATION.json');jobs=[]
        for part,gpu in enumerate(auth['GPUs']):
            budget();assert not (RUN/'private'/('ACCELERATION_PART_'+str(part)+'_DONE.json')).exists()
            uuid=auth['hardware']['UUIDs'][str(gpu)]
            jobs.append(controller.launch('accelerate.py',dict(ACTION='bank_accelerated',GPU=str(gpu),PARTITION=str(part),CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuid),'accelerated_'+str(part)+'.log'))
        write(RUN/'private/ACCELERATION_JOBS.json',[s for p,s in jobs])
        controller.wait(jobs)
        assert all((RUN/'private'/('ACCELERATION_PART_'+str(p)+'_DONE.json')).exists() for p in range(8))
        # All original bank outputs, not single-output concatenation, must exist.
        counts={}
        for arm in ('FROZEN_W0','CE_ONLY','CE_U_SINGLE','CE_U_MULTI'):
            paths=list((RUN/'private/outputs/bank').glob('*/*/'+arm+'/*/*/*/*.json'))
            assert len(paths)==1400,(arm,len(paths));counts[arm]=len(paths)
        write(RUN/'private/BANK_CE_U_MULTI_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',experts=24,prefixes=[8,16,24],real_incremental_insertion=True,parallel_remaining_evaluation=True))
        write(RUN/'private/ACCELERATION_COMPLETE.json',dict(status='GENERATION_COMPLETE',counts=counts,epoch=time.time(),budgets_reset=False,training_repeated=False,existing_outputs_preserved=True))
        controller.scoring('P2');controller.cpu('report.py');controller.cpu('final_audit.py')
        write(RUN/'private/CONTROLLER_COMPLETE.json',dict(status='RESULTS_AND_AUDIT_COMPLETE_SCIENTIFIC_REVIEW_PUBLICATION_PENDING',epoch=time.time(),parallel_inference_amendment=True,no_next_round=True,no_monitor=True))
        write(RUN/'public/PROGRESS.json',dict(status='RESULTS_AND_AUDIT_COMPLETE_PUBLICATION_PENDING',GPU_hours=used()/3600))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(RUN/'private/ACCELERATION_CONTROLLER_FAILURE.json',dict(status='RETAINED_STATE',error=repr(e),traceback=traceback.format_exc(),epoch=time.time(),automatic_retry=False));raise
