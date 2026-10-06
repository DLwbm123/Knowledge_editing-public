"""User-authorized parallel completion of existing frozen bank inference only."""
import json
import os
from pathlib import Path
import signal
import time
import traceback
import common

RUN=common.RUN
original_read=common.read

def amended_read(path):
    result=original_read(path)
    if Path(path)==RUN/'PLAN_CONFIG.json':
        result=dict(result,hardware=original_read(RUN/'private/ACCELERATION_AUTHORIZATION.json')['hardware'])
    return result

common.read=amended_read
import sys
sys.path.insert(0,str(RUN/'private/tools'))
from common import read,write,lease,load,budget,digest
import worker
from train import teachers_for


def main():
    gpu=int(os.environ['GPU']);part=int(os.environ['PARTITION']);n=8
    with lease(gpu):
        runtime,bindings=load(gpu)
        ts=worker.tasks('P2');assert len(ts)==24
        bank=[];points={};arm='CE_U_MULTI';node=read(RUN/'private/T_STAR.json')['t_star']
        for t in ts:
            bank+=worker.router(runtime,t)
            points[t['edit_id']]=Path(read(RUN/'private/edits'/t['anonymous_edit']/'s0/COMPLETE.json')['points'][arm])
        check=teachers_for(runtime,read(RUN/'private/U_ROLES.json')['CHECK'])
        # One distinct owner writes the union R0 panel; owner partitions write
        # disjoint forced panels. The original frozen evaluator is unchanged.
        if part==6:
            worker.evaluate(runtime,bindings,ts[-1],0,arm,node,points,bank,ts,mode='bank',prefix=24,teacher_rows=check)
        for t in ts[part::n]:
            worker.evaluate(runtime,bindings,t,0,arm,node,points,bank,[t],mode='bank',prefix=24,forced=True,teacher_rows=check)
        write(RUN/'private'/('ACCELERATION_PART_'+str(part)+'_DONE.json'),dict(status='GENERATED_NOT_SCORED',GPU=gpu,part=part,owners=[t['anonymous_edit'] for t in ts[part::n]],frozen_evaluator=True,epoch=time.time()))

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Bounded inference interrupted; completed outputs retained')))
    try:main()
    except BaseException as e:
        write(RUN/'private'/('ACCELERATION_FAILURE_'+os.environ['PARTITION']+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time()));raise
