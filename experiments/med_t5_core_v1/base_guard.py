"""Guard the two-shard Base panel before any qualification or editing."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import pipeline_guard as guard

ROOT=Path(os.environ['T5_REMOTE_ROOT'])


def main():
    with (ROOT/'BASE_CHAIN_ONCE').open('x'):pass
    cfg=guard.read('BASE_PIPELINE_CONFIG.private.json');clock=guard.read('CLOCK.json')
    assert guard.read('SOURCE_ADMISSION.json')['status']=='PASS'
    for gpu in cfg['gpus']:
        row=subprocess.check_output(['nvidia-smi','-i',gpu['index'],'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        assert row[0].strip()==gpu['uuid'] and int(row[1])>=45056
    guard.run_phase('BASE',cfg['workers'],clock)
    completed=[guard.read(f'BASE_COMPLETE_{i}.json') for i in range(len(cfg['workers']))]
    expected=len(guard.read('SOURCE_INPUTS.private.json')['base_plan'])
    assert sum(c['counts']['consumers'] for c in completed)==expected
    assert len(list((ROOT/'base_outputs').glob('*.json')))==expected
    assert sum(c['counts']['generations']+c['counts']['diagnostic_generations'] for c in completed)<=646
    guard.write('BASE_GENERATION_COMPLETE.json',dict(status='ALL_BASE_OUTPUTS_FROZEN',time=time.time(),
        consumers=expected,counts=[c['counts'] for c in completed],actual_precision='float32'))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;guard.write('BASE_CHAIN_FAILURE.json',dict(time=time.time(),error=traceback.format_exc()));raise
    finally:guard.write('BASE_CHAIN_EXIT.json',dict(code=code,time=time.time()))
