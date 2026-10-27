"""Finite diagnostic controller using the existing independent resource guard."""
import json
import os
from pathlib import Path
import time
import traceback
import v81_control as c

if __name__=='__main__':
    try:
        if os.environ['CONTROL_ACTION']=='guard':c.guard()
        else:
            with (c.ROOT/'CONTROLLER_ONCE').open('x'):pass
            c.stage('teacher')
            assert len(list((c.ROOT/'diagnosis').glob('*/*/*.json')))==280
            c.write(c.ROOT/'MAIN_COMPLETE.json',dict(status='DIAGNOSIS_COMPLETE_REVIEW_PENDING',records=280,generations=56,time=time.time()))
    except BaseException as e:
        c.write(c.ROOT/f'FAILURE_{os.environ["CONTROL_ACTION"]}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
    finally:
        if os.environ['CONTROL_ACTION']=='controller':
            c.write(c.ROOT/'RESOURCE_SCOPE_CLOSED.json',dict(time=time.time()))
