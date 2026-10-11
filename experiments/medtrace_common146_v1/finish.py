"""Finite report tail: wait for the admitted scorers, then run CPU reporting once."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

GROUP=Path(os.environ['SCORER_GROUP'])


def alive(row):
    process=subprocess.run(['ps','-p',str(row['pid']),'-o','args='],capture_output=True,text=True)
    return row['entry'] in process.stdout


def main():
    rows=json.loads((GROUP/'START.json').read_text())
    assert len(rows)==4 and len({r['pid'] for r in rows})==4
    assert all(all(r['isolation_checks'].values()) for r in rows)
    while any(alive(row) for row in rows):time.sleep(30)
    config=json.loads((GROUP/'0/config.json').read_text())
    code='import os,sys,runpy\nsys.dont_write_bytecode=True\nos.nice(10)\nos.environ["RUN_ROOT"]='+repr(config['RUN_ROOT'])+'\nrunpy.run_path(os.environ["RUN_ROOT"]+"/private/tools/common146_report.py",run_name="__main__")\n'
    result=subprocess.run(['ssh','pro5000','python3 -'],input=code,text=True,capture_output=True)
    (GROUP/'REPORT_OUTPUT.txt').write_text(result.stdout+result.stderr)
    assert result.returncode==0, 'CPU report did not finish; inspect retained output, do not restart scorers'
    (GROUP/'REPORT_DONE.json').write_text(json.dumps(dict(epoch=time.time(),status='REPORTED_PUBLICATION_PENDING')))


if __name__=='__main__':
    try:main()
    except Exception:
        (GROUP/'FINISH_FAILURE.json').write_text(json.dumps(dict(epoch=time.time(),traceback=traceback.format_exc())))
        raise
