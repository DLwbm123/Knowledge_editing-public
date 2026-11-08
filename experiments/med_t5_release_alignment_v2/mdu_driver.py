"""Run admitted new experts and the actual expanded bank under nested guards."""
import json
import os
from pathlib import Path
import runpy
import traceback

r=Path(os.environ['COMPARE_ROOT'])/'MDU'


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))


try:
    runpy.run_path(str(r/'tools/pipeline_guard.py'),run_name='__main__')
    assert json.loads((r/'GUARD_EXIT.json').read_text())['code']==0
    v=json.loads((r/'EVAL_GENERATION_COMPLETE.json').read_text())
    assert v['expected_consumers']==1908
    write(r/'T5/GENERATION_COMPLETE.json',dict(status='COMPLETE',single_edits=159,bank_edits=159,
        consumers=1908,new_experts=11,reused_experts=148,counts=v['counts']))
except BaseException:
    write(r/'FAILURE.json',dict(error=traceback.format_exc()));raise
