"""A single sealed check after the complete independent review, no repair feedback."""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).parent))
import collision_screen as c
c.S=c.W/'.local_run/revision2'
c.REPORT_DIR=c.W/'reports/purew_cpol146_v1_20261013/revision2'

def main():
    final=c.S/'QUALIFICATION_FINAL.json'
    if final.exists():
        print(final.read_text());return
    source=json.loads((c.S/'SOURCE_QUALIFICATION.json').read_text())
    assert source['candidates']==584 and source['model_calls_attempted']==16
    # Exclusive marker: an uncertain prior check is never silently repeated.
    with (c.S/'COLLISION_CHECK_STARTED.json').open('x') as f:
        json.dump(dict(epoch=time.time(),candidate_binding=source['candidate_binding'],attempt=1),f)
    c.main()
if __name__=='__main__':main()
