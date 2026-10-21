"""Use the established isolated scorer without its experiment wall timer."""
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(os.environ['SCORER_BASE']).parent))
import scorer as base

def call(bundle,batch,work,sibling):
    return base.judge.run_batch(bundle,batch,work,[work,sibling],base.SOURCE,base.CLI,explicit_proxy=True,reasoning_effort='medium')
base.call=call
try:
    base.main()
except Exception as e:
    import json,time,traceback
    data=dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time(),new_requests_stopped=True)
    base.LOCAL.mkdir(parents=True,exist_ok=True)
    (base.LOCAL/'SCORER_FAILURE.json').write_text(json.dumps(data,indent=2))
    try:base.publish_file('SCORER_FAILURE.json',json.dumps(data).encode())
    except Exception:pass
    raise
