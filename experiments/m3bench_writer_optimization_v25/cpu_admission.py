"""Synthetic CPU admission only; inherits the frozen parent dependencies."""
import json,os,signal,sys,time,traceback
from pathlib import Path
ROOT=Path(os.environ['V25_ROOT']);P=Path(os.environ['PARENT_RUN'])
sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
def write(name,value):
    p=ROOT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
if __name__=='__main__':
    try:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU admission alarm')));signal.alarm(180)
        with (ROOT/'CPU_ONCE').open('x'):pass
        import torch
        torch.set_num_threads(1)
        import module_run as r
        from alignment import selfcheck
        from consumers import selfcheck as consumer_check
        seed=json.loads((P/'private/M13_SOURCE_PACKET.json').read_text())['original_tasks'][4]['seed']
        result=selfcheck(r,seed);result['consumers']=consumer_check();
        from diagnostics import selfcheck as diagnostics_check
        result['diagnostics']=diagnostics_check(r,seed);result['time']=time.time()
        write('CPU_ADMISSION.json',result)
    except BaseException:
        write('CPU_FAILURE.json',dict(error=traceback.format_exc(),time=time.time(),automatic_retry=False));raise
