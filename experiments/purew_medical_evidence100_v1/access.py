"""A current-case Python I/O boundary; not an adversarial OS/privacy sandbox."""
import os
from pathlib import Path
import sys


def install(run,window,source_image_root):
    run=Path(run).resolve();window=Path(window).resolve();image_root=Path(source_image_root).resolve()
    denied=(run/'private/operator',run/'private/evaluation',run/'private/outputs',run/'private/judge_purew146')
    forbidden_names={'BENCHMARK146_QUEUE.json','EVAL_LEDGER.json','EVAL_BINDINGS.json','SOURCE_PACKET_PRIVATE.json',
                     'SOURCE_REVIEW_PRIVATE.json','BASIS_BUFFER.json','C_REFERENCE.json'}
    old_root=run.parent.parent
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_dir():return
        if window in p.parents or p==window:return
        code_root=(run/'private/source').resolve()
        if code_root in p.parents and p.suffix in ('.py','.pyc','.so'):return
        blocked=(p.name in forbidden_names or any(p==x or x in p.parents for x in denied)
                 or p==image_root or image_root in p.parents
                 or (old_root in p.parents and run not in p.parents))
        if blocked:raise PermissionError('Current-case learner boundary rejects historical/evaluation access')
    sys.addaudithook(audit)
    return audit


def selfcheck():
    # Run this in a disposable subprocess; audit hooks cannot be removed.
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        r=Path(t)/'run';w=r/'private/window';w.mkdir(parents=True)
        (w/'current.json').write_text('{}')
        p=r/'private/EVAL_LEDGER.json';p.write_text('{}')
        install(r,w,Path(t)/'images')
        assert (w/'current.json').read_text()=='{}'
        try:p.read_text()
        except PermissionError:pass
        else:raise AssertionError('Evaluation read allowed')
        assert not (w/'old.pt').exists()
    print('{"status":"PASS","current_case_read":true,"evaluation_read_denied":true,"OS_security_guarantee":false}')


if __name__=='__main__':selfcheck()
