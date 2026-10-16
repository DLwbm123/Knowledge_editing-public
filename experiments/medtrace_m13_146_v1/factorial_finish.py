"""Resume scoring/reporting only after the audited zero-Judge initialization failure."""
import time
import traceback
import factorial_controller as c

def main():
    r=c.RUN;c.f.admit(r)
    assert c.read(r/'private/FACTORIAL_SCORING_ADMISSION1.json')['status']=='PASS'
    assert c.read(r/'private/FACTORIAL_GENERATION_COMPLETE1.json')['consumers']==13917
    for part in (0,1):
        assert c.read(r/'private'/f'FACTORIAL_EXIT_m13_146_single_{part}.json')['exit_code']==0
    with (r/'private/FACTORIAL_FINISH_ONCE1').open('x'):pass
    c.finish()

if __name__=='__main__':
    code=0
    try:main()
    except BaseException as e:
        code=1;c.write(c.RUN/'private/FAILURE_factorial_finish1.json',dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time()));raise
    finally:
        c.merge('Judge')
        c.write(c.RUN/'private/FACTORIAL_FINISH_EXIT1.json',dict(exit_code=code,epoch=time.time()))
