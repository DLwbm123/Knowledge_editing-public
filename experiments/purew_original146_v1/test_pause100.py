"""CPU-only tests for planned-stop isolation and exact retained resume clocks."""
import json
import os
from pathlib import Path
import tempfile
import types
import torch
import pause100 as p


def check(owner,step):
    original_os,original_subprocess,original_alive=p.os,p.subprocess,p.alive
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);control=root/'private/pause_prefix100'
        control.mkdir(parents=True);(root/'private/failures').mkdir()
        auth=dict(prefix=100,user_authorized=True,protocol_binding='test_binding',GPU_UUID='test_uuid',gpu=dict(pid=11),controller=dict(pid=12))
        p.write(control/'AUTHORIZATION.json',auth);p.write(root/'RUN_MANIFEST.json',dict(deadline_epoch=10**12))
        marker=root/'private/outputs/bank_R0/0/100/COMPLETE.json'
        p.write(marker,dict(phase=dict(prefix=100,owner=0,protocol='test_binding'),outputs=1));p.write(marker.parent/'example.json',{})
        p.write(root/'private/units/bank_R0/100.json',dict(step=160));p.write(root/'RESOURCE_LEDGER.json',dict(gpu_sessions=[dict(ended_epoch=1)],gpu_seconds_used=10))
        event=dict(owner=owner,step=step);p.write(root/'private/steps/bank_R0'/str(owner)/(str(step)+'.json'),event)
        master=torch.tensor([[.25,.5],[.75,1.]],dtype=torch.float32)
        state=dict(mode='bank_R0',owner=owner,step=step,committed=(owner-1)*160+step,lock='test_binding',master=master,native=master.half(),event=event,rng={},numpy_rng=())
        active=root/'private/active/bank_R0.pt';active.parent.mkdir();torch.save(state,active)
        live=[True];signals=[]
        p.alive=lambda row:live[0]
        p.subprocess=types.SimpleNamespace(check_output=lambda *a,**k:'test_uuid\n')
        def kill(pid,signal):
            signals.append(pid);live[0]=False
            p.write(root/'private/failures/controller_12.json',dict(traceback='AssertionError: gpu failed; inspect saved state, never blindly restart\n'))
        p.os=types.SimpleNamespace(environ={'RUN_ROOT':str(root)},kill=kill,link=os.link)
        try:
            failure=root/'private/failures/controller_12.json'
            p.write(failure,dict(traceback='AssertionError: gpu failed; inspect saved state, never blindly restart\n'))
            assert not p.planned_controller_failure(failure,root);failure.unlink()
            p.main()
            receipt=json.loads((root/'private/PAUSED_USER_REQUEST.json').read_text())
            assert receipt['continuous_committed']==state['committed'] and receipt['extra_committed_after_prefix100']==(0 if owner==100 else 1)
            assert signals==[11] and active.exists() and not (root/'private/TRAINING_GENERATION_COMPLETE.json').exists()
            assert (control/'retained_stop_records/controller_12.json').exists()
            p.write(failure,dict(traceback='RuntimeError: unexpected failure\n'));assert not p.planned_controller_failure(failure,root)
            try:p.main()
            except AssertionError:pass
            else:raise AssertionError('Must refuse repeated signal')
        finally:p.os,p.subprocess,p.alive=original_os,original_subprocess,original_alive


check(100,160);check(101,1)
fields=['99','(python)','R']+['0']*18+['123']
class ExitRace:
    calls=0
    def __truediv__(self,name):return self
    def exists(self):return True
    def read_text(self):
        self.calls+=1
        return ' '.join(fields if self.calls==1 else fields[:2]+['Z']+fields[3:])
    def read_bytes(self):return b''
assert not p.alive(dict(pid=99,start_ticks='123',entry='entry.py'),proc_root=ExitRace())
print('PASS: planned failure isolation, durable target, exact resume clocks, retained state, no repeat signal or full completion')
