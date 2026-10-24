"""Linux CPU-only fault injection for the real supervisor; no model or GPU calls."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace


def test(mode):
    with tempfile.TemporaryDirectory(dir=os.environ.get('TEST_PARENT')) as tmp:
        root=Path(tmp)
        cfg=dict(original_start=0,hard_end=100,gpu_seconds=3600,storage_bytes=1_000_000_000,
                 owned_roots=[tmp],gpu=0,gpu_uuid='test',python='unused',neutral_entry='unused')
        (root/'CONFIG.json').write_text(json.dumps(cfg))
        (root/'DECISION_D.json').write_text(json.dumps(dict(selected=None,all_64_image_outputs_reviewed=True)))
        if mode=='existing_log':
            (root/'O_D.log').write_text('preserve')
        os.environ['TRIAL_ROOT']=tmp
        os.environ.pop('ADOPT_DOWNLOAD',None)
        os.environ.pop('ADOPT_MANIFEST',None)
        os.environ.pop('RECOVER_PREGEN',None)
        if mode=='recovery_normal':
            os.environ['RECOVER_PREGEN']='1'
            (root/'RECOVERY_PREGEN.json').write_text(json.dumps(dict(original_start=0,hard_end=100,gpu_process_seconds=20)))
            (root/'pregen_incident').mkdir()
            (root/'pregen_incident/GUARD_EXIT.json').write_text(json.dumps(dict(gpu_process_seconds=20)))
            (root/'DOWNLOAD_COMPLETE.json').write_text('{}')
        adoption=mode.startswith('adopt_')
        if adoption:
            os.environ['ADOPT_DOWNLOAD']='1'
            ticks=Path('/proc/self/stat').read_text().split()[21]
            (root/'ADOPT_DOWNLOAD.json').write_text(json.dumps(dict(original_start=0,hard_end=100,
                gpu_process_seconds=0,download_pid=os.getpid(),download_start_ticks=ticks if mode=='adopt_timeout' else '0',download_start=0)))
        spec=importlib.util.spec_from_file_location('trial_under_test',Path(__file__).with_name('run.py'))
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        clock=SimpleNamespace(now=99 if adoption else 0)
        def sleep(seconds):
            clock.now+=seconds
        module.time=SimpleNamespace(time=lambda:clock.now,sleep=sleep)
        launches=[]
        signals=[]
        current=[None]
        class Process:
            def __init__(self,args,env,**kwargs):
                self.action=env['TRIAL_ACTION']
                if mode=='spawn_failure' and self.action=='O_D':
                    raise OSError('injected spawn failure')
                if mode=='guard_interrupt' and self.action=='O_D':
                    clock.now=99
                self.started=clock.now
                self.pid=os.getpid()
                self.returncode=None
                launches.append(self.action)
                current[0]=self
            def poll(self):
                if self.returncode is None and clock.now-self.started>=5:
                    self.returncode=1 if mode=='nonzero' and self.action=='O_D' else 0
                return self.returncode
            def wait(self,timeout=None):
                if self.returncode is None:
                    self.returncode=-15
                return self.returncode
        def killpg(pid,sig):
            assert pid==os.getpid()
            signals.append(int(sig))
            if current[0] is not None:
                current[0].returncode=-15
        module.os=SimpleNamespace(environ=dict(os.environ),killpg=killpg)
        module.subprocess=SimpleNamespace(Popen=Process,check_output=lambda *a,**k:'test, 999999\n',
            TimeoutExpired=subprocess.TimeoutExpired,STDOUT=subprocess.STDOUT)
        failure=None
        try:
            module.supervisor()
        except BaseException as exc:
            failure=type(exc).__name__
        exit_receipt=json.loads((root/'GUARD_EXIT.json').read_text())
        expected={'normal':10,'recovery_normal':30,'nonzero':5,'guard_interrupt':94,'spawn_failure':0,'existing_log':0,'adopt_timeout':0,'adopt_identity_mismatch':0}[mode]
        assert exit_receipt['gpu_process_seconds']==expected,(mode,exit_receipt,(root/'FAILURE.json').read_text() if (root/'FAILURE.json').exists() else None)
        assert (failure is None)==(mode in ['normal','recovery_normal'])
        receipt=root/'O_D/PROCESS_EXIT.json'
        assert receipt.exists()==(mode in ['normal','recovery_normal','nonzero','guard_interrupt'])
        if mode=='recovery_normal':
            assert launches==['O_D','M_D']
        if receipt.exists():
            assert json.loads(receipt.read_text())['settled_once'] is True
        if mode=='existing_log':
            assert (root/'O_D.log').read_text()=='preserve'
        if mode in ['spawn_failure','existing_log']:
            assert launches==['download']
        if mode=='guard_interrupt':
            assert signals
        if mode=='adopt_timeout':
            # Mock signals cannot change the real /proc record; outer cleanup
            # may safely repeat them against the same verified identity.
            assert signals[:2]==[15,9] and set(signals)=={15,9}
        if mode=='adopt_identity_mismatch':
            assert not signals
        return dict(case=mode,passed=True,gpu_process_seconds=expected,simulated=True)


if __name__=='__main__':
    print(json.dumps([test(mode) for mode in ['normal','nonzero','guard_interrupt','spawn_failure','existing_log','adopt_timeout','adopt_identity_mismatch','recovery_normal']]))
