"""Process-local integration, leaving all prior deployed experiments immutable."""
import os
from pathlib import Path

VERSION = 'FULL_CONTEXT_SUMMARY8_V1_20261018'
ARMS = ('B','C','N','D','E')


def install():
    from experiments.fixed_input_guard8_v1 import configuration as cfg, executor, worker, launch, complete
    from .guard import InputGuard
    from .editor import Editor, summary_admission
    from .storage import ensure
    for module in (cfg,executor,worker,launch):
        module.VERSION = VERSION
    cfg.ARMS = launch.ARMS = ARMS
    executor.ARMS = (*ARMS,'P')
    cfg.CAPS.update(summary_forwards=50,summary_vjps=50*1024)
    executor.InputGuard = InputGuard
    worker.Editor = Editor
    def plan():
        yield 'B','calibration',0
        for arm in ARMS:
            yield arm,'base',0
            yield arm,'admit',1
        for arm in ARMS:
            for owner in range(1,9):
                yield arm,'learn',owner
                if owner in (6,7,8):
                    yield arm,'diagnostic',owner
                yield arm,'insertion',owner
                if owner in (4,8):
                    yield arm,'prefix',owner
            yield arm,'reload',8
    launch.plan = plan
    original_admit = worker.admit_current
    def admit(runtime,arm,calibration,record,fits,folder):
        value = original_admit(runtime,arm,calibration,record,fits,folder)
        value['summary_admission'] = summary_admission(runtime,arm,calibration,record,fits,Path(folder))
        return value
    worker.admit_current = admit
    original_aux = launch.auxiliary
    def auxiliary(arm,action,owner,config):
        if action == 'diagnostic':
            from .diagnostic import evaluate
            return evaluate(arm,owner,config)
        return original_aux(arm,action,owner,config)
    launch.auxiliary = auxiliary
    original_provide = launch.provide
    def provide(owner,arm):
        _,root = launch.helpers()
        # Worst source: 20 native diagnostic exports plus atomic optimizer save.
        ensure(root,3_800_000_000 if owner in (7,8) else 1_400_000_000)
        return original_provide(owner,arm)
    launch.provide = provide
    complete.remove_resource_gates()
    c,root = launch.helpers()
    from experiments.fixed_input_guard8_v1 import diagnostic
    original_snapshot = diagnostic.save_snapshot
    def save_snapshot(weight,folder,owner,slot):
        ensure(root,weight.numel()*weight.element_size()+65536)
        return original_snapshot(weight,folder,owner,slot)
    diagnostic.save_snapshot = save_snapshot
    original_save = Editor.save
    def save(self,path):
        ensure(root,executor.CHECKPOINT_BYTES)
        return original_save(self,path)
    Editor.save = save
    previous = c.budget
    def budget():
        previous()
        # Stage reservation protects heavy writes; this check covers small outputs.
        if __import__('shutil').disk_usage(root).free < 8*1024**3:
            raise OSError('Disk reserve reached')
    c.budget = budget


def main():
    from experiments.fixed_input_guard8_v1 import launch
    c,root = launch.helpers()
    config = c.read(root/'private/config.json')
    assert config['version'] == VERSION and config['authorization']['execution']
    install()
    (launch.child if os.environ['ENTRY_MODE'] == 'child' else launch.controller)()


if __name__ == '__main__':
    main()
