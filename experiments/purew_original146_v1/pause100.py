"""One user-authorized pause after durable prefix100; no training or Judge replay."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def planned_controller_failure(path,run):
    receipt=run/'private/pause_prefix100/SIGNAL_ATTEMPT.json'
    if not receipt.exists():return False
    request=json.loads(receipt.read_text())
    return (Path(path).name=='controller_'+str(request['controller']['pid'])+'.json'
            and (run/'private/outputs/bank_R0/0/100/COMPLETE.json').exists()
            and json.loads(Path(path).read_text()).get('traceback','').endswith('AssertionError: gpu failed; inspect saved state, never blindly restart\n'))


def alive(row):
    p=Path('/proc')/str(row['pid'])
    if not p.exists():return False
    try:fields=(p/'stat').read_text().split()
    except FileNotFoundError:return False
    if fields[2]=='Z':return False
    assert fields[21]==str(row['start_ticks']), 'PID reused; do not signal'
    argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert row['entry'] in argv and not any(x in argv for x in ('wangbomin','Knowledge_editing','purew'))
    return True


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    run=Path(os.environ['RUN_ROOT']);root=run/'private/pause_prefix100'
    auth=json.loads((root/'AUTHORIZATION.json').read_text())
    assert auth['prefix']==100 and auth['user_authorized']
    assert not (root/'SIGNAL_ATTEMPT.json').exists(), 'Signal outcome requires review, never resend blindly'
    assert not (run/'private/PAUSED_USER_REQUEST.json').exists()
    marker=run/'private/outputs/bank_R0/0/100/COMPLETE.json'
    deadline=json.loads((run/'RUN_MANIFEST.json').read_text())['deadline_epoch']
    owner=0;next_check=0
    while not marker.exists():
        assert time.time()<deadline, 'Original wall budget expired before prefix100'
        if time.time()>=next_check:
            assert alive(auth['gpu']) and alive(auth['controller']), 'Predecessor stopped before target; inspect state'
            assert not list((run/'private/failures').glob('*.json')), 'Existing failure requires review'
            owner=max(int(p.name) for p in (run/'private/steps/bank_R0').iterdir())
            assert owner<=100, 'Target marker absent after target edit'
            write(root/'STATUS.json',dict(status='WAITING_DURABLE_PREFIX100',owner=owner,epoch=time.time()))
            next_check=time.time()+(10 if owner==100 else 2)
        time.sleep(.05 if owner==100 else 2)
    panel=json.loads(marker.read_text());assert panel['phase']['prefix']==100 and panel['phase']['owner']==0
    assert panel['phase']['protocol']==auth['protocol_binding']
    files=[p for p in marker.parent.glob('*.json') if p.name!='COMPLETE.json' and not p.name.endswith('.STARTED.json')]
    assert len(files)==panel['outputs'] and alive(auth['gpu']) and alive(auth['controller'])
    assert not list((run/'private/failures').glob('*.json')), 'Failure before planned signal remains blocking'
    uuid=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid==auth['GPU_UUID']
    # A cooperative stop may commit the next in-flight step; preserve and disclose it.
    request=dict(gpu=auth['gpu'],controller=auth['controller'],prefix=100,epoch=time.time())
    signal_temp=root/'signal_attempt.tmp'
    signal_temp.write_text(json.dumps(request))
    os.link(signal_temp,root/'SIGNAL_ATTEMPT.json');signal_temp.unlink()
    os.kill(auth['gpu']['pid'],signal.SIGTERM)
    deadline_exit=time.time()+120
    while alive(auth['gpu']) or alive(auth['controller']):
        assert time.time()<deadline_exit, 'Cooperative stop not confirmed; never escalate to SIGKILL'
        time.sleep(.2)
    import torch
    active=run/'private/active/bank_R0.pt'
    state=torch.load(active,map_location='cpu',weights_only=True)
    assert state['mode']=='bank_R0' and state['owner'] in (100,101)
    assert state['lock']==auth['protocol_binding']
    assert state['step']==160 if state['owner']==100 else 1<=state['step']<160
    assert state['committed']==(state['owner']-1)*160+state['step']
    assert torch.equal(state['native'],state['master'].half())
    event=json.loads((run/'private/steps/bank_R0'/str(state['owner'])/(str(state['step'])+'.json')).read_text())
    assert state['event']==event and 'numpy_rng' in state and 'rng' in state
    assert json.loads((run/'private/units/bank_R0/100.json').read_text())['step']==160
    ledger=json.loads((run/'RESOURCE_LEDGER.json').read_text())
    assert not any(not x.get('ended_epoch') for x in ledger['gpu_sessions'])
    failure_dir=run/'private/failures'
    for failure in failure_dir.glob('*.json'):
        assert planned_controller_failure(failure,run), 'Unexpected failure remains blocking'
        dest=root/'retained_stop_records'/failure.name;dest.parent.mkdir(exist_ok=True);failure.replace(dest)
    assert not (run/'private/TRAINING_GENERATION_COMPLETE.json').exists()
    assert not (run/'private/FINAL_CONSUMED.json').exists()
    receipt=dict(status='PAUSED_USER_REQUEST',prefix100_generated=True,prefix100_outputs=panel['outputs'],
                 checkpoint_owner=state['owner'],checkpoint_step=state['step'],continuous_committed=state['committed'],
                 extra_committed_after_prefix100=state['committed']-16000,active_checkpoint_retained=True,
                 native_master_match=True,RNG_and_event_preserved=True,GPU_lease_closed=True,
                 GPU_seconds_used=ledger['gpu_seconds_used'],full_run_complete=False,OL_dependency_satisfied=False,
                 scoring_may_continue=True,automatic_resume=False,epoch=time.time())
    write(root/'STATUS.json',receipt);write(run/'private/PAUSED_USER_REQUEST.json',receipt)
    write(run/'public/PAUSE_PREFIX100_RECEIPT.json',receipt)
    write(run/'public/PROGRESS.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(Path(os.environ['RUN_ROOT'])/'private/pause_prefix100/GUARD_FAILURE.json',dict(epoch=time.time(),traceback=traceback.format_exc(),automatic_retry=False))
        raise
