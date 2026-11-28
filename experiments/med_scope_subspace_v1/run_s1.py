"""Run the locked dense training stages on the two admitted devices."""
import time

from .run_s0 import owned_bytes, read, record, run_group
from pathlib import Path


def main():
    cfg = read('chain.json')
    deadline, cap = cfg['deadline'], cfg['storage_cap']
    if cfg['gpus']!=[6,7] or not time.time()<deadline or owned_bytes(Path.cwd())>=cap:
        raise ValueError('Dense run GPU, deadline, or storage binding differs')
    module = 'experiments.med_scope_subspace_v1.s1_dense'
    lock = read('readiness.json')
    from .s1_dense import digest
    if lock['lock']!=digest('lock.json') or any(
            read(f'smoke{i}.json')['lock']!=lock['lock'] for i in (0,1)):
        raise ValueError('Dense-forward readiness differs')
    for stage in ('warm','scale','grid','select'):
        cpu = stage in {'scale','select'}
        args = [[stage]] if cpu else [[stage,str(i)] for i in (0,1)]
        run_group(stage,args,[''] if cpu else [6,7],deadline,cap,entrypoint=module)
    selected = read('selection.json')
    if not selected['training_arms']:
        record('complete.json',dict(state='NO_BETA',ended=time.time(),selection=selected))
        return
    run_group('train',[['train',str(i)] for i in (0,1)],[6,7],deadline,cap,entrypoint=module)
    layout = read('layout.json')
    eval_args = []
    for i in (0,1):
        eval_args.append(['eval','--policy','writer','--manifest',layout['manifest'],
            '--solution',layout['solution'],'--contract',f'config{i}.json','--shards','2','--shard',str(i),
            '--roles','official_ind','hold_pos','hold_ind','hold_vis','native','fit',
            '--bank','bank0.pt','bank1.pt','--writer-arms',*selected['training_arms'],
            '--base-prefixes',*layout['prefixes'],'--output',f'behavior{i}.json'])
    run_group('behavior',eval_args,[6,7],deadline,cap)
    run_group('score_input',[['score-input','--inputs','behavior0.json','behavior1.json',
                              '--output','scores']],[''],deadline,cap)
    record('complete.json',dict(state='AWAITING_BLIND_CORE',ended=time.time(),selection=selected))


if __name__=='__main__':
    main()
