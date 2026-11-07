"""Generate the frozen T5 prior/follow-up Base panel under matched FP32 inference."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(os.environ['T5_REMOTE_ROOT']);SHARD=int(os.environ['SHARD'])
CLOCK=json.loads((ROOT/'CLOCK.json').read_text())
COUNTS=dict(forwards=0,generations=0,consumers=0,reused=0,diagnostic_generations=0)


def write(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False));tmp.replace(p)


def budget():
    assert time.time()<CLOCK['hard_end']-120 and not (ROOT/'BASE_STOP.json').exists()
    assert time.time()-json.loads((ROOT/'BASE_GUARD_HEALTH.json').read_text())['time']<45
    assert COUNTS['generations']<=646 and COUNTS['diagnostic_generations']<=1


def main():
    with (ROOT/f'BASE_{SHARD}_ONCE').open('x'):pass
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('guard stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original clock')))
    signal.alarm(max(1,int(CLOCK['hard_end']-time.time()-120)))
    parent=Path(os.environ['PARENT_RUN']);cfg=json.loads((ROOT/'CONFIG.private.json').read_text())
    sys.path[:0]=[str(ROOT/'tools'),cfg['med150_tools'],str(parent/'private/tools'),str(parent/'private/source'),str(parent/'private/official_llava')]
    import torch
    import numpy as np
    import module_run as r
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);r.c.budget=budget
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=parent/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    configuration=json.loads((ROOT/'BASE_CONFIGURATION.private.json').read_text())
    assert runtime.generation_config==configuration['generation']
    runtime.run_root=ROOT/f'base_runtime{SHARD}';runtime.run_root.mkdir()
    data=json.loads((ROOT/'SOURCE_INPUTS.private.json').read_text());tasks={t['edit_id']:t for t in data['tasks']}
    rows=data['base_plan'][SHARD::int(os.environ['BASE_SHARDS'])]
    rng=torch.load(Path(os.environ['RAW_MED_ROOT'])/'GENERATION_RNG.pt',map_location='cpu',weights_only=False)
    handle=runtime.model.register_forward_pre_hook(lambda *_:COUNTS.__setitem__('forwards',COUNTS['forwards']+1))
    cache={}
    for index,row in enumerate(rows):
        budget();task=tasks[row['edit_id']];query=row['query'];runtime.model.half()
        try:item=r.s.prepare(runtime,{'configuration_only':configuration},task,[query])[0]
        finally:r.old.fp32(runtime)
        key=r.c.digest(item[4]);reused=key in cache
        if not reused:
            r.c.restore_rng(rng['rng']);np.random.set_state(rng['numpy_rng']);COUNTS['generations']+=1
            value,trace=r.s.fit.short.generate(runtime,None,item[2],item[3])
            answer=dict(raw_answer=value.decoded_text,raw_token_ids=list(value.raw_token_ids))
            assert 0<len(answer['raw_token_ids'])<=1024
            cache[key]=answer
            if index==0:
                r.c.restore_rng(rng['rng']);np.random.set_state(rng['numpy_rng']);COUNTS['diagnostic_generations']+=1
                repeated,_=r.s.fit.short.generate(runtime,None,item[2],item[3])
                assert list(repeated.raw_token_ids)==answer['raw_token_ids'],'Fixed Base generation parity failed'
        else:COUNTS['reused']+=1
        cid=r.c.digest(['BASE',row['owner'],row['role'],query['query_id']]);dest=ROOT/'base_outputs'/f'{cid}.json'
        assert not dest.exists()
        write('base_outputs/'+dest.name,dict(consumer_id=cid,owner=row['owner'],edit_id=row['edit_id'],role=row['role'],
            query_id=query['query_id'],judge_input=item[4],answer=cache[key],actual_precision='float32',
            prediction_reused=reused,EOS=cache[key]['raw_token_ids'][-1]==runtime.adapter.tokenizer.eos_token_id,
            at_cap=len(cache[key]['raw_token_ids'])>=1024))
        COUNTS['consumers']+=1
        if index%5==0:write(f'BASE_PROGRESS_{SHARD}.json',dict(time=time.time(),counts=COUNTS,total=len(rows)))
    handle.remove();assert COUNTS['consumers']==len(rows)
    write(f'BASE_COMPLETE_{SHARD}.json',dict(status='SHARD_COMPLETE',time=time.time(),counts=COUNTS))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write(f'BASE_FAILURE_{SHARD}.json',dict(time=time.time(),error=traceback.format_exc(),counts=COUNTS));raise
    finally:write(f'BASE_EXIT_{SHARD}.json',dict(code=code,time=time.time(),counts=COUNTS))
