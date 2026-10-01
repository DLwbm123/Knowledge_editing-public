"""Paired Base-model prompt diagnostic; no optimizer or weight writes."""
import json,signal,time,traceback
from pathlib import Path
import torch
from .gate import require_external_approval
from .native_io import load,prepare,generation_snapshot,WEIGHT
from .editor import mean_answer_logprob


def run(config,approval,trusted):
    require_external_approval('EXPLORATORY_PILOT',config['bindings'],approval,config,trusted_authorization=trusted)
    root=Path(config['run_root']);started=time.monotonic();model=None
    status=dict(state='LOADING',completed=0,intended=len(config['prompt_rows']),cases=[],
        result_kind='READ_ONLY_TRAIN_SUPPORT_PROMPT_DIAGNOSTIC',judge_calls=0,new_GGN_calls=0,
        previous_GGN_calls=config['prior_GGN_calls'],previous_native_seconds=config['prior_native_seconds'],
        original_scientific_data_status='BLOCKED_DATA')
    def save():
        status.update(elapsed_seconds=time.monotonic()-started,cumulative_GGN_calls=config['prior_GGN_calls'],
            cumulative_native_seconds=config['prior_native_seconds']+time.monotonic()-started)
        p=root/'STATUS.tmp';p.write_text(json.dumps(status,indent=2,allow_nan=False)+'\n');p.replace(root/'STATUS.json')
    def timeout(*_):raise TimeoutError('prompt diagnostic wall ceiling reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(config['native_wall_seconds_cap'])
    def unchanged():
        return (versions=={n:p._version for n,p in model.named_parameters()} and
                torch.equal(dict(model.named_parameters())[WEIGHT],base))
    try:
        save();torch.manual_seed(20261001);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model,tokenizer,processor=load(config)
        versions={n:p._version for n,p in model.named_parameters()}
        base=dict(model.named_parameters())[WEIGHT].detach().clone()
        status['state']='EXPERIMENT_RUNNING';save()
        for i,row in enumerate(config['prompt_rows']):
            status['active_case']=i;save();raw={};metrics={}
            for arm,suffix in [('original',''),('short_answer',config['question_suffix'])]:
                inputs,labels,generation,_=prepare(model,tokenizer,processor,row,question_suffix=suffix)
                with torch.no_grad():
                    score=float(mean_answer_logprob(model(**inputs).logits.float(),labels,eos_id=tokenizer.eos_token_id)[0])
                if not torch.isfinite(torch.tensor(score)):raise RuntimeError('nonfinite answer score')
                output=generation_snapshot(model,tokenizer,generation,row['answer'],config['generation_max_new_tokens'])
                raw[arm]=output;metrics[arm]=dict(answer_score=score,**output['metrics'])
                if not unchanged():raise RuntimeError('read-only model parameters changed')
                del inputs,labels,generation
            # Preserve both arms even if the original control fails its replay.
            (root/f'{i}.generation.private.json').write_text(json.dumps(raw,indent=2)+'\n')
            replay=raw['original']['tokens']==config['expected_base_tokens'][str(i)]
            if not replay:raise RuntimeError('original generation differs from frozen E11 Base')
            result=dict(index=str(i),**metrics,original_generation_replay_exact=replay,parameters_unchanged=True,
                score_change=metrics['short_answer']['answer_score']-metrics['original']['answer_score'],
                tokens_changed=raw['original']['tokens']!=raw['short_answer']['tokens'])
            status['cases'].append(result);status['completed']=len(status['cases']);save()
        status['state']='EXPERIMENT_COMPLETE'
    except BaseException as error:
        status['state']='STOPPED_ON_ERROR';status['error']=type(error).__name__+': '+str(error)
        (root/'FAILURE.private.json').write_text(json.dumps(dict(error=status['error'],traceback=traceback.format_exc()),indent=2))
    finally:
        if model is not None and 'base' in locals():status['final_Base_unchanged']=unchanged()
        status['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
        signal.alarm(0);save()
    return status
