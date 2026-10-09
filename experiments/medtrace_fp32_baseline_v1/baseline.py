"""Check FP32 Base answer identity before inheriting semantic qualification."""
import os
import time
import traceback
from pathlib import Path
import torch
import qualified_probe as prior

q,c,p,RUN=prior.q,prior.c,prior.p,prior.RUN


def rows():
    basis,held=q.split()
    return [dict(row,identity_role='BASIS' if i<61 else 'HELDOUT',identity_index=i)
            for i,row in enumerate(basis+held)]


def decision(values):
    assert values and all(isinstance(x['text_equal'],bool) for x in values)
    return 'BASELINE_IDENTITY_PRESERVED' if all(x['text_equal'] for x in values) else 'BASELINE_REQUALIFICATION_REQUIRED'


def selfcheck():
    assert decision([dict(text_equal=True)])=='BASELINE_IDENTITY_PRESERVED'
    assert decision([dict(text_equal=True),dict(text_equal=False)])=='BASELINE_REQUALIFICATION_REQUIRED'
    return dict(status='PASS',any_changed_answer_requires_requalification=True)


def plan():
    values=rows();assert len(values)==157 and sum(x['semantic_correct'] for x in values)==124
    lock=dict(queries=157,basis=61,heldout=96,previously_correct=124,FP32_generations=157,
        native_control_generations=6,max_LLM_forwards=163*128,backwards=0,updates=0,new_Judge=0,
        max_answer_tokens=128,checkpoint_outputs=0,source=c.read(RUN/'private/GPU_SOURCE_VERSION.json'),
        roles=c.digest(values),precision='MATH_FP32_SAME_FP16_PREFILL_AND_WEIGHT_VALUES')
    c.write(RUN/'private/BASELINE_LOCK.json',lock)
    c.write(RUN/'private/BASELINE_ROWS.json',values)
    c.write(RUN/'public/ADMISSION.json',dict(status='PASS',selfcheck=selfcheck(),**{k:v for k,v in lock.items() if k not in ('source','roles')}))
    p.done('PLAN_COMPLETE')


def worker():
    import subprocess
    part=int(os.environ['PARTITION']);gpu=q.GPUS[part]
    assert int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=48000
    with p.lease(gpu):
        runtime,bindings=c.load(gpu);model=runtime.llava_model()
        assert not runtime.get_module(c.LAYER)._forward_hooks and not runtime.generation_config['do_sample']
        assert runtime.generation_config['max_new_tokens']==128
        selected=c.read(RUN/'private/BASELINE_ROWS.json')[part::6]
        original=model.prepare_inputs_labels_for_multimodal
        frozen=[]
        for row in selected:
            c.budget()
            raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
            old=c.read(row['response_path'])
            assert raw['input_ids'][0].tolist()==old['binding']['judge_input']['prompt_ids']
            assert raw['image_sha256']==old['binding']['judge_input']['image_sha256']
            with torch.no_grad():expanded=original(raw['input_ids'],None,raw['attention_mask'],None,None,raw['images'],image_sizes=None)
            assert expanded[4] is not None and expanded[4].dtype==torch.float16
            frozen.append((row,raw,tuple(x.detach().cpu() if isinstance(x,torch.Tensor) else x for x in expanded)))
        forwards=[0]
        handle=runtime.model.register_forward_pre_hook(lambda *_:forwards.__setitem__(0,forwards[0]+1))
        def generate(raw,expanded):
            used=[0];dtype=next(runtime.model.parameters()).dtype
            def prepared(*args,**kwargs):
                if args[0] is not None and torch.equal(args[0],raw['input_ids']):
                    assert args[3] is None and args[4] is None
                    used[0]+=1;assert used[0]==1
                    return tuple(x.to(device=runtime.device,dtype=dtype if x.is_floating_point() else x.dtype) if isinstance(x,torch.Tensor) else x for x in expanded)
                return original(*args,**kwargs)
            model.prepare_inputs_labels_for_multimodal=prepared
            start=forwards[0]
            try:
                with torch.inference_mode():answer=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
            finally:model.prepare_inputs_labels_for_multimodal=original
            assert used[0]==1 and answer.raw_token_ids
            return answer,forwards[0]-start
        try:
            row,raw,expanded=frozen[0]
            control,control_f=generate(raw,expanded)
            old=c.read(row['response_path'])
            assert list(control.raw_token_ids)==old['R0']['raw_token_ids'],'Frozen-prefill control differs from original native Base'
            c.write(RUN/'private'/f'CONTROL_{part}.json',dict(status='PASS',forwards=control_f,exact_tokens=True))
            runtime.model.float()
            torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            for row,raw,expanded in frozen:
                c.budget();dest=RUN/'private/outputs'/f"{row['identity_index']:03d}.json"
                assert not dest.exists(),'No implicit generation retry'
                answer,nf=generate(raw,expanded);old=c.read(row['response_path'])
                same_tokens=list(answer.raw_token_ids)==old['R0']['raw_token_ids']
                same_text=answer.decoded_text==old['R0']['raw_answer']
                value=dict(old)
                value['R0']=dict(raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids))
                value['identity']=dict(role=row['identity_role'],index=row['identity_index'],
                    text_equal=same_text,tokens_equal=same_tokens,previously_correct=row['semantic_correct'],
                    forwards=nf,generated_tokens=len(answer.raw_token_ids),at_cap=len(answer.raw_token_ids)>=128,
                    ended_EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id)
                value['binding']=dict(old['binding'],execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'))
                value['precision']='MATH_FP32_FROZEN_FP16_PREFILL'
                value['identity_parent']=row['response_path']
                value['audit']=dict(old['audit'],annotation_audit_inherited_from_FP16=True,generated_tokens=len(answer.raw_token_ids),at_generation_cap=len(answer.raw_token_ids)>=128)
                c.write(dest,value)
                print('BASELINE',part,row['identity_index'],'TEXT_EQUAL',same_text,'TOKENS_EQUAL',same_tokens,flush=True)
            assert not any(v.grad is not None for v in runtime.model.parameters())
        finally:
            model.prepare_inputs_labels_for_multimodal=original;handle.remove()
    p.done('WORKER_'+str(part))


def report():
    values=[c.read(path)['identity'] for path in sorted((RUN/'private/outputs').glob('*.json'))]
    assert len(values)==157 and {x['index'] for x in values}==set(range(157))
    controls=[c.read(RUN/'private'/f'CONTROL_{i}.json') for i in range(6)]
    assert all(x['exact_tokens'] for x in controls) and not list((RUN/'private').glob('FAILURE*'))
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');inherited=c.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==6 and all(x.get('ended_epoch') for x in ledger['gpu_sessions'])
    panels={}
    for role in ('BASIS','HELDOUT','PRIMARY63'):
        rs=[x for x in values if x['role']==role or role=='PRIMARY63' and x['role']=='HELDOUT' and x['previously_correct']]
        panels[role]=dict(queries=len(rs),text_equal=sum(x['text_equal'] for x in rs),tokens_equal=sum(x['tokens_equal'] for x in rs),
            changed_previously_correct=sum(x['previously_correct'] and not x['text_equal'] for x in rs),at_cap=sum(x['at_cap'] for x in rs),EOS=sum(x['ended_EOS'] for x in rs))
    total=sum(x['forwards'] for x in values+controls);assert total<=163*128
    result=dict(status='COMPLETE',decision=decision(values),panels=panels,native_controls=6,
        new_generations=163,LLM_forwards=total,backwards=0,updates=0,new_Judge=0,new_checkpoints=0,
        changed_texts=sum(not x['text_equal'] for x in values),no_requalification_or_training_claim=True,
        resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-inherited['gpu_seconds_used'])/3600,
            cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,cumulative_Judge=ledger['Judge_attempts']))
    c.write(RUN/'public/RESULTS.json',result)
    c.write(RUN/'public/COMPLETION_AUDIT.json',dict(status='PASS',queries=157,native_controls=6,sessions_ended=6,Base_updates=0))
    (RUN/'public/REPORT_ZH.md').write_text('# FP32 Base回答一致性\n\n'+json_text(result)+'\n\n答案文本变化不是医学错误判定；变化项须独立资格核验，不过滤失败或改原主面板分母。本阶段没有候选优化或保护评估。\n')
    p.done('REPORT_COMPLETE',dict(decision=result['decision']))


def json_text(value):
    import json
    return json.dumps(value,ensure_ascii=False,indent=2)


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('baseline.py','baseline_worker',gpu,i) for i,gpu in enumerate(q.GPUS)])
    pipeline.wait([pipeline.launch('baseline.py','baseline_report')])


if __name__=='__main__':
    try:{'baseline_plan':plan,'baseline_worker':worker,'baseline_report':report,'baseline_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
