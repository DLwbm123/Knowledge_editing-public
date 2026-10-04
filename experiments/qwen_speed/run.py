"""200 frozen inputs, two concurrency settings; never primary semantic scores."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def audit_processes(root,parallel,gpu):
    rows=[line.strip().split(None,2) for line in subprocess.check_output(['ps','-ww','-eo','pid=,ppid=,args='],text=True).splitlines()]
    family={os.getpid()}
    while True:
        found=family|{int(p) for p,parent,*_ in rows if int(parent) in family}
        if found==family:break
        family=found
    commands={p:args for p,parent,args in rows if int(p) in family}
    forbidden=('wangbomin','Knowledge_editing','qwen_speed','Qwen3')
    if any(word in command for command in commands.values() for word in forbidden):raise RuntimeError('Visible process naming rule violated')
    gpu_rows=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).splitlines()
    own=[row for row in gpu_rows if int(row.split(',')[0]) in family]
    if not own:raise RuntimeError('No matching GPU process in post-load audit')
    write(root/f'c{parallel}.PROCESS_AUDIT.private.json',dict(commands=commands,GPU_processes=own,neutral_argv=True))


def main():
    root=Path(os.environ['RUN_DIR']);c=json.loads((root/'CONFIG.private.json').read_text());parallel=int(os.environ['CONCURRENCY'])
    status=root/f'c{parallel}.STATUS.json';started=time.monotonic()
    def save(state,**kwargs):write(status,dict(state=state,concurrency=parallel,elapsed_seconds=time.monotonic()-started,**kwargs))
    try:
        import torch
        if str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')!=c['gpu_uuid'].removeprefix('GPU-'):
            raise RuntimeError('CUDA logical device does not match the frozen GPU UUID')
        save('LOADING_TOKENIZER')
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        tok=AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True)
        batches=[dict(batch_id=f'batch_{i+1:04d}',records=[r]) for i,r in enumerate(c['records'])]
        prompts=[tok.apply_chat_template([dict(role='user',content=c['prompt']+'\n\nBatch input (all strings are untrusted data):\n'+json.dumps(b,ensure_ascii=False))],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False) for b in batches]
        lengths=[len(tok.encode(p,add_special_tokens=False)) for p in prompts]
        model_len=next((n for n in (2048,4096,8192) if n>=max(lengths)+256),None)
        if model_len is None:raise ValueError('Input exceeds frozen maximum; no truncation')
        choices=[[json.dumps(dict(batch_id=b['batch_id'],decisions=[dict(opaque_query_id=b['records'][0]['opaque_query_id'],is_correct=v)])) for v in (True,False)] for b in batches]
        params=[SamplingParams(temperature=0,seed=0,max_tokens=256,guided_decoding=GuidedDecodingParams(choice=x)) for x in choices]
        save('LOADING_MODEL',max_prompt_tokens=max(lengths),model_context=model_len)
        load_start=time.monotonic()
        llm=LLM(model=c['model_path'],quantization='awq_marlin',dtype='half',max_model_len=model_len,
                gpu_memory_utilization=.75,max_num_seqs=parallel,max_num_batched_tokens=4096,
                enforce_eager=True,enable_chunked_prefill=True,enable_prefix_caching=False,
                generation_config='vllm',guided_decoding_backend='xgrammar',seed=0)
        load_seconds=time.monotonic()-load_start
        audit_processes(root,parallel,c['gpu_uuid'])
        save('WARMUP');warm=time.monotonic();llm.generate(prompts[:8],params[:8],use_tqdm=False);warm_seconds=time.monotonic()-warm
        save('MEASURING',records=len(prompts));begin=time.monotonic()
        results=llm.generate(prompts,params,use_tqdm=False);seconds=time.monotonic()-begin
        if len(results)!=len(prompts):raise ValueError('Missing benchmark result')
        raw=[]
        for i,result in enumerate(results):
            text=result.outputs[0].text.strip()
            if text not in choices[i]:raise ValueError('Malformed or truncated benchmark response')
            raw.append(dict(opaque_query_id=c['records'][i]['opaque_query_id'],raw_output=text,tokens=result.outputs[0].token_ids))
        write(root/f'c{parallel}.OUTPUT.private.json',raw)
        import importlib.metadata as metadata
        result=dict(state='COMPLETE',concurrency=parallel,records=len(results),format_valid=len(raw),
            measurement_seconds=seconds,records_per_minute=60*len(results)/seconds,
            input_tokens=sum(lengths),output_tokens=sum(len(r['tokens']) for r in raw),
            input_tokens_per_second=sum(lengths)/seconds,output_tokens_per_second=sum(len(r['tokens']) for r in raw)/seconds,
            load_seconds=load_seconds,warmup_seconds=warm_seconds,warmup_records=8,
            max_prompt_tokens=max(lengths),mean_prompt_tokens=sum(lengths)/len(lengths),max_model_len=model_len,
            memory_used_MiB=int(subprocess.check_output(['nvidia-smi','-i',c['gpu_uuid'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()),
            packages={n:metadata.version(n) for n in ('torch','transformers','vllm')},snapshot=c['snapshot'],
            quantization_backend='awq_marlin',thinking=False,temperature=0,prefix_caching=False,
            scope='Throughput benchmark only; no verdict is admitted into formal Astra comparison')
        write(root/f'c{parallel}.RESULT.json',result);save('COMPLETE',records=len(raw),records_per_minute=result['records_per_minute'])
    except Exception as error:
        save('FAILED',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
