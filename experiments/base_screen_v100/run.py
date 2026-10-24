"""Frozen two-base qualification; private paths/rows supplied via a private config."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ['SCREEN_ROOT'])
CFG = json.loads((ROOT/'CONFIG.json').read_text())


def write(name, value):
    p = ROOT/name
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(p)


def worker(key):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    assert transformers.__version__ == '4.51.3'
    model_path = CFG['models'][key]['path']
    counts = dict(forwards=0, generations=0, vision=0, backwards=0, updates=0)
    started = time.time()
    write(key+'/START.json', dict(time=started, pid=os.getpid()))
    def check():
        assert time.time() < CFG['hard_end'], 'Original deadline reached'
        assert time.time()-started < CFG['gpu_seconds'], 'GPU chain cap reached'
        assert not (ROOT/'STOP.json').exists(), 'Guard stopped chain'
    def forward(*args):
        check()
        counts['forwards'] += 1
        assert counts['forwards'] <= 100000
    def vision(*args):
        counts['vision'] += 1
    try:
        check()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map={'':'cuda:0'}, local_files_only=True)
        model.eval().requires_grad_(False)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
            min_pixels=256*28*28, max_pixels=1280*28*28)
        model.register_forward_pre_hook(forward)
        model.visual.register_forward_hook(vision)
        argv = subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='], text=True).strip()
        assert not any(s.lower() in argv.lower() for s in CFG['forbidden_argv'])
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True).splitlines()
        own = [s for s in apps if s.split(',')[0].strip()==str(os.getpid())]
        assert own and all(s.split(',')[1].strip()==CFG['gpu_uuid'] for s in own)
        write(key+'/ENVIRONMENT.json',dict(torch=torch.__version__,transformers=transformers.__version__,
            model_class=type(model).__name__,processor=processor.image_processor.to_dict(),
            generation_config=model.generation_config.to_dict(),argv=argv,gpu_processes=own))
        rows = json.loads((ROOT/'SOURCE_ROWS.json').read_text())
        assert len(rows)==90 and len({r['qid'] for r in rows})==90
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos,int) else eos
        for row in rows:
            check()
            dest = key+'/outputs/'+str(row['qid'])+'.json'
            assert not (ROOT/dest).exists()
            with Image.open(row['image_path']) as im:
                assert im.mode=='RGB'
                messages = [dict(role='user', content=[dict(type='image', image=im),dict(type='text',text=row['question'])])]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[prompt],images=[im],padding=True,return_tensors='pt').to('cuda')
            grid = inputs['image_grid_thw'].tolist()
            visual_tokens = int((inputs['input_ids']==model.config.image_token_id).sum())
            assert 1 <= visual_tokens <= 1280
            assert visual_tokens == int(inputs['image_grid_thw'].prod(dim=-1).sum())//4
            before = dict(counts)
            write(key+'/ATTEMPT.json',dict(qid=row['qid'],time=time.time(),counts=counts))
            counts['generations'] += 1
            assert counts['generations']<=90
            with torch.inference_mode():
                out = model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=1024,
                    repetition_penalty=1.05,temperature=None,top_p=None,top_k=None,use_cache=True)
            tokens = out[0,inputs['input_ids'].shape[1]:].cpu().tolist()
            assert counts['vision']-before['vision']==1
            write(dest,dict(qid=row['qid'],query_id=row['query_id'],raw_answer=processor.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                raw_token_ids=tokens,EOS=bool(tokens and tokens[-1] in eos),at_cap=len(tokens)>=1024,
                input_tokens=int(inputs['input_ids'].shape[1]),image_grid_thw=grid,visual_tokens=visual_tokens,
                vision_calls=counts['vision']-before['vision'],forwards=counts['forwards']-before['forwards'],time=time.time()))
            write(key+'/PROGRESS.json',dict(counts=counts,time=time.time()))
        write(key+'/COMPLETE.json',dict(counts=counts,time=time.time(),seconds=time.time()-started))
    except BaseException as exc:
        write(key+'/FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time(),counts=counts))
        raise
    finally:
        write(key+'/EXIT.json',dict(counts=counts,time=time.time(),seconds=time.time()-started))


def supervisor():
    # This process independently bounds both sequential workers, including loading.
    with (ROOT/'ONCE').open('x'): pass
    started = time.time()
    used = 0.0
    child = None
    def stop_child():
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
    try:
        for key in ['Q','L']:
            assert time.time()<CFG['hard_end'] and used<CFG['gpu_seconds']
            uuid,free = subprocess.check_output(['nvidia-smi','-i',str(CFG['gpu']),
                '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
            assert uuid==CFG['gpu_uuid'] and int(free)>=45*1024
            with (ROOT/(key+'.log')).open('x') as log:
                env=dict(os.environ,SCREEN_MODE=key)
                begin=time.time()
                child=subprocess.Popen([CFG['python'],CFG['neutral_entry']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(key+'/IDENTITY.json',dict(pid=child.pid,start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],start=begin))
                while child.poll() is None:
                    size=sum(p.stat().st_size for root in CFG['owned_roots'] for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())
                    reason=None
                    if time.time()>=CFG['hard_end']-10:reason='WALL_LIMIT'
                    if used+time.time()-begin>=CFG['gpu_seconds']-10:reason='GPU_PROCESS_LIMIT'
                    if size>=CFG['storage_bytes']-10_000_000:reason='STORAGE_LIMIT'
                    write('GUARD_HEALTH.json',dict(time=time.time(),size=size,gpu_seconds=used+time.time()-begin,key=key))
                    if reason:
                        write('STOP.json',dict(reason=reason,time=time.time()))
                        stop_child()
                        raise RuntimeError(reason)
                    time.sleep(5)
                used+=time.time()-begin
                write(key+'/PROCESS_EXIT.json',dict(code=child.returncode,seconds=time.time()-begin,time=time.time()))
                assert child.returncode==0 and (ROOT/key/'COMPLETE.json').exists()
        write('COMPLETE.json',dict(status='COMPLETE',gpu_process_seconds=used,time=time.time()))
    except BaseException as exc:
        write('FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()))
        stop_child()
        raise
    finally:
        write('GUARD_EXIT.json',dict(time=time.time(),elapsed=time.time()-started,gpu_process_seconds=used))


if __name__=='__main__':
    if os.environ.get('SCREEN_MODE') in ['Q','L']: worker(os.environ['SCREEN_MODE'])
    else: supervisor()
