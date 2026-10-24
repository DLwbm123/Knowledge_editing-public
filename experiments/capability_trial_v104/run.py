"""One native two-model capability trial, bounded independently of the caller."""
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ['TRIAL_ROOT'])
CFG = json.loads((ROOT / 'CONFIG.json').read_text())


def write(name, data):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def check():
    assert time.time() < CFG['hard_end'] - 10, 'Original wall deadline'
    assert not (ROOT / 'STOP.json').exists(), 'Guard stopped trial'


def download():
    from huggingface_hub import snapshot_download
    credential = Path(CFG['credential_file'])
    try:
        check()
        snapshot_download(repo_id=CFG['candidate_id'], revision=CFG['revision'],
            local_dir=CFG['candidate_path'], token=credential.read_text().strip(),
            allow_patterns=CFG['download_files'], max_workers=2)
        actual = {}
        for name, size in CFG['download_sizes'].items():
            path = Path(CFG['candidate_path']) / name
            assert path.is_file() and path.stat().st_size == size, name
            actual[name] = path.stat().st_size
        write('DOWNLOAD_COMPLETE.json', dict(time=time.time(), sizes=actual,
            revision=CFG['revision'], verification='hub completion plus known file sizes; no additional hashes'))
    finally:
        credential.unlink(missing_ok=True)


def setup_old():
    sys.path.insert(0, CFG['old_dependencies'])
    sys.path.insert(0, CFG['old_source'])
    import torch
    import transformers
    import tokenizers
    assert transformers.__version__ == '4.36.2' and tokenizers.__version__ == '0.15.2'
    from llava.model.builder import load_pretrained_model
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32=False
    disable_torch_init()
    original = CLIPVisionTower.__init__
    def local(instance, name, *args, **kwargs):
        original(instance, CFG['old_vision'], *args, **kwargs)
    CLIPVisionTower.__init__ = local
    try:
        tokenizer, model, processor, _ = load_pretrained_model(CFG['old_model'], None,
            get_model_name_from_path(CFG['old_model']), device='cuda')
    finally:
        CLIPVisionTower.__init__ = original
    assert not getattr(type(model), '_m3bench_transformers_compat_installed', False)
    model.half().eval().requires_grad_(False)
    return torch, model, tokenizer, processor


def old_inputs(model, tokenizer, processor, row, with_image):
    from PIL import Image
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
    question = row['question']
    pixels = None
    if with_image:
        marker = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
        question = marker + '\n' + question
        with Image.open(row['image_path']) as im:
            assert im.mode == 'RGB'
            pixels = process_images([im], processor, model.config)[0].unsqueeze(0).half().cuda()
    conv = conv_templates['mistral_instruct'].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    assert int((ids == IMAGE_TOKEN_INDEX).sum()) == int(with_image)
    return dict(input_ids=ids, images=pixels)


def worker(action):
    key, panel = action.split('_')
    if key == 'O':
        torch, model, tokenizer, processor = setup_old()
        vision_module = model.get_vision_tower()
    else:
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(CFG['candidate_path'],
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'}, local_files_only=True)
        model.eval().requires_grad_(False)
        processor = AutoProcessor.from_pretrained(CFG['candidate_path'], local_files_only=True)
        tokenizer = processor.tokenizer
        vision_module = model.vision_tower
    import transformers
    from PIL import Image
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    counts = dict(forwards=0, generations=0, vision=0, backwards=0, updates=0)
    def forward(*_):
        check()
        counts['forwards'] += 1
        assert counts['forwards'] <= 60000
    def vision(*_):
        counts['vision'] += 1
    model.register_forward_pre_hook(forward)
    vision_module.register_forward_hook(vision)
    argv = subprocess.check_output(['ps', '-ww', '-p', str(os.getpid()), '-o', 'args='], text=True).strip()
    assert not any(s.lower() in argv.lower() for s in CFG['forbidden_argv'])
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_memory', '--format=csv,noheader'], text=True).splitlines()
    own = [s for s in apps if s.split(',')[0].strip() == str(os.getpid())]
    assert own and all(s.split(',')[1].strip() == CFG['gpu_uuid'] for s in own)
    image_processor = processor if key == 'O' else processor.image_processor
    write(action + '/ENVIRONMENT.json', dict(torch=torch.__version__, transformers=transformers.__version__,
        architecture=type(model).__name__, processor=image_processor.to_dict(), generation_config=model.generation_config.to_dict(),
        argv=argv, gpu_processes=own))
    rows = [r for r in json.loads((ROOT / 'SOURCE_ROWS.json').read_text()) if r['panel'] == panel]
    assert len(rows) == 32
    if panel == 'D':
        # One text-only input for each class/form; there is no image-conditioned reference.
        unique = {}
        for row in rows:
            unique.setdefault((row['label'], row['form']), row)
        rows += [dict(r, qid='N_' + r['label'] + '_' + r['form'], no_image=True) for r in unique.values()]
    assert len(rows) == (36 if panel == 'D' else 32)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    eos = eos or [tokenizer.eos_token_id]
    for row in rows:
        check()
        image = not row.get('no_image', False)
        dest = action + '/outputs/' + row['qid'] + '.json'
        assert not (ROOT / dest).exists()
        if key == 'O':
            inputs = old_inputs(model, tokenizer, processor, row, image)
            pixels = inputs['images']
        else:
            content = []
            if image:
                with Image.open(row['image_path']) as im:
                    assert im.mode == 'RGB'
                    content.append(dict(type='image', image=im.copy()))
            content.append(dict(type='text', text=row['question']))
            inputs = processor.apply_chat_template([dict(role='user', content=content)],
                add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to('cuda', dtype=torch.bfloat16)
            pixels = inputs.get('pixel_values')
            assert (pixels is not None) == image
            assert int((inputs['input_ids'] == model.config.image_token_index).sum()) == (256 if image else 0)
        pixel_info = None
        if pixels is not None:
            assert bool(torch.isfinite(pixels).all())
            pixel_info = dict(shape=list(pixels.shape), min=float(pixels.min()), max=float(pixels.max()), mean=float(pixels.float().mean()))
        write(action + '/attempts/' + row['qid'] + '.json', dict(time=time.time()))
        before = dict(counts)
        counts['generations'] += 1
        with torch.inference_mode():
            # Native LLaVA-Mistral names this argument `inputs`, unlike HF Gemma.
            generation_inputs = dict(inputs=inputs['input_ids'], images=inputs['images']) if key == 'O' else inputs
            generated = model.generate(**generation_inputs, do_sample=False, num_beams=1,
                max_new_tokens=1024, use_cache=True)
        # Native LLaVA generates from inputs_embeds and returns only new IDs (plus BOS).
        tokens = generated[0].cpu().tolist() if key == 'O' else generated[0, inputs['input_ids'].shape[1]:].cpu().tolist()
        assert counts['vision'] - before['vision'] == int(image), 'Unexpected native visual encoder calls'
        write(dest, dict(qid=row['qid'], panel=panel, condition='image' if image else 'no_image',
            raw_answer=tokenizer.decode(tokens, skip_special_tokens=True).strip(), raw_token_ids=tokens,
            EOS=bool(tokens and tokens[-1] in eos), at_cap=len(tokens)-(int(key=='O' and bool(tokens) and tokens[0]==tokenizer.bos_token_id))>=1024,
            pixels=pixel_info, input_tokens=int(inputs['input_ids'].shape[1]),
            vision_calls=counts['vision']-before['vision'], forwards=counts['forwards']-before['forwards']))
        write(action + '/PROGRESS.json', dict(counts=counts, time=time.time()))
    write(action + '/COMPLETE.json', dict(counts=counts, time=time.time()))


def supervisor():
    recovery = json.loads((ROOT/'RECOVERY_PREGEN.json').read_text()) if os.environ.get('RECOVER_PREGEN')=='1' else None
    manifest=os.environ.get('ADOPT_MANIFEST','ADOPT_DOWNLOAD.json')
    assert manifest in ['ADOPT_DOWNLOAD.json','ADOPT_DOWNLOAD_ACCEL.json']
    adoption = json.loads((ROOT/manifest).read_text()) if os.environ.get('ADOPT_DOWNLOAD')=='1' else None
    once='ADOPTION_ACCEL_ONCE' if manifest=='ADOPT_DOWNLOAD_ACCEL.json' else 'ADOPTION_ONCE'
    assert not (recovery and adoption)
    with (ROOT / ('RECOVERY_PREGEN_ONCE' if recovery else once if adoption else 'ONCE')).open('x'):
        pass
    child = None
    gpu_used = float(recovery['gpu_process_seconds']) if recovery else 0.0
    def stop_child():
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        elif child is None and adoption:
            pid=adoption['download_pid']
            stat=Path('/proc')/str(pid)/'stat'
            def alive():
                if not stat.exists():
                    return False
                fields=stat.read_text().split()
                return fields[21]==str(adoption['download_start_ticks']) and fields[2]!='Z'
            if alive():
                os.killpg(pid,signal.SIGTERM)
                for _ in range(5):
                    if not alive():
                        break
                    time.sleep(1)
                if alive():
                    os.killpg(pid,signal.SIGKILL)
    def guard(begin=None):
        size = sum(p.stat().st_size for r in CFG['owned_roots'] for p in Path(r).rglob('*') if p.is_file() and not p.is_symlink())
        used = gpu_used + (time.time()-begin if begin is not None else 0)
        reason = 'WALL_LIMIT' if time.time() >= CFG['hard_end']-15 else 'STORAGE_LIMIT' if size >= CFG['storage_bytes']-100000000 else 'GPU_LIMIT' if used >= CFG['gpu_seconds']-15 else None
        write('GUARD_HEALTH.json', dict(time=time.time(), bytes=size, gpu_seconds=used))
        if reason:
            write('STOP.json', dict(reason=reason, time=time.time()))
            stop_child()
            raise RuntimeError(reason)
    def run(action, gpu=False):
        nonlocal child, gpu_used
        guard()
        if gpu:
            uuid, free = subprocess.check_output(['nvidia-smi', '-i', str(CFG['gpu']), '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
            assert uuid == CFG['gpu_uuid'] and int(free) >= 45*1024
        start = time.time()
        env = dict(os.environ, TRIAL_ACTION=action)
        if gpu:
            env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        child = None
        launched = False
        try:
            with (ROOT / (action+'.log')).open('x') as log:
                child = subprocess.Popen([CFG['python'], CFG['neutral_entry']], env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                launched = True
                write(action+'/IDENTITY.json', dict(pid=child.pid, start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21], start=start))
                while child.poll() is None:
                    guard(start if gpu else None)
                    time.sleep(5)
        finally:
            if launched:
                stop_child()
                elapsed = time.time()-start
                if gpu:
                    gpu_used += elapsed
                write(action+'/PROCESS_EXIT.json', dict(code=child.returncode, seconds=elapsed,
                    gpu_process_seconds=gpu_used, settled_once=True))
        assert child is not None and child.returncode == 0, action+' failed'
    try:
        if recovery:
            assert recovery['original_start']==CFG['original_start'] and recovery['hard_end']==CFG['hard_end']
            assert 0 <= gpu_used < CFG['gpu_seconds']-15
            prior=json.loads((ROOT/'pregen_incident/GUARD_EXIT.json').read_text())
            assert prior['gpu_process_seconds']==gpu_used
            assert (ROOT/'DOWNLOAD_COMPLETE.json').exists()
            assert not any((ROOT/a/'IDENTITY.json').exists() for a in ['O_D','M_D'])
            assert all(not list((ROOT/a/'outputs').glob('*.json')) for a in ['O_D','M_D'])
            assert not list((ROOT/'pregen_incident/O_D/outputs').glob('*.json'))
        elif adoption:
            assert adoption['gpu_process_seconds']==0
            assert adoption['original_start']==CFG['original_start'] and adoption['hard_end']==CFG['hard_end']
            assert not (ROOT/'O_D/IDENTITY.json').exists() and not (ROOT/'M_D/IDENTITY.json').exists()
            stat = Path('/proc')/str(adoption['download_pid'])/'stat'
            while stat.exists():
                fields=stat.read_text().split()
                assert fields[21]==str(adoption['download_start_ticks']), 'Download PID reused'
                if fields[2]=='Z':
                    break
                guard()
                time.sleep(5)
            assert (ROOT/'DOWNLOAD_COMPLETE.json').exists() and not (ROOT/'download/FAILURE.json').exists()
            write('download/ADOPTED_COMPLETION.json',dict(time=time.time(),original_start=adoption['download_start'],
                seconds=time.time()-adoption['download_start'],gpu_process_seconds=0,exit_code_available=False))
            adoption=None
        else:
            run('download')
        for action in ['O_D', 'M_D']:
            run(action, True)
        write('D_READY_FOR_REVIEW.json', dict(time=time.time(), gpu_process_seconds=gpu_used))
        while not (ROOT/'DECISION_D.json').exists():
            guard()
            time.sleep(5)
        decision = json.loads((ROOT/'DECISION_D.json').read_text())
        selected = decision['selected']
        assert selected in [None, 'O', 'M']
        assert decision['all_64_image_outputs_reviewed']
        if selected:
            assert decision[selected]['gate_pass']
            assert selected == 'O' or not decision['O']['gate_pass']
            run(selected+'_C', True)
        write('COMPLETE.json', dict(status='GENERATIONS_COMPLETE_REVIEW_PENDING' if selected else 'D_CAPABILITY_FAIL_C_SEALED', selected=selected, gpu_process_seconds=gpu_used, time=time.time()))
    except BaseException as exc:
        write('FAILURE.json', dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        stop_child()
        raise
    finally:
        write('GUARD_EXIT.json', dict(time=time.time(), gpu_process_seconds=gpu_used))


if __name__ == '__main__':
    action = os.environ.get('TRIAL_ACTION', 'supervisor')
    try:
        if action == 'supervisor':
            supervisor()
        elif action == 'download':
            if os.environ.get('TRIAL_RANGE_DOWNLOAD')=='1':
                import runpy
                runpy.run_path(str(ROOT/'ranged_download.py'),run_name='__main__')
            else:
                download()
        else:
            worker(action)
    except BaseException as exc:
        write(action+'/FAILURE.json', dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        raise
