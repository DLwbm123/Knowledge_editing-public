"""Finite real-model compatibility/calibration worker; never calls Editor.edit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import torch

from .editor import calibrate, editable_fp32, gradient_pair, paired_batches


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main(config):
    if config['authorization'] != dict(compatibility=True, calibration=True,
                                        formal_edit=False, judge=False, monitoring=False):
        raise PermissionError('Preparation-only authorization required')
    root = Path(config['output'])
    started = time.time()
    public = root/'public'
    with (root/'STARTED.json').open('x') as handle:
        json.dump(dict(pid=os.getpid(), started_epoch=started), handle)
    completed = []
    def progress(event):
        event = dict(event)
        factor = event.pop('factor', None)
        if factor is not None:
            index = int(event['layer'].split('.')[2])
            torch.save(factor, root/'private'/f'factor_{index:02d}.pt')
            completed.append(event)
            write(public/'LAYER_PROFILE.json', completed)
        if event['stage'] in ('START', 'COMPLETE') or event['completed'] % 8 == 0:
            write(public/'STATUS.json', dict(status='RUNNING', elapsed_seconds=time.time()-started,
                 layers_complete=len(completed), layers_total=32, current=event, formal_updates=0))
    try:
        sys.path.insert(0, config['source_root'])
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=config['source_root'], text=True).strip()
        if actual != config['source_commit']:
            raise ValueError('Frozen runtime source mismatch')
        from scripts.medtrace.run_realmodel_core import load_real_runtime
        from m3bench_repro.editors.llava_runtime import EditorRecord
        raw = json.loads((root/'private/records.json').read_text())
        manifest = json.loads((root/'private/manifest.json').read_text())
        records = [EditorRecord(**dict(row, image_path=Path(row['image_path']))) for row in raw]
        if [r.record_id for r in records] != manifest['calibration_ids'] or len(records) != 32:
            raise ValueError('Frozen training list mismatch')
        if set(manifest['calibration_source_groups']) & set(manifest['excluded_source_groups']):
            raise ValueError('Image-group leakage')
        write(public/'STATUS.json', dict(status='LOADING_MODEL', formal_updates=0))
        runtime = load_real_runtime(SimpleNamespace(cpu_gate=Path(config['cpu_gate'])))
        argv = subprocess.check_output(['ps', '-p', str(os.getpid()), '-o', 'args='], text=True).strip()
        if any(word in argv.lower() for word in ('wangbomin', 'knowledge_editing', 'vigor')):
            raise ValueError('Visible process name is not neutral')
        gpu_processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                                '--format=csv,noheader'], text=True)
        own = [line for line in gpu_processes.splitlines() if line.split(',')[0].strip() == str(os.getpid())]
        if len(own) != 1 or any(word in own[0].lower() for word in ('wangbomin', 'knowledge_editing', 'vigor')):
            raise ValueError('GPU process identity not verified')
        write(root/'private/PROCESS.json', dict(argv=argv, gpu_process=own[0]))
        candidates = [f'model.layers.{i}.mlp.down_proj' for i in range(32)]
        if runtime.model.config.num_hidden_layers != 32:
            raise ValueError('Unexpected backbone depth')
        for name in candidates:
            layer = runtime.get_module(name)
            if tuple(layer.weight.shape) != (4096, 14336) or layer.weight.dtype != torch.float16:
                raise ValueError('Unexpected backbone projection')
        first = records[0]
        base_generation = runtime.generate(first)
        layer = runtime.get_module(candidates[0])
        with editable_fp32(runtime.model, layer, readonly=True):
            pair = paired_batches(runtime, first)
            losses, gradients = gradient_pair(runtime, layer, pair)
            if any(not torch.count_nonzero(g) for g in gradients):
                raise ValueError('Real-model paired gradient is zero')
            compatibility = dict(status='PASS', model='LLaVA-Med-v1.5-Mistral-7B',
                native_dtype='float16', paired_mean_target_nll=losses,
                gradient_shapes=[list(g.shape) for g in gradients],
                image_placeholder_count=pair[0].image_token_count,
                expanded_visual_token_count=pair[0].image_token_end-pair[0].image_token_start,
                generated_token_count=len(base_generation['raw_token_ids']),
                postprocessed_pixel_null=True, native_weight_unchanged=True,
                formal_updates=0, judge_calls=0)
        del pair, gradients
        write(public/'COMPATIBILITY.json', compatibility)
        result = calibrate(runtime, records, candidates, train_ids=manifest['calibration_ids'],
                           excluded_ids=manifest['excluded_ids'], binding=config['binding'], progress=progress)
        artifact = root/'private/calibration.pt'
        torch.save(result, artifact)
        restored = torch.load(artifact, map_location='cpu', weights_only=True)
        if restored['selected_layer'] != result['selected_layer'] or restored['binding'] != result['binding']:
            raise ValueError('Calibration reload metadata mismatch')
        for name in candidates:
            for key in ('a', 'b'):
                if not torch.equal(restored['factors'][name][key], result['factors'][name][key]):
                    raise ValueError('Calibration reload factor mismatch')
            layer = runtime.get_module(name)
            if layer.weight.dtype != torch.float16 or layer._forward_hooks or layer._forward_pre_hooks:
                raise ValueError('Native layer/hook restoration failed')
        final_generation = runtime.generate(first)
        if final_generation != base_generation:
            raise ValueError('Unedited native generation changed after calibration')
        torch.cuda.synchronize()
        write(public/'RESULT.json', dict(status='COMPLETE', selected_layer=result['selected_layer'],
            training_records=32, training_image_groups=32, candidates=32, mean_E=result['mean_E'],
            calibration_reload='PASS', native_generation_parity='PASS', formal_updates=0, judge_calls=0,
            elapsed_seconds=time.time()-started, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            binding=config['binding'], source_commit=config['source_commit']))
        write(public/'STATUS.json', dict(status='COMPLETE', layers_complete=32, formal_updates=0,
                                        elapsed_seconds=time.time()-started))
    except BaseException as error:
        (root/'private/FAILURE.txt').write_text(traceback.format_exc())
        write(public/'STATUS.json', dict(status='FAILED', error_type=type(error).__name__,
            error=str(error) if isinstance(error, ValueError) else 'See private failure receipt',
            layers_complete=len(completed), elapsed_seconds=time.time()-started, formal_updates=0))
        raise
    finally:
        write(root/'ENDED.json', dict(pid=os.getpid(), ended_epoch=time.time(),
                                     resident_seconds=time.time()-started))
