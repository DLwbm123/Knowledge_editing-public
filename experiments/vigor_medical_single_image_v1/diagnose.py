"""Four preregistered single-step diagnostics, no generation or Judge."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .editor import damp, direction, editable_fp32, finite, gradient_pair, nll, paired_batches, update_metrics
from .prepare import write

DETERMINISTIC_ANCHOR = dict(visual_nll=4.921504020690918, null_nll=4.466132164001465,
                            E=65214837377.33204)

class DegenerateCandidateError(ValueError):
    """A scientific non-updatable direction, distinct from a lifecycle failure."""


FIRST_D = 4.911504020690918
CASES = (('quarter', 0, 'original', FIRST_D/4),
         ('sixteenth', 0, 'original', FIRST_D/16),
         ('fisher_sum', 0, 'sum_target', FIRST_D/16),
         ('layer5', 5, 'original', FIRST_D/16))


def length_audit(runtime, records):
    lengths, positions = [], []
    for record in records:
        batch = runtime.build_edit_batch(record)
        lengths.append(int((batch.labels[:, 1:] != -100).sum()))
        positions.append(int(batch.attention_mask.sum()) if batch.attention_mask is not None else batch.labels.numel())
    if min(lengths) <= 0:
        raise ValueError('Empty target')
    return lengths, positions


def sum_target_factor(runtime, records, original, lengths, positions):
    layer = runtime.get_module('model.layers.0.mlp.down_proj')
    moment = torch.zeros(layer.weight.shape[0], dtype=torch.float64, device=layer.weight.device)
    with editable_fp32(runtime.model, layer, readonly=True):
        for record, target_count, position_count in zip(records, lengths, positions):
            batch = runtime.build_edit_batch(record)
            if int((batch.labels[:, 1:] != -100).sum()) != target_count:
                raise ValueError('Target length changed')
            captured = []
            hook = layer.register_forward_hook(lambda m, args, out: captured.append(out), prepend=True)
            try:
                loss = nll(runtime.model, batch)
            finally:
                hook.remove()
            if len(captured) != 1:
                raise ValueError('FFN invocation count mismatch')
            gradient = torch.autograd.grad(loss, captured[0])[0].detach().double()*target_count
            valid = batch.attention_mask.bool() if batch.attention_mask is not None else torch.ones(
                gradient.shape[:-1], dtype=torch.bool, device=gradient.device)
            if int(valid.sum()) != position_count:
                raise ValueError('Position count changed')
            moment += finite(gradient[valid]).square().sum(0)
    count = sum(positions)
    if count != original['positions']:
        raise ValueError('Calibration position denominator changed')
    return dict(a=original['a'].clone(), b=damp((moment/count).float()).cpu(), positions=count)


def candidate(runtime, record, layer_index, factor, requested, historical=None, *, retain=False, solve_factor=None):
    layer = runtime.get_module(f'model.layers.{layer_index}.mlp.down_proj')
    if layer.weight.dtype != torch.float16:
        raise ValueError('Diagnostic comparison is frozen to native FP16')
    original = layer.weight.detach().clone()
    pair = paired_batches(runtime, record)
    def losses():
        with torch.no_grad():
            return [float(nll(runtime.model, x)) for x in pair]
    native_before = losses()
    a, b = (factor[k].to(layer.weight.device) for k in ('a','b'))
    completed = False
    try:
        with editable_fp32(runtime.model, layer):
            fp_before, gradients = gradient_pair(runtime, layer, pair)
            tangent, geometry = direction(*gradients, a, b, diagonal=solve_factor)
            if tangent is None:
                raise DegenerateCandidateError('Degenerate candidate; no replacement permitted')
            if historical is not None:
                if (abs(fp_before[0]-historical['visual_nll']) > 1e-5
                        or abs(fp_before[1]-historical['null_nll']) > 1e-5
                        or abs(geometry['E']/historical['E']-1) > 1e-5):
                    raise ValueError(f'Frozen deterministic anchor mismatch: current={fp_before}, E={geometry["E"]}; historical={[historical["visual_nll"],historical["null_nll"]]}, E={historical["E"]}')
            if solve_factor is None:
                effective = min(requested, math.sqrt(2*.001*geometry['E']))
            else:
                # The new solve metric never replaces the original trust ruler.
                unit_cost = update_metrics(tangent, *gradients, a, b)['fisher_cost']
                effective = min(requested, geometry['E']*math.sqrt(.001/unit_cost))
            delta = finite(-effective/geometry['E']*tangent)
            with torch.no_grad():
                layer.weight.add_(delta)
            actual = layer.weight.detach()-original.float()
            native_delta = layer.weight.detach().half().float()-original.float()
            rounding = float(torch.linalg.vector_norm(native_delta-actual))
            norm = float(torch.linalg.vector_norm(actual))
            fp_metrics = update_metrics(actual, *gradients, a, b)
            native_metrics = update_metrics(native_delta, *gradients, a, b)
            quantization = dict(absolute_rounding_l2=rounding,relative_rounding_l2=rounding/norm if norm else None,
                delta_cosine=float(F.cosine_similarity(actual.flatten(),native_delta.flatten(),dim=0)),
                changed_elements=int(torch.count_nonzero(native_delta)),
                fp32_delta_l2=norm,native_delta_l2=float(torch.linalg.vector_norm(native_delta)))
            fp_after = losses()
        native_after = losses()
        completed = True
        return dict(requested_reduction=requested,effective_reduction=effective,clipped=effective<requested,
            E=geometry['E'],native_before=native_before,native_after=native_after,
            fp32_before=fp_before,fp32_after=fp_after,fp32_update=fp_metrics,native_update=native_metrics,
            quantization=quantization,native_gain=native_before[0]-native_after[0],
            native_gain_ratio=(native_before[0]-native_after[0])/effective,
            native_null_change=native_after[1]-native_before[1],
            fp32_gain_ratio=(fp_before[0]-fp_after[0])/effective,
            fp32_null_change=fp_after[1]-fp_before[1])
    finally:
        if not retain or not completed:
            with torch.no_grad():
                layer.weight.copy_(original)
            if not torch.equal(layer.weight,original) or losses()!=native_before:
                raise ValueError('Candidate base restoration failed')
        if layer._forward_hooks or layer._forward_pre_hooks:
            raise ValueError('Candidate hooks were not removed')


def main(config):
    if config['authorization'] != dict(candidates=4,tune=False,expand=False,generation=False,judge=False):
        raise PermissionError('Only four fixed diagnostic candidates authorized')
    # Fix stochastic CUDA backward accumulation; keep the historical anchor gate unchanged.
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Deterministic CUDA workspace must be configured before loading the model')
    torch.use_deterministic_algorithms(True)
    root = Path(config['output'])
    started = time.time()
    with (root/'STARTED.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),epoch=started),f)
    try:
        sys.path.insert(0,config['source_root'])
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=config['source_root'],text=True).strip()!=config['source_commit']:
            raise ValueError('Runtime binding mismatch')
        from scripts.medtrace.run_realmodel_core import load_real_runtime
        from m3bench_repro.editors.llava_runtime import EditorRecord
        read = lambda p:json.loads(Path(p).read_text())
        raw = read(config['record'])
        record = EditorRecord(**dict(raw,image_path=Path(raw['image_path'])))
        records = [EditorRecord(**dict(r,image_path=Path(r['image_path']))) for r in read(config['records'])]
        manifest = read(config['manifest'])
        if [r.record_id for r in records]!=manifest['calibration_ids'] or len(records)!=32:
            raise ValueError('Calibration list changed')
        group = 'SLAKE:'+raw['relative_image_path'].split('/')[0]
        if group in set(manifest['calibration_source_groups'])|set(manifest['excluded_source_groups']):
            raise ValueError('Diagnostic sample overlap')
        calibration = torch.load(config['calibration'],map_location='cpu',weights_only=True)
        if calibration['binding']!=config['binding'] or calibration['selected_layer']!='model.layers.0.mlp.down_proj':
            raise ValueError('Fisher binding mismatch')
        write(root/'public/NUMERICS.json',dict(deterministic_algorithms=True,
            cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],historical_E_relative_tolerance=1e-5))
        runtime = load_real_runtime(SimpleNamespace(cpu_gate=Path(config['cpu_gate'])))
        def forbidden(*args,**kwargs):
            raise PermissionError('Generation is forbidden in this diagnosis')
        runtime.generate = forbidden
        argv = subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True).strip()
        rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        own=[r for r in rows.splitlines() if r.split(',')[0].strip()==str(os.getpid())]
        if len(own)!=1 or any(w in (argv+own[0]).lower() for w in ['wangbomin','knowledge_editing','vigor']):
            raise ValueError('Process identity not neutral')
        lengths,positions = length_audit(runtime,records)
        histogram={str(n):lengths.count(n) for n in sorted(set(lengths))}
        write(root/'public/LENGTHS.json',dict(samples=32,histogram=histogram,min=min(lengths),
            max=max(lengths),mean=sum(lengths)/32,positions=sum(positions),constant_length=len(histogram)==1))
        original = calibration['factors']['model.layers.0.mlp.down_proj']
        alternative = sum_target_factor(runtime,records,original,lengths,positions)
        torch.save(dict(factor=alternative,binding=config['binding'],variant='sum_target',lengths=lengths),
                   root/'private/sum_target_factor.pt')
        ratio=alternative['b'].double()/original['b'].double()
        write(root/'public/FACTOR_CHANGE.json',dict(input_factor_unchanged=torch.equal(alternative['a'],original['a']),
            output_factor_ratio_min=float(ratio.min()),output_factor_ratio_median=float(ratio.median()),
            output_factor_ratio_max=float(ratio.max())))
        history=dict(DETERMINISTIC_ANCHOR)
        write(root/'public/BASELINE.json',dict(deterministic_anchor=history,
            historical_reference=read(config['historical_result'])['edit']['steps'][0],
            E_relative_tolerance=1e-5,NLL_absolute_tolerance=1e-5,
            historical_reference_is_acceptance_gate=False))
        results=[]
        for name,index,variant,requested in CASES:
            write(root/'public/STATUS.json',dict(status='RUNNING',candidate=name,complete=len(results)))
            factor=alternative if variant=='sum_target' else calibration['factors'][f'model.layers.{index}.mlp.down_proj']
            value=candidate(runtime,record,index,factor,requested,history if index==0 and variant=='original' else None)
            results.append(dict(name=name,layer=index,fisher=variant,base_restored=True,**value))
            write(root/'public/CANDIDATES.json',results)
        write(root/'public/RESULT.json',dict(status='COMPLETE',candidates=results,training_samples=1,
            generation_calls=0,judge_calls=0,elapsed_seconds=time.time()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),all_base_restored=True))
        write(root/'public/STATUS.json',dict(status='COMPLETE',complete=4))
    except BaseException as e:
        (root/'private/FAILURE.txt').write_text(traceback.format_exc())
        write(root/'public/STATUS.json',dict(status='FAILED',error_type=type(e).__name__))
        raise
    finally:
        write(root/'ENDED.json',dict(pid=os.getpid(),epoch=time.time(),resident_seconds=time.time()-started))
