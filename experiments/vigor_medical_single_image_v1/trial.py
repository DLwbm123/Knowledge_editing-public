"""One isolated training edit with fixed calibration and unconditional base restore."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import torch

from .editor import Editor, MAX_STEPS, TARGET_NLL, TRUST_BUDGET, nll, paired_batches
from .prepare import write


def exercise(runtime, calibration, record, root, binding):
    if calibration['selected_layer'] != 'model.layers.0.mlp.down_proj':
        raise ValueError('This trial is fixed to layer zero')
    editor = Editor(runtime, calibration, binding=binding)
    base = editor.state_dict()
    generated = {'base': runtime.generate(record)}
    pair = paired_batches(runtime, record)
    def losses():
        with torch.no_grad():
            return [float(nll(runtime.model, batch)) for batch in pair]
    initial_loss = losses()
    result = dict(status='RUNNING', initial_native_nll=initial_loss)
    checkpoint = root/'private/edited.pt'
    try:
        edited = editor.edit(record, diagnostics=True)
        result['edit'] = edited
        write(root/'public/EDIT.json', edited)
        generated['edited'] = runtime.generate(record)
        state = editor.state_dict()
        torch.save(state, checkpoint)
        checkpoint_bytes = checkpoint.stat().st_size
        editor.load_state_dict(base)
        if not torch.equal(editor.layer.weight.detach().cpu(), base['weight']):
            raise ValueError('Intermediate base restore failed')
        restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
        editor.load_state_dict(restored)
        if not torch.equal(editor.layer.weight.detach().cpu(), state['weight']):
            raise ValueError('Edited native weight reload mismatch')
        generated['reloaded'] = runtime.generate(record)
        reload_loss = losses()
        if generated['reloaded'] != generated['edited'] or reload_loss != edited['native_final_nll']:
            raise ValueError('Edited generation/loss reload mismatch')
        result.update(status='COMPLETE', native_weight_reload='PASS', generation_reload='PASS',
                      loss_reload='PASS', reloaded_native_nll=reload_loss, checkpoint_bytes=checkpoint_bytes)
    finally:
        editor.load_state_dict(base)
        weight_equal = torch.equal(editor.layer.weight.detach().cpu(), base['weight'])
        generated['restored_base'] = runtime.generate(record)
        restored_loss = losses()
        restored = weight_equal and generated['restored_base'] == generated['base'] and restored_loss == initial_loss
        result.update(base_restoration='PASS' if restored else 'FAIL', restored_base_nll=restored_loss)
        write(root/'private/GENERATIONS.json', generated)
        write(root/'public/RESTORATION.json', dict(status=result['base_restoration'],
              native_weight_exact=weight_equal, generation_exact=generated['restored_base']==generated['base'],
              target_losses_exact=restored_loss==initial_loss))
        if not restored:
            raise ValueError('Final base restoration failed')
    # No consumers remain after successful reload and base restoration.
    checkpoint.unlink()
    result['generated_checkpoint_retired'] = True
    normal = lambda text: ' '.join(text.casefold().split()).rstrip('.')
    result['generation_summary'] = {name: dict(token_count=len(value['raw_token_ids']),
        simple_normalized_exact_match=normal(value['decoded_text'])==normal(record.target))
        for name, value in generated.items()}
    result['output_changed'] = generated['base']['raw_token_ids'] != generated['edited']['raw_token_ids']
    return result


def main(config):
    if config['authorization'] != dict(single_training_edit=True, max_steps=5, tune=False,
                                        expand=False, judge=False):
        raise PermissionError('One training edit only')
    if (MAX_STEPS, TARGET_NLL, TRUST_BUDGET) != (5, .01, .001):
        raise ValueError('Frozen constants changed')
    root = Path(config['output'])
    started = time.time()
    with (root/'STARTED.json').open('x') as handle:
        json.dump(dict(pid=os.getpid(), started_epoch=started), handle)
    try:
        sys.path.insert(0, config['source_root'])
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=config['source_root'],text=True).strip()!=config['source_commit']:
            raise ValueError('Runtime source changed')
        from scripts.medtrace.run_realmodel_core import load_real_runtime
        from m3bench_repro.editors.llava_runtime import EditorRecord
        raw = json.loads((root/'private/record.json').read_text())
        manifest = json.loads(Path(config['manifest']).read_text())
        group = 'SLAKE:'+raw['relative_image_path'].split('/')[0]
        if (group in set(manifest['calibration_source_groups']) | set(manifest['excluded_source_groups'])
                or raw['record_id'] in set(manifest['calibration_ids']) | set(manifest['excluded_ids'])):
            raise ValueError('Training trial overlaps calibration or evaluation')
        record = EditorRecord(**dict(raw, image_path=Path(raw['image_path'])))
        calibration = torch.load(config['calibration'], map_location='cpu', weights_only=True)
        write(root/'public/STATUS.json', dict(status='LOADING_MODEL'))
        runtime = load_real_runtime(SimpleNamespace(cpu_gate=Path(config['cpu_gate'])))
        argv = subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True).strip()
        rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        own = [r for r in rows.splitlines() if r.split(',')[0].strip()==str(os.getpid())]
        if len(own)!=1 or any(s in (argv+own[0]).lower() for s in ['wangbomin','knowledge_editing','vigor']):
            raise ValueError('Process name audit failed')
        write(root/'private/PROCESS.json',dict(argv=argv,gpu_process=own[0]))
        write(root/'public/STATUS.json', dict(status='SINGLE_EDIT_RUNNING'))
        result = exercise(runtime, calibration, record, root, config['binding'])
        torch.cuda.synchronize()
        result.update(training_samples=1, layer='model.layers.0.mlp.down_proj', judge_calls=0,
            expanded_formal_experiment=False, elapsed_seconds=time.time()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(root/'public/RESULT.json', result)
        write(root/'public/STATUS.json', dict(status='COMPLETE'))
    except BaseException as error:
        (root/'private/FAILURE.txt').write_text(traceback.format_exc())
        write(root/'public/STATUS.json',dict(status='FAILED',error_type=type(error).__name__))
        raise
    finally:
        write(root/'ENDED.json',dict(pid=os.getpid(),resident_seconds=time.time()-started,ended_epoch=time.time()))
