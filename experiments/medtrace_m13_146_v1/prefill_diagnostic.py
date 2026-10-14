"""One bounded first-proposal diagnostic; never trains, corrects, or scores."""
import copy
import os
from pathlib import Path
import sys
import time
import traceback
import torch
import runtime146 as worker
sys.path.insert(0, str(Path(__file__).parent))
import trajectory
import native_fmin as native

r, c, RUN = worker.r, worker.c, worker.RUN


def backend():
    return {key: getattr(torch.backends.cuda, key + '_sdp_enabled')()
            for key in ('flash', 'mem_efficient', 'cudnn', 'math')}


def main():
    diagnostic = int(os.environ.get('PREFILL_DIAGNOSTIC_ID', '1'))
    assert diagnostic in (1, 2)
    gate = c.read(RUN/'private'/f'PREFILL_DIAGNOSTIC_ADMISSION{diagnostic}.json')
    owner, gpu = diagnostic + 2, diagnostic - 1
    assert gate['status'] == 'PASS' and gate['owner'] == owner and gate['GPU'] == gpu
    assert gate['max_GPU_seconds'] == 900 and gate['max_backwards'] == 4
    assert gate['max_non_generation_forwards'] == 24 and gate['scientific_updates'] == 0
    assert c.read(RUN/'private/SELECTED_RECOVERY_EXIT1.json')['exit_code'] == 1
    assert all(v.get('ended_epoch') for v in c.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions'])
    assert c.read(RUN/'private/PREFILL_POLICY_CPU_DIAGNOSIS1.json')['status'] == 'PASS'
    with (RUN/'private'/f'PREFILL_DIAGNOSTIC_ONCE{diagnostic}').open('x'):
        pass
    worker.configure()
    native.g.SOURCE_WEIGHTS = (.5, .5)
    started = time.time()
    counted = r.count

    def count(*args):
        assert time.time() - started < gate['max_GPU_seconds']
        counted(*args)
        assert r.COUNTS['non_generation_forwards'] <= gate['max_non_generation_forwards']

    r.count = count
    task = worker.tasks()[owner-1]
    rows = trajectory.source_rows(worker.packet(), owner)
    prior = c.read(RUN/'private/selected_recovery_original1'/f'NATIVE_FMIN_FIRST_STEP_REAL_{owner}.json')
    reference = c.read(RUN/'private/results/C'/f'{owner}.json')
    initial_path = RUN/'private/resume146'/f'FISHER_MIN_INITIAL_{owner}.pt'
    assert initial_path.is_file() and not initial_path.is_symlink()
    saved = torch.load(initial_path, map_location='cpu', weights_only=False)
    assert saved['step'] == saved['scalar']['step'] == 0
    assert saved['scalar']['second_moments'] == [0., 0.]
    assert saved['binding']['seed'] == task['seed']
    from m3bench_repro.editors.llava_runtime import seed_everything
    report = dict(status='DIAGNOSTIC_RUNNING', source=gate['source_commit'],
                  scientific_updates=0, corrections=0, generation=0, Judge=0, variants=[])
    with r.p.lease(gpu):
        runtime, bindings, handle = r.load(gpu)
        handle.remove()
        handle = runtime.model.register_forward_pre_hook(lambda *args: r.count(*args))
        try:
            tower = runtime.llava_model().get_vision_tower().vision_tower
            report['vision_attention_classes'] = sorted({type(layer.self_attn).__name__
                for layer in tower.vision_model.encoder.layers})
            batches_cpu = []
            for name, flags in [('fresh', gate['fresh_prefill_backend']),
                                ('persistent_math', gate['persistent_prefill_backend'])]:
                for key, value in flags.items():
                    getattr(torch.backends.cuda, 'enable_' + key + '_sdp')(value)
                runtime.model.half()
                assert backend() == flags
                prepared = r.s.prepare(runtime, bindings, task, rows, True)
                batches_cpu.append([copy.deepcopy(value[1]) for value in prepared])
                r.old.fp32(runtime)
                batches = [r.old.batch_on(runtime, value[1]) for value in prepared]
                seed_everything(task['seed'])
                initial = r.p.tr.TT4(task['seed'], 8, 8).state_dict()
                base = r.mm.canonicalize(initial)[0]
                assert all(torch.equal(base[key], saved['state'][key]) for key in base)
                expert = r.p.expert(base, task['seed'], runtime.device)
                hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER), expert)
                hook.attach()
                try:
                    r.q.logits(runtime, hook, batches[0], verify_base=True)
                    before = r.old.snapshot(expert)
                    _, losses = trajectory.source_gradients(r, runtime, hook, expert, batches, 1)
                    norm = torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.)
                    assert torch.isfinite(norm)
                    pure, scalar = r.mm.canonical_rms_step(before,
                        {key: value.grad.detach() for key, value in expert.named_parameters()},
                        dict(step=0, second_moments=[0., 0.]))
                    pure = native.g.canonical_match(before, pure, reference['curve'][0]['operator_step_norm'])
                    gain = native.g.gain(losses, r.losses_at(runtime, hook, expert, pure,
                        native.g.source_batches(batches, 1)))
                    report['variants'].append(dict(prefill=name, backend=flags, pure_gain=gain,
                        old_pure_gain=prior['pure_gain'], exact_old_pure_gain=gain == prior['pure_gain'],
                        source_RMS_diagnostic_clock=scalar['step']))
                    c.write(RUN/'private'/f'PREFILL_DIAGNOSTIC_RESULT{diagnostic}.json', report)
                finally:
                    hook.detach()
            differences = []
            for first, second in zip(*batches_cpu):
                a, b = first[0]['inputs_embeds'], second[0]['inputs_embeds']
                differences.append(dict(equal=torch.equal(a, b),
                    changed_values=int(torch.count_nonzero(a != b)), max_abs=float((a.float()-b.float()).abs().max())))
            report.update(status='DIAGNOSTIC_COMPLETE', prefill_differences=differences,
                counts=dict(r.COUNTS), epoch=time.time(), no_training_resume=True)
            assert r.COUNTS['updates'] == r.COUNTS['generation_attempts'] == 0
            assert r.COUNTS['backwards'] == gate['max_backwards']
            c.write(RUN/'private'/f'PREFILL_DIAGNOSTIC_RESULT{diagnostic}.json', report)
        finally:
            handle.remove()
            r.save_counts()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException as error:
        code = 1
        c.write(RUN/'private'/f'PREFILL_DIAGNOSTIC_FAILURE{os.environ.get("PREFILL_DIAGNOSTIC_ID", "1")}.json',
            dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        c.write(RUN/'private'/f'PREFILL_DIAGNOSTIC_EXIT{os.environ.get("PREFILL_DIAGNOSTIC_ID", "1")}.json',
                dict(exit_code=code, epoch=time.time()))
