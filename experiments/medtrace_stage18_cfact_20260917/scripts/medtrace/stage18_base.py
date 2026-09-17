"""Fresh Base outputs for the frozen DEV source candidates; never trains a writer."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage17_single import setup


def rows_for(packet):
    rows = {}
    for package in packet['candidate_packages']:
        task = package['training']
        for row in [task['native']] + task['H_fit'] + task['G_fit'] + task['U_fit'] + package['evaluation']:
            key = digest([row['image_sha256'], row['question']])
            if key in rows and rows[key]['reference'] != row['reference']:
                raise ValueError('Conflicting source references for the same query')
            rows.setdefault(key, row)
    return rows


def worker(cfg):
    setup(cfg)
    import torch
    from m3bench_repro.editors.llava_runtime import EditorRecord, write_json_atomic
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from scripts.medtrace.stage18_cfact import assert_base_off
    from scripts.medtrace import stage15
    root = Path(cfg['run'])
    packet = read(root/'private/PILOT_CANDIDATES.json')
    if digest(packet) != cfg['packet_binding'] or cfg['mode'] != 'DEV_BASE_ONLY':
        raise ValueError('Base dispatch binding/scope changed')
    rows = rows_for(packet)
    if len(rows) != cfg['N_queries']:
        raise ValueError('Query count changed')
    print('LOADING_RUNTIME', flush=True)
    runtime = load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['cpu_gate'])))
    runtime.run_root = root/'private/work'
    if runtime.generation_config != cfg['generation_lock'] or next(runtime.model.parameters()).dtype != torch.float16:
        raise ValueError('Frozen generation/precision changed')
    runtime.model.eval()
    frozen = [(p, p._version, p.data_ptr()) for p in runtime.model.parameters()]
    outputs = []
    for key, row in rows.items():
        assert_base_off(runtime)
        record = EditorRecord(key, row['dataset'], row['question'], '', '', Path(row['image_path']),
                              row['image_path'], len(outputs), 'BASE_ONLY', 'NO_REPHRASE')
        out, _, _, _ = stage15.base_output(runtime, root, row, record)
        outputs.append(dict(query_id=key, source=row, output=out))
        write_json_atomic(root/'public/PROGRESS.json', dict(status='BASE_GENERATING', completed=len(outputs), N=len(rows)))
        print('BASE_QUERY_COMPLETE', len(outputs), len(rows), flush=True)
    if any(p._version != v or p.data_ptr() != ptr or p.requires_grad for p, v, ptr in frozen):
        raise ValueError('Base changed')
    write_new(root/'private/BASE_OUTPUTS.json', dict(code=cfg['code_commit'], runtime=cfg['runtime_lock'],
        packet_binding=cfg['packet_binding'], scope='DEV_SOURCE_LABEL_BASE_ONLY', records=outputs))
    write_json_atomic(root/'public/PROGRESS.json', dict(status='BASE_COMPLETE_NOT_JUDGED', completed=len(outputs), N=len(rows)))


if __name__ == '__main__':
    worker(read(os.environ['JOB_CONFIG']))
