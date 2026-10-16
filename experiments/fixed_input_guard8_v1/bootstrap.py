"""One installation, existing runtime/data, explicit first-eight authorization."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def install(payload):
    root = Path('/data/bmw/Knowledge_editing/outputs/fixed-input-guard8-v1-20261016/run')
    parent = Path('/data/bmw/Knowledge_editing/outputs/purew-odds-hinge-first1-v1-20261016/run')
    assert not root.exists(), 'Inspect existing installation; never replay'
    assert payload['authorization']['execution'] is True
    assert payload['cpu']['status'] == 'PASS'
    gpu = payload['gpu']
    uuid, free = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.free',
                                        '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert uuid == payload['uuid'] and int(free) >= 60000
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(parent)], text=True))['filesystems'][0]
    assert mount['target'] == '/data' and shutil.disk_usage(parent).free >= 24*1024**3
    memory = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    assert memory >= 32*1024**3
    read = lambda p: json.loads(Path(p).read_text())
    tasks = read(parent/'private/BENCHMARK146_QUEUE.json')['tasks']
    ledger = read(parent/'private/EVAL_LEDGER.json')
    assert [t['seed'] for t in tasks[:8]] == payload['protocol']['seeds']
    active = {(t['native']['image_sha256'], t['native']['question']) for t in tasks}
    protected = list(dict.fromkeys(q for t in tasks for e in t['events'] if e['task'] in ('T1L', 'T2L')
                                  for q in e['all_probe_query_ids'] if ledger['Base_correctness'][q]
                                  and (ledger['queries'][q]['image_sha256'], ledger['queries'][q]['question']) not in active))
    assert len(protected) == 54
    for task in tasks[:8]:
        assert all(sum(len(e['all_probe_query_ids']) for e in task['events'] if e['task'] == name) <= 4 for name in ('T1G', 'T2G'))
    excluded = {q['source_group'] for q in ledger['queries'].values()}
    data = Path('/data/bmw/DataP/knowledge_editing/data/m3bench/SLAKE')
    train = read(data/'train.json')
    records, groups = [], []
    for row in sorted(train, key=lambda x: int(x['qid'])):
        group = 'SLAKE:'+row['img_name'].split('/')[0]
        if row['q_lang'] != 'en' or group in excluded or group in groups:
            continue
        image = data/'imgs'/row['img_name']
        assert image.is_file()
        records.append(dict(record_id='train.SLAKE.'+str(row['qid']), dataset='SLAKE', question=row['question'],
                            target=row['answer'], official_rephrase=row['question'], image_path=str(image),
                            relative_image_path=row['img_name'], formal_sequence_position=len(records)+1,
                            question_type='CALIBRATION_TRAIN', router_positive_source='TRAIN_ONLY_NO_EDIT_LABELS'))
        groups.append(group)
        if len(records) == 32:
            break
    assert len(records) == 32 and not set(groups) & excluded
    manifest = dict(role='CALIBRATION_TRAIN', calibration_ids=[r['record_id'] for r in records],
                    calibration_source_groups=groups, edit_and_eval_ids=list(ledger['queries']),
                    edit_and_eval_source_groups=sorted(excluded), selection='qid ascending, first English item per disjoint training image')
    source_commit = read(parent/'private/SOURCE_COMMIT.json')['commit']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=parent/'private/source', text=True).strip() == source_commit
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=parent/'private/source', text=True).strip()
    root.mkdir(parents=True)
    def write(path, value):
        path = root/path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2)+'\n')
    for path in ('logs', 'private/control', 'private/failures', 'private/implementation', 'public'):
        (root/path).mkdir(parents=True, exist_ok=True)
    probe = root/'private/.probe'
    probe.write_bytes(b'guard8')
    assert probe.read_bytes() == b'guard8'
    probe.unlink()
    for name in ('source', 'official_llava', 'cpu_gate', 'learner_gate', 'derived_inputs'):
        (root/'private'/name).symlink_to((parent/'private'/name).resolve(), target_is_directory=True)
    for name in ('BENCHMARK146_QUEUE.json', 'EVAL_LEDGER.json', 'EVAL_BINDINGS.json', 'SOURCE_COMMIT.json'):
        shutil.copy2(parent/'private'/name, root/'private'/name)
    for name, content in payload['files'].items():
        assert '..' not in Path(name).parts and not Path(name).is_absolute()
        path = root/'private/implementation'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        compile(content, name, 'exec')
        path.write_text(content)
    write('private/calibration_records.json', records)
    write('private/protection_ids.json', protected)
    write('private/calibration_manifest.json', manifest)
    write('public/PROTOCOL.json', payload['protocol'])
    write('public/EXECUTION_AUTHORIZATION.json', payload['authorization'])
    write('public/CPU_VALIDATION.json', payload['cpu'])
    write('public/DATA_ADMISSION.json', dict(status='PASS', training_records=32, distinct_training_source_groups=32,
                                           no_eval_group_overlap=True, protection_count=54, original_order_and_seeds=True))
    write('public/STORAGE_ADMISSION.json', dict(status='PASS', mount=mount, free_bytes=shutil.disk_usage(root).free,
                                              host_available_bytes=memory, read_write_probe=True))
    now = time.time()
    write('RUN_MANIFEST.json', dict(starting_epoch=now, deadline_epoch=now+12*3600, GPU_seconds_limit=6*3600,
                                   min_free_bytes=8*1024**3, owned_weight_limit_bytes=3*1024**3))
    write('STAGE_CAP.json', dict(GPU_seconds_limit=6*3600))
    write('RESOURCE_LEDGER.json', dict(gpu_seconds_used=0, gpu_sessions=[], Judge_attempts=0))
    write('PLAN_CONFIG.json', dict(hardware=dict(allowed_GPUs=[gpu], UUIDs={str(gpu):uuid})))
    write('private/config.json', dict(version='FIXED_INPUT_GUARD8_V1_20261016', authorization=payload['authorization'],
                                    admission={k:'PASS' for k in ('calibration_disjoint','source_order','storage','hardware','runtime_binding')},
                                    calibration_manifest=manifest, calibration_inputs=str(root/'private/calibration_inputs.pt'),
                                    layer='model.layers.21.mlp.down_proj', native_dtype='float16',
                                    physical_GPU=gpu, GPU_UUID=uuid, source_root=str(root/'private/source'), source_commit=source_commit,
                                    implementation_commit=payload['commit'], cpu_gate=str(root/'private/learner_gate'),
                                    source_image_root='/data/bmw/DataP/knowledge_editing/data/m3bench'))
    temporary = Path(tempfile.mkdtemp(prefix='e.', dir='/data/bmw/tmp'))
    entry = temporary/'e.py'
    entry.write_text("import runpy\nrunpy.run_module('experiments.fixed_input_guard8_v1.launch',run_name='__main__')\n")
    env = dict(os.environ, **read(parent/'private/LAUNCH_ENV.json'))
    env.update(RUN_ROOT=str(root), TMPDIR=str(temporary), ENTRY_MODE='controller', NEUTRAL_ENTRY=str(entry),
               PYTHONPATH=str(root/'private/implementation')+':'+str(root/'private/official_llava')+':'+str(root/'private/source'),
               CUDA_VISIBLE_DEVICES=str(gpu), M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(gpu),
               M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=str(gpu), M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuid,
               M3BENCH_LLAVA_SOURCE=str(root/'private/official_llava'), M3BENCH_EXPECTED_LLAVA_SOURCE=str(root/'private/official_llava'),
               ACTION='controller', ARM='CONTROLLER', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
               OPENBLAS_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1')
    write('private/LAUNCH_ENV.json', env)
    with (root/'logs/controller.log').open('ab') as log:
        process = subprocess.Popen([env['TRAIN_PYTHON'], '-u', str(entry)], env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    argv = subprocess.check_output(['ps','-ww','-p',str(process.pid),'-o','args='], text=True).strip()
    assert not any(x in argv for x in ('wangbomin','Knowledge_editing','purew','guard'))
    startup = dict(pid=process.pid, argv=argv, epoch=now, physical_GPU=gpu, GPU_UUID=uuid,
                   run_root=str(root), commit=payload['commit'], stage='CALIBRATION_PENDING')
    write('private/START_CONTROLLER.json', startup)
    return startup


if __name__ == '__main__':
    print(json.dumps(install(json.loads(sys.stdin.readline()))))
