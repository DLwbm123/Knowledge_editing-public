"""Explicit preparation deployment, using the existing data-disk environment."""
import inspect
import json
from pathlib import Path
import subprocess


def remote_install(payload):
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import tempfile
    import time

    parent = Path('/data/bmw/Knowledge_editing/outputs/full-context-summary8-v1-20261018/run')
    root = parent.parent.parent/'vigor-medical-preparation-v1-20261018/run'
    if root.exists():
        raise FileExistsError('Preparation already installed; inspect, never replay')
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(parent)], text=True))['filesystems'][0]
    if mount['target'] != '/data' or shutil.disk_usage(parent).free < 8*1024**3:
        raise OSError('Data disk unavailable or reserve insufficient')
    old = json.loads((parent/'private/config.json').read_text())
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=old['source_root'], text=True).strip()
    if source_commit != old['source_commit'] or subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=old['source_root'], text=True).strip():
        raise ValueError('Frozen runtime checkout changed')
    manifest_old = old['calibration_manifest']
    data = Path(old['source_image_root'])/'SLAKE'
    train = json.loads((data/'train.json').read_text())
    excluded = set(manifest_old['edit_and_eval_source_groups'])
    excluded_ids = set(manifest_old['edit_and_eval_ids'])
    overlaps = {}
    for filename in ('validation.json', 'test.json'):
        rows = json.loads((data/filename).read_text())
        groups = {'SLAKE:'+x['img_name'].split('/')[0] for x in rows}
        overlaps[filename] = len(groups & set(manifest_old['calibration_source_groups']))
        excluded.update(groups)
        excluded_ids.update('train.SLAKE.'+str(x['qid']) for x in rows)
    records, groups = [], []
    for row in sorted(train, key=lambda x: int(x['qid'])):
        group = 'SLAKE:'+row['img_name'].split('/')[0]
        rid = 'train.SLAKE.'+str(row['qid'])
        if row['q_lang'] != 'en' or group in excluded or group in groups or rid in excluded_ids:
            continue
        image = data/'imgs'/row['img_name']
        if not image.is_file():
            raise FileNotFoundError('Selected training image unavailable')
        groups.append(group)
        records.append(dict(record_id=rid, dataset='SLAKE', question=row['question'], target=row['answer'],
            official_rephrase=row['question'], image_path=str(image), relative_image_path=row['img_name'],
            formal_sequence_position=len(records)+1, question_type='CALIBRATION_TRAIN',
            router_positive_source='TRAIN_ONLY_NO_EDIT_LABELS'))
        if len(records) == 32:
            break
    if len(records) != 32 or set(groups) & excluded:
        raise ValueError('Insufficient isolated training images')
    manifest = dict(calibration_ids=[r['record_id'] for r in records], calibration_source_groups=groups,
        excluded_ids=sorted(excluded_ids), excluded_source_groups=sorted(excluded),
        selection='qid ascending; first English item per image group; exclude official val/test and existing edit/eval')
    root.mkdir(parents=True)
    for name in ('private', 'public', 'logs', 'implementation'):
        (root/name).mkdir()
    probe = root/'private/.probe'
    probe.write_text('ok')
    if probe.read_text() != 'ok':
        raise OSError('Storage probe failed')
    probe.unlink()
    for name, content in payload['files'].items():
        path = root/'implementation'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    def write(path, value):
        (root/path).write_text(json.dumps(value, indent=2)+'\n')
    write('private/records.json', records)
    write('private/manifest.json', manifest)
    write('public/DATA_AUDIT.json', dict(training_records=32, distinct_image_groups=32,
        image_group_overlap=0, official_splits_checked=['train','validation','test'],
        existing_eval_ids=len(manifest_old['edit_and_eval_ids']),
        previous_calibration_official_overlap=overlaps, previous_calibration_reused=False,
        selection=manifest['selection'], patient_id_audit='NOT_AVAILABLE_IN_SOURCE_METADATA'))
    env = dict(os.environ)
    for line in Path('/data/bmw/env.sh').read_text().splitlines():
        if line.startswith('export '):
            key, value = line[7:].split('=', 1)
            env[key] = value
    entry_root = Path(tempfile.mkdtemp(prefix='p.', dir='/data/bmw/tmp'))
    env.update(PYTHONPATH=str(root/'implementation'), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1')
    cpu = entry_root/'c.py'
    cpu.write_text('import json\nfrom experiments.vigor_medical_single_image_v1.cpu_check import run\nprint(json.dumps(run()))\n')
    checked = subprocess.run(['/data/bmw/envs/v0/bin/python', str(cpu)], env=env, capture_output=True, text=True)
    write('private/CPU_RESPONSE.json', dict(returncode=checked.returncode, stdout=checked.stdout, stderr=checked.stderr))
    if checked.returncode:
        raise RuntimeError('Remote CPU check failed; no model launched')
    write('public/CPU_CHECK.json', json.loads(checked.stdout.strip().splitlines()[-1]))
    gpu = 5
    uuid, free = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    if uuid != 'GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca' or int(free) < 48*1024:
        raise RuntimeError('Selected GPU lacks the preparation peak-memory margin')
    lock = json.loads((Path(old['cpu_gate'])/'locks/CANONICAL_LLVAMED_RUNTIME_LOCK.json').read_text())
    config = dict(output=str(root), source_root=old['source_root'], source_commit=source_commit,
        cpu_gate=old['cpu_gate'], implementation_commit=payload['commit'],
        authorization=dict(compatibility=True, calibration=True, formal_edit=False, judge=False, monitoring=False),
        binding=dict(model_revision=lock['model_snapshot_sha'], vision_revision=lock['vision_snapshot_sha'],
                     preprocessing_revision=lock['llava_med_code_commit'],
                     dataset_revision='SLAKE-existing-train-image-disjoint-v1',
                     calibration_revision='vigor-medical-preparation-v1-20261018'))
    write('private/config.json', config)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(gpu),
        M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=str(gpu), M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuid,
        M3BENCH_LLAVA_SOURCE=str(parent/'private/official_llava'),
        M3BENCH_EXPECTED_LLAVA_SOURCE=str(parent/'private/official_llava'), CONFIG_FILE=str(root/'private/config.json'))
    entry = entry_root/'r.py'
    entry.write_text('import json,os\nfrom experiments.vigor_medical_single_image_v1.prepare import main\n'
                     'main(json.load(open(os.environ["CONFIG_FILE"])))\n')
    with (root/'logs/worker.log').open('x') as log:
        process = subprocess.Popen(['/data/bmw/envs/v0/bin/python', '-u', str(entry)], env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=process.pid, epoch=time.time(), physical_gpu=gpu, gpu_uuid=uuid,
                   free_mib_at_launch=int(free), root=str(root), implementation_commit=payload['commit'])
    write('private/LAUNCH.json', receipt)
    return receipt


def deploy():
    root = Path(__file__).resolve().parents[2]
    files = list((root/'experiments/vigor_medical_single_image_v1').glob('*.py'))
    files.append(root/'experiments/medtrace_stage12_v2_20260911/m3bench_repro/editors/llava_runtime.py')
    payload = dict(files={str(p.relative_to(root)): p.read_text() for p in files},
                   commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip())
    code = inspect.getsource(remote_install)+'\nprint(json.dumps(remote_install(json.loads(sys.stdin.readline()))))'
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', 'pro5000',
        "python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
        input=json.dumps(code)+'\n'+json.dumps(payload)+'\n', text=True, capture_output=True)
