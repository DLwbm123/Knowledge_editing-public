"""Deploy exactly one fixed training trial into a new data-disk directory."""
import inspect
import json
from pathlib import Path
import subprocess


def remote_trial(payload):
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import tempfile
    import time

    outputs = Path('/data/bmw/Knowledge_editing/outputs')
    preparation = outputs/'vigor-medical-preparation-v1-20261018/run/attempt2'
    parent = outputs/'full-context-summary8-v1-20261018/run'
    root = outputs/'vigor-training-trial1-v1-20261018/run'
    if root.exists():
        raise FileExistsError('Trial already installed; do not replay')
    if json.loads((preparation/'public/RESULT.json').read_text())['status'] != 'COMPLETE':
        raise ValueError('Calibration is not complete')
    mount = json.loads(subprocess.check_output(['findmnt','-J','-T',str(outputs)],text=True))['filesystems'][0]
    if mount['target'] != '/data' or shutil.disk_usage(outputs).free < 9*1024**3:
        raise OSError('Storage reserve unavailable')
    config = json.loads((preparation/'private/config.json').read_text())
    manifest = json.loads((preparation/'private/manifest.json').read_text())
    blocked_groups = set(manifest['calibration_source_groups']) | set(manifest['excluded_source_groups'])
    blocked_ids = set(manifest['calibration_ids']) | set(manifest['excluded_ids'])
    data = Path('/data/bmw/DataP/knowledge_editing/data/m3bench/SLAKE')
    row = next(x for x in sorted(json.loads((data/'train.json').read_text()),key=lambda x:int(x['qid']))
        if x['q_lang']=='en' and 'SLAKE:'+x['img_name'].split('/')[0] not in blocked_groups
        and 'train.SLAKE.'+str(x['qid']) not in blocked_ids)
    image = data/'imgs'/row['img_name']
    if not image.is_file():
        raise FileNotFoundError('Training image missing')
    record = dict(record_id='train.SLAKE.'+str(row['qid']),dataset='SLAKE',question=row['question'],
        target=row['answer'],official_rephrase=row['question'],image_path=str(image),
        relative_image_path=row['img_name'],formal_sequence_position=1,question_type='TRAINING_TRIAL',
        router_positive_source='TRAIN_ONLY_NO_EDIT_LABELS')
    root.mkdir(parents=True)
    for name in ('private','public','logs','implementation'):
        (root/name).mkdir()
    probe = root/'private/.probe'
    probe.write_text('ok')
    if probe.read_text() != 'ok':
        raise OSError('Storage probe failed')
    probe.unlink()
    for name, content in payload['files'].items():
        path = root/'implementation'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content)
    def write(name,value):
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    write('private/record.json',record)
    write('public/DATA_AUDIT.json',dict(training_samples=1,calibration_group_overlap=0,
        official_heldout_and_existing_eval_group_overlap=0,
        selection='qid ascending first English training item on eligible uncalibrated image',
        clinical_patient_id_isolation='NOT_VERIFIABLE_FROM_SOURCE_METADATA'))
    config.update(output=str(root),implementation_commit=payload['commit'],
        calibration=str(preparation/'private/calibration.pt'),manifest=str(preparation/'private/manifest.json'),
        authorization=dict(single_training_edit=True,max_steps=5,tune=False,expand=False,judge=False))
    write('private/config.json',config)
    env = dict(os.environ)
    for line in Path('/data/bmw/env.sh').read_text().splitlines():
        if line.startswith('export '):
            key,value=line[7:].split('=',1);env[key]=value
    previous = json.loads((parent/'private/LAUNCH_ENV.json').read_text())
    for key in ('M3BENCH_MODEL_PATH','M3BENCH_VISION_PATH'):
        env[key]=previous[key]
        if not (Path(env[key])/'config.json').is_file():
            raise FileNotFoundError('Existing model path unavailable')
    env.update(PYTHONPATH=str(root/'implementation')+':'+str(parent/'private/official_llava'),
        CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    entry_root = Path(tempfile.mkdtemp(prefix='p.',dir='/data/bmw/tmp'))
    cpu = entry_root/'c.py'
    cpu.write_text('import json\nfrom experiments.vigor_medical_single_image_v1.cpu_check import run\nprint(json.dumps(run()))\n')
    checked = subprocess.run(['/data/bmw/envs/v0/bin/python',str(cpu)],env=env,text=True,capture_output=True)
    write('private/CPU_RESPONSE.json',dict(returncode=checked.returncode,stdout=checked.stdout,stderr=checked.stderr))
    if checked.returncode:
        raise RuntimeError('CPU trial lifecycle check failed')
    write('public/CPU_CHECK.json',json.loads(checked.stdout.strip().splitlines()[-1]))
    uuid,free = subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'],text=True).strip().split(', ')
    if uuid!='GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca' or int(free)<32*1024:
        raise RuntimeError('GPU peak-memory margin unavailable')
    env.update(CUDA_VISIBLE_DEVICES='5',M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES='5',
        M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES='5',M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuid,
        M3BENCH_LLAVA_SOURCE=str(parent/'private/official_llava'),
        M3BENCH_EXPECTED_LLAVA_SOURCE=str(parent/'private/official_llava'),CONFIG_FILE=str(root/'private/config.json'))
    entry = entry_root/'t.py'
    entry.write_text('import json,os\nfrom experiments.vigor_medical_single_image_v1.trial import main\nmain(json.load(open(os.environ["CONFIG_FILE"])))\n')
    with (root/'logs/worker.log').open('x') as log:
        process = subprocess.Popen(['/data/bmw/envs/v0/bin/python','-u',str(entry)],env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=process.pid,epoch=time.time(),root=str(root),physical_gpu=5,implementation_commit=payload['commit'])
    write('private/LAUNCH.json',receipt)
    return receipt


def deploy():
    root = Path(__file__).resolve().parents[2]
    files = list((root/'experiments/vigor_medical_single_image_v1').glob('*.py'))
    files.append(root/'experiments/medtrace_stage12_v2_20260911/m3bench_repro/editors/llava_runtime.py')
    payload = dict(files={str(p.relative_to(root)):p.read_text() for p in files},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    code=inspect.getsource(remote_trial)+'\nprint(json.dumps(remote_trial(json.loads(sys.stdin.readline()))))'
    return subprocess.run(['ssh','pro5000',"python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
        input=json.dumps(code)+'\n'+json.dumps(payload)+'\n',text=True,capture_output=True)
