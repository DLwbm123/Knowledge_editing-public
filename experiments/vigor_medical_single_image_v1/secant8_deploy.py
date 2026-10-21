"""Deploy the frozen three-point precision diagnostic into a new data-disk directory."""
import inspect
import json
from pathlib import Path
import subprocess


def remote_secant8(payload):
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
    root = outputs/'vigor-secant8-v1-20261018/run'
    batch8 = outputs/'vigor-amplitude-stress16-v1-20261018/run'
    if root.exists():
        raise FileExistsError('Trial already installed; do not replay')
    if json.loads((preparation/'public/RESULT.json').read_text())['status'] != 'COMPLETE':
        raise ValueError('Calibration is not complete')
    mount = json.loads(subprocess.check_output(['findmnt','-J','-T',str(outputs)],text=True))['filesystems'][0]
    if mount['target'] != '/data' or shutil.disk_usage(outputs).free < 9*1024**3:
        raise OSError('Storage reserve unavailable')
    config = json.loads((batch8/'private/config.json').read_text())
    if json.loads((batch8/'public/RESULT.json').read_text())['status']!='COMPLETE':
        raise ValueError('Original batch incomplete')
    if time.time()>=payload['deadline_epoch']:raise TimeoutError('Authorization expired')
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
    config.update(output=str(root),implementation_commit=payload['commit'],batch8_root=str(batch8),
        deadline_epoch=payload['deadline_epoch'],resident_budget_seconds=900,
        authorization=dict(points=[['S01',6],['S02',11],['S03',9],['S04',6],['S05',8],['S06',5],['S07',5],['S08',7]],extra_steps=0,extra_corrections=0,
            tune=False,replace=False,generation=False,judge=False,variant='quantized_secant_one_shot',candidates_per_point=1))
    write('public/DATA_AUDIT.json',dict(status='FROZEN',frozen_epoch=time.time(),points=config['authorization']['points'],
        source='vigor_amplitude_stress16_v1_20261018',new_samples=0,replacement_allowed=False))
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
        CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    entry_root = Path(tempfile.mkdtemp(prefix='p.',dir='/data/bmw/tmp'))
    cpu = entry_root/'c.py'
    cpu.write_text('import json\nfrom experiments.vigor_medical_single_image_v1.secant8_cpu_check import run\nprint(json.dumps(run()))\n')
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
    entry = entry_root/'d.py'
    entry.write_text('import json,os\nfrom experiments.vigor_medical_single_image_v1.secant8 import main\nmain(json.load(open(os.environ["CONFIG_FILE"])))\n')
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
    payload = dict(deadline_epoch=1792324140,files={str(p.relative_to(root)):p.read_text() for p in files},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    code=inspect.getsource(remote_secant8)+'\nprint(json.dumps(remote_secant8(json.loads(sys.stdin.readline()))))'
    return subprocess.run(['ssh','pro5000',"python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
        input=json.dumps(code)+'\n'+json.dumps(payload)+'\n',text=True,capture_output=True)
