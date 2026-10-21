"""Deploy S03 A8 and one-shot D fifth-step NLL alignment."""
import inspect
import json
from pathlib import Path
import subprocess


def remote_editperf8_resume(payload):
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
    root = outputs/'vigor-editperf8-v1-20261019/attempt2'
    batch8 = outputs/'vigor-finite-null-batch8-v1-20261018/run'
    if root.exists():
        raise FileExistsError('Trial already installed; do not replay')
    previous_root=outputs/'vigor-editperf8-v1-20261019/run'
    ended=json.loads((previous_root/'ENDED.json').read_text())
    prior_launch=json.loads((previous_root/'private/LAUNCH.json').read_text())
    assert ended['generation_attempts']==5 and ended['model_forward_calls']==197
    assert json.loads((previous_root/'public/STATUS.json').read_text())['status']=='FAILED'
    if subprocess.run(['ps','-p',str(prior_launch['pid'])],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Prior process present')
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(outputs)],text=True))['filesystems'][0]
    if mount['target']!='/data' or shutil.disk_usage(outputs).free<9*1024**3:raise OSError('Storage reserve unavailable')
    root.mkdir(parents=True)
    for name in ('private','public','logs','implementation'):(root/name).mkdir()
    probe=root/'private/.probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink()
    for name,content in payload['files'].items():
        path=root/'implementation'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    def write(name,value):(root/name).write_text(json.dumps(value,indent=2)+'\n')
    for name in ('MANIFEST.json','BINDINGS.json'):shutil.copy2(previous_root/'private'/name,root/'private'/name)
    for name in ('PANEL.json','DATA_AUDIT.json'):shutil.copy2(previous_root/'public'/name,root/'public'/name)
    for phase,aliases in {'base':['Q001','Q002','Q003'],'base_repeat':['Q001','Q002']}.items():
        for alias in aliases:
            src=previous_root/'private/outputs'/phase/(alias+'.json')
            assert json.loads(src.read_text())['status']=='COMPLETE'
            dst=root/'private/outputs'/phase/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst);shutil.copy2(src.with_suffix('.STARTED.json'),dst.with_suffix('.STARTED.json'))
    config=json.loads((previous_root/'private/config.json').read_text())
    config.update(output=str(root),implementation_commit=payload['commit'],authorization=payload['authorization'],
        original_launch_epoch=prior_launch['epoch'],prior_generation_attempts=5,prior_model_forward_calls=197,
        prior_gpu_seconds=ended['resident_seconds'],paired_current_base=True)
    write('public/RECOVERY.json',dict(user_approved=True,reused_generation_attempts=5,prior_gpu_seconds=ended['resident_seconds'],
        prior_model_forward_calls=197,one_Q003_repeat_only=True,judge_payload_cap=246,generation_cap=250))
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
    cpu.write_text('import json\nfrom experiments.vigor_medical_single_image_v1.editperf8_cpu_check import run\nprint(json.dumps(run()))\n')
    cpu_started=time.time()
    checked = subprocess.run(['/data/bmw/envs/v0/bin/python',str(cpu)],env=env,text=True,capture_output=True)
    cpu_elapsed=time.time()-cpu_started
    write('private/CPU_RESPONSE.json',dict(returncode=checked.returncode,stdout=checked.stdout,stderr=checked.stderr,elapsed_seconds=cpu_elapsed))
    if checked.returncode:
        raise RuntimeError('CPU trial lifecycle check failed')
    write('public/CPU_CHECK.json',dict(json.loads(checked.stdout.strip().splitlines()[-1]),elapsed_seconds=cpu_elapsed))
    uuid,free = subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'],text=True).strip().split(', ')
    if uuid!='GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca' or int(free)<48*1024:
        raise RuntimeError('GPU peak-memory margin unavailable')
    env.update(CUDA_VISIBLE_DEVICES='5',M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES='5',
        M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES='5',M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuid,
        M3BENCH_LLAVA_SOURCE=str(parent/'private/official_llava'),
        M3BENCH_EXPECTED_LLAVA_SOURCE=str(parent/'private/official_llava'),CONFIG_FILE=str(root/'private/config.json'))
    config.pop('trajectory_steps',None)
    config['hard_deadline_epoch']=min(time.time()+3600-config['prior_gpu_seconds'],config['original_launch_epoch']+10800)
    config['deadline_epoch']=config['hard_deadline_epoch']-120
    write('private/config.json',config)
    entry = entry_root/'d.py'
    entry.write_text('import json,os\nfrom experiments.vigor_medical_single_image_v1.editperf8_resume import main\nmain(json.load(open(os.environ["CONFIG_FILE"])))\n')
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
    from .editperf8_resume import AUTH
    payload = dict(authorization=AUTH,files={str(p.relative_to(root)):p.read_text() for p in files},
        frozen_targets=json.loads((root/'reports/vigor_editperf8_v1_20261019/FROZEN_TARGETS.json').read_text()),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    code=inspect.getsource(remote_editperf8_resume)+'\nprint(json.dumps(remote_editperf8_resume(json.loads(sys.stdin.readline()))))'
    return subprocess.run(['ssh','pro5000',"python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
        input=json.dumps(code)+'\n'+json.dumps(payload)+'\n',text=True,capture_output=True)
