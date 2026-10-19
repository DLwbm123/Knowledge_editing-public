"""Deploy the authorized A/H finite-response S validation; exclusive run path."""
import inspect
import json
from pathlib import Path
import subprocess


def remote_responseS(payload):
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
    root = outputs/'vigor-responseS-v1-20261019/run'
    batch8 = outputs/'vigor-finite-null-batch8-v1-20261018/run'
    if root.exists():
        raise FileExistsError('Trial already installed; do not replay')
    prior=outputs/'vigor-editperf8-v1-20261019/attempt2'
    if not (prior/'ENDED.json').exists() or json.loads((prior/'public/STATUS.json').read_text())['status']!='GENERATION_COMPLETE':raise ValueError('Formal predecessor not terminal')
    pid=json.loads((prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(pid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Prior process present')
    probe_prior=outputs/'vigor-t0probe-v1-20261019/run'
    if json.loads((probe_prior/'ENDED.json').read_text())['status']!='COMPLETE':raise ValueError('E01 diagnostic predecessor incomplete')
    previous_pid=json.loads((probe_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(previous_pid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Prior diagnostic process present')
    AB_prior=outputs/'vigor-marginab-v1-20261019/run'
    if json.loads((AB_prior/'ENDED.json').read_text())['status']!='COMPLETE':raise ValueError('A/B predecessor incomplete')
    AB_pid=json.loads((AB_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(AB_pid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('A/B predecessor still present')
    C_prior=outputs/'vigor-marginc-v1-20261019/run'
    if json.loads((C_prior/'ENDED.json').read_text())['status']!='STOPPED_INACTIVE_OLD_B':raise ValueError('C predecessor not terminal')
    C_pid=json.loads((C_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(C_pid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('C predecessor still present')
    decomposition=outputs/'vigor-bdecompose-v1-20261019/run'
    if json.loads((decomposition/'ENDED.json').read_text())['status']!='COMPLETE':raise ValueError('Decomposition not terminal')
    dep_pid=json.loads((decomposition/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(dep_pid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Decomposition process present')
    shrink=outputs/'vigor-bshrink-v1-20261019/run'
    if json.loads((shrink/'ENDED.json').read_text())['status']!='STOPPED_NO_ELIGIBLE_SCALE':raise ValueError('Shrink predecessor not terminal')
    shrink_pid=json.loads((shrink/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(shrink_pid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Shrink process present')
    H_prior=outputs/'vigor-hybridH-v1-20261019/run'
    if json.loads((H_prior/'ENDED.json').read_text())['status']!='STOPPED_MAIN_ACTUAL_NLL_FAILED':raise ValueError('H predecessor not terminal')
    H_pid=json.loads((H_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(H_pid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('H process present')
    H_decomposition=outputs/'vigor-hdecompose-v1-20261019/run'
    if json.loads((H_decomposition/'ENDED.json').read_text())['status']!='COMPLETE':raise ValueError('H decomposition not terminal')
    hd_pid=json.loads((H_decomposition/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(hd_pid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('H decomposition process present')
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(outputs)],text=True))['filesystems'][0]
    if mount['target']!='/data' or shutil.disk_usage(outputs).free<9*1024**3:raise OSError('Storage reserve unavailable')
    root.mkdir(parents=True)
    for name in ('private','public','logs','implementation'):(root/name).mkdir()
    probe=root/'private/.probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink()
    for name,content in payload['files'].items():
        path=root/'implementation'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    def write(name,value):(root/name).write_text(json.dumps(value,indent=2)+'\n')
    config=json.loads((prior/'private/config.json').read_text())
    config.update(output=str(root),prior=str(prior),AB_result=str(AB_prior/'public/RESULT.json'),H_result=str(H_prior/'public/RESULT.json'),H_identity=str(H_prior/'private/IDENTITY.json'),implementation_commit=payload['commit'],authorization=payload['authorization'])
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
    cpu.write_text('import json\nfrom experiments.vigor_medical_single_image_v1.responseS_cpu_check import run\nprint(json.dumps(run()))\n')
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
    config['hard_deadline_epoch']=time.time()+900
    config['deadline_epoch']=config['hard_deadline_epoch']-30
    write('private/config.json',config)
    entry = entry_root/'d.py'
    entry.write_text('import json,os\nfrom experiments.vigor_medical_single_image_v1.responseS import main\nmain(json.load(open(os.environ["CONFIG_FILE"])))\n')
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
    from .responseS import AUTH
    payload = dict(authorization=AUTH,files={str(p.relative_to(root)):p.read_text() for p in files},
        frozen_targets=json.loads((root/'reports/vigor_editperf8_v1_20261019/FROZEN_TARGETS.json').read_text()),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    code=inspect.getsource(remote_responseS)+'\nprint(json.dumps(remote_responseS(json.loads(sys.stdin.readline()))))'
    return subprocess.run(['ssh','pro5000',"python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
        input=json.dumps(code)+'\n'+json.dumps(payload)+'\n',text=True,capture_output=True)
