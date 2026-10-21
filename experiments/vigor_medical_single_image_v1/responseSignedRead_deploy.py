"""Deploy the authorized A/H finite-response S validation; exclusive run path."""
import inspect
import json
from pathlib import Path
import subprocess


def remote_responseSignedRead(payload):
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import tempfile
    import time

    if time.time()>=1792467136:raise TimeoutError('12-hour window ended')
    if sum(len(v.encode()) for v in payload['files'].values())>10485760:raise OSError('Source transfer ceiling')
    outputs = Path('/data/bmw/Knowledge_editing/outputs')
    preparation = outputs/'vigor-medical-preparation-v1-20261018/run/attempt2'
    parent = outputs/'full-context-summary8-v1-20261018/run'
    root = outputs/'vigor-responseSignedRead-v1-20261020/run'
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
    last=outputs/'vigor-responseS-v1-20261019/run'
    if json.loads((last/'ENDED.json').read_text())['status']!='STOPPED_QUANTIZED_MODEL_CONSTRAINT_FAILED':raise ValueError('S not terminal')
    last_pid=json.loads((last/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(last_pid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('S process present')
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(outputs)],text=True))['filesystems'][0]
    if mount['target']!='/data' or shutil.disk_usage(outputs).free<9*1024**3:raise OSError('Storage reserve unavailable')
    R_prior=outputs/'vigor-responseReserve-v1-20261019/run'
    if json.loads((R_prior/'ENDED.json').read_text())['status']!='STOPPED_MAIN_ACTUAL_NLL_FAILED':raise ValueError('R anchor not terminal')
    rpid=json.loads((R_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(rpid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('R process present')
    R3_prior=outputs/'vigor-response3-v1-20261019/run'
    if json.loads((R3_prior/'ENDED.json').read_text())['status']!='COMPLETE':raise ValueError('R3 predecessor not complete')
    pid3=json.loads((R3_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(pid3)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('R3 process present')
    Follow_prior=outputs/'vigor-responseFollow-v1-20261020/run'
    if json.loads((Follow_prior/'ENDED.json').read_text())['status']!='STOPPED_MAIN_ACTUAL_NLL_FAILED':raise ValueError('Follow not terminal')
    follow_pid=json.loads((Follow_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(follow_pid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Follow process present')
    refresh_prior=outputs/'vigor-responseRefresh-v1-20261020/run'
    if json.loads((refresh_prior/'ENDED.json').read_text())['status']!='STOPPED_MAIN_ACTUAL_NLL_FAILED':raise ValueError('Refresh not terminal')
    refresh_pid=json.loads((refresh_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(refresh_pid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Refresh process present')
    N_prior=outputs/'vigor-responseNcontrol-v1-20261020/run'
    if json.loads((N_prior/'ENDED.json').read_text())['status']!='STOPPED_MAIN_ACTUAL_NLL_FAILED':raise ValueError('N control not terminal')
    npid=json.loads((N_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(npid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('N control process present')
    Quad_prior=outputs/'vigor-responseQuad-v1-20261020/run'
    Center_prior=outputs/'vigor-responseCenter-v1-20261020/run'
    for prior0,expected in [(Quad_prior,'STOPPED_QUADRATIC_SOLVE_FAILED'),(Center_prior,'COMPLETE')]:
        if json.loads((prior0/'ENDED.json').read_text())['status']!=expected:raise ValueError('Transfer predecessor not terminal')
        pid0=json.loads((prior0/'private/LAUNCH.json').read_text())['pid']
        if subprocess.run(['ps','-p',str(pid0)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Transfer predecessor process present')
    Path_prior=outputs/'vigor-responsePath-v1-20261020/run'
    if json.loads((Path_prior/'ENDED.json').read_text())['status']!='STOPPED_TRANSFER_MATCHED_PREDICTION_FAILED':raise ValueError('Path predecessor not terminal')
    ppid=json.loads((Path_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(ppid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Path predecessor process present')
    Native_prior=outputs/'vigor-responseNative-v1-20261020/run'
    if json.loads((Native_prior/'ENDED.json').read_text())['status']!='COMPLETE':raise ValueError('Native predecessor not complete')
    npid=json.loads((Native_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(npid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Native process present')
    Endpoint_prior=outputs/'vigor-responseEndpoint-v1-20261020/run'
    if json.loads((Endpoint_prior/'ENDED.json').read_text())['status']!='COMPLETE':raise ValueError('Endpoint diagnosis not complete')
    epid=json.loads((Endpoint_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(epid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Endpoint process present')
    Nullstep_prior=outputs/'vigor-responseNullstep-v1-20261020/run'
    if json.loads((Nullstep_prior/'ENDED.json').read_text())['status']!='COMPLETE':raise ValueError('Nullstep predecessor not complete')
    spid=json.loads((Nullstep_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(spid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Nullstep process present')
    Small_prior=outputs/'vigor-responseSmallDirection-v1-20261020/run'
    Quant_prior=outputs/'vigor-responseQuantReserve-v1-20261020/run'
    for old_prior in [Small_prior,Quant_prior]:
        if json.loads((old_prior/'ENDED.json').read_text())['status']!='STOPPED_SMALL_NATIVE_ADMISSION_FAILED':raise ValueError('Native subset predecessor not terminal')
        opid=json.loads((old_prior/'private/LAUNCH.json').read_text())['pid']
        if subprocess.run(['ps','-p',str(opid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Predecessor process present')
    Orthant_prior=outputs/'vigor-responseNativeOrthant-v1-20261020/run'
    if json.loads((Orthant_prior/'ENDED.json').read_text())['status']!='COMPLETE':raise ValueError('Small orthant predecessor not complete')
    rpid=json.loads((Orthant_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(rpid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Small orthant process present')
    Radius_prior=outputs/'vigor-responseOrthantRadius-v1-20261020/run'
    if json.loads((Radius_prior/'ENDED.json').read_text())['status']!='STOPPED_ORTHANT_NATIVE_ADMISSION_FAILED':raise ValueError('Radius predecessor not terminal')
    rrpid=json.loads((Radius_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(rrpid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Radius process present')
    Balance_prior=outputs/'vigor-responseOrthantBalance-v1-20261020/run'
    if json.loads((Balance_prior/'ENDED.json').read_text())['status']!='COMPLETE':raise ValueError('Balanced predecessor not complete')
    bpid=json.loads((Balance_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(bpid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Balanced process present')
    Refresh_prior=outputs/'vigor-responseBalancedRefresh-v1-20261020/run'
    if json.loads((Refresh_prior/'ENDED.json').read_text())['status']!='STOPPED_BALANCE_NO_NEGATIVE_EXCURSION':raise ValueError('Refresh predecessor not terminal')
    fpid=json.loads((Refresh_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(fpid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Refresh process present')
    Signed_prior=outputs/'vigor-responseSignedBalance-v1-20261020/run'
    if json.loads((Signed_prior/'ENDED.json').read_text())['status']!='STOPPED_MAIN_ACTUAL_NLL_FAILED':raise ValueError('Signed predecessor not terminal')
    spid=json.loads((Signed_prior/'private/LAUNCH.json').read_text())['pid']
    if subprocess.run(['ps','-p',str(spid)],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Signed predecessor present')
    root.mkdir(parents=True)
    for name in ('private','public','logs','implementation'):(root/name).mkdir()
    probe=root/'private/.probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink()
    for name,content in payload['files'].items():
        path=root/'implementation'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    def write(name,value):(root/name).write_text(json.dumps(value,indent=2)+'\n')
    config=json.loads((prior/'private/config.json').read_text())
    config.update(SignedBalance_result=str(Signed_prior/'public/RESULT.json'),BalancedRefresh_result=str(Refresh_prior/'public/RESULT.json'),Balance_result=str(Balance_prior/'public/RESULT.json'),OrthantRadius_result=str(Radius_prior/'public/RESULT.json'),radius_scale=payload['radius_scale'],NativeOrthant_result=str(Orthant_prior/'public/RESULT.json'),SmallDirection_result=str(Small_prior/'public/RESULT.json'),QuantReserve_result=str(Quant_prior/'public/RESULT.json'),Nullstep_result=str(Nullstep_prior/'public/RESULT.json'),nullstep_scale=payload['nullstep_scale'],Endpoint_result=str(Endpoint_prior/'public/RESULT.json'),Native_result=str(Native_prior/'public/RESULT.json'),PreviousPath_result=str(Path_prior/'public/RESULT.json'),transfer_cpu=payload['transfer_cpu'],Center_result=str(Center_prior/'public/RESULT.json'),Quad_result=str(Quad_prior/'public/RESULT.json'),N_result=str(N_prior/'public/RESULT.json'),Refresh_result=str(refresh_prior/'public/RESULT.json'),Follow_result=str(Follow_prior/'public/RESULT.json'),R3_result=str(R3_prior/'public/RESULT.json'),R_result=str(R_prior/'public/RESULT.json'),output=str(root),prior=str(prior),AB_result=str(AB_prior/'public/RESULT.json'),H_result=str(H_prior/'public/RESULT.json'),H_identity=str(H_prior/'private/IDENTITY.json'),implementation_commit=payload['commit'],authorization=payload['authorization'])
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
    cpu.write_text('import json\nfrom experiments.vigor_medical_single_image_v1.responseSignedRead_cpu_check import run\nprint(json.dumps(run()))\n')
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
    entry.write_text('import json,os\nfrom experiments.vigor_medical_single_image_v1.responseSignedRead import main\nmain(json.load(open(os.environ["CONFIG_FILE"])))\n')
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
    from .responseSignedRead import AUTH
    payload = dict(radius_scale=json.loads((root/'reports/vigor_responseOrthantRadius_v1_20261020/FROZEN_SCALE.json').read_text()),nullstep_scale=json.loads((root/'reports/vigor_responseNullstep_v1_20261020/FROZEN_SCALE.json').read_text()),transfer_cpu=json.loads((root/'reports/vigor_responsePath_v1_20261020/CPU_TRANSFER_REPRESENTATION.json').read_text()),authorization=AUTH,files={str(p.relative_to(root)):p.read_text() for p in files},
        frozen_targets=json.loads((root/'reports/vigor_editperf8_v1_20261019/FROZEN_TARGETS.json').read_text()),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    code=inspect.getsource(remote_responseSignedRead)+'\nprint(json.dumps(remote_responseSignedRead(json.loads(sys.stdin.readline()))))'
    return subprocess.run(['ssh','pro5000',"python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
        input=json.dumps(code)+'\n'+json.dumps(payload)+'\n',text=True,capture_output=True)
