"""Deploy S03 A8 and one-shot D fifth-step NLL alignment."""
import inspect
import json
from pathlib import Path
import subprocess


def remote_editperf8(payload):
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
    root = outputs/'vigor-editperf8-v1-20261019/run'
    batch8 = outputs/'vigor-finite-null-batch8-v1-20261018/run'
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
    config.update(output=str(root),implementation_commit=payload['commit'],
        deadline_epoch=time.time()+900,count_model_calls=True,
        amplitude_compensation='secant',historical_batch=str(batch8/'public/RESULT.json'),
        authorization=payload['authorization'],
        old_result=str(outputs/'vigor-backoff16-v1-20261018/run/public/RESULT.json'),
        new_result=str(outputs/'vigor-direct-fisher16-v1-20261018/run/public/RESULT.json'),
        new_fisher_summary=str(outputs/'vigor-direct-fisher16-v1-20261018/run/public/DIRECT_FISHER.json'),
        normpath_result=str(outputs/'vigor-normpath8-v1-20261019/run/public/RESULT.json'),
        normd_result=str(outputs/'vigor-normd8-v1-20261019/run/public/RESULT.json'),
        nllmatch_result=str(outputs/'vigor-nllmatch-v1-20261019/run/public/RESULT.json'),
        nllbisect_result=str(outputs/'vigor-nllbisect-v1-20261019/run/public/RESULT.json'),
        frozen_targets=payload['frozen_targets'])
    local=json.loads((outputs/'vigor-direct-fisher16-v1-20261018/run/public/RESULT.json').read_text())
    if local['status']!='COMPLETE' or len(local['samples'])!=8 or local['passed']!=8:
        raise ValueError('Local secant predecessor incomplete')
    # Freeze formal benchmark first eight, explicitly authorized replacement cohort.
    import importlib.util
    spec=importlib.util.spec_from_file_location('panel_plan',root/'implementation/experiments/vigor_medical_single_image_v1/editperf8_plan.py')
    planner=importlib.util.module_from_spec(spec);spec.loader.exec_module(planner)
    tasks=json.loads((parent/'private/BENCHMARK146_QUEUE.json').read_text())['tasks'][:8]
    ledger=json.loads((parent/'private/EVAL_LEDGER.json').read_text());bindings=json.loads((parent/'private/EVAL_BINDINGS.json').read_text())
    panels,aliases,raw_counts,eligible_counts=planner.build(tasks,ledger)
    def mapped(path):
        if '/derived_inputs/' in path:return str(parent/'private/derived_inputs'/path.split('/derived_inputs/',1)[1])
        if '/DataP/' in path:return str(Path('/data/bmw/DataP')/path.split('/DataP/',1)[1])
        return path
    queries={alias:dict(ledger['queries'][qid],image_path=mapped(ledger['queries'][qid]['image_path'])) for qid,alias in aliases.items()}
    selected={x['opaque_Base_id']:bindings[x['opaque_Base_id']] for x in queries.values()}
    for i,t in enumerate(tasks):
        n=t['native'];panels[i]['record']=dict(record_id=t['edit_id'],dataset=n['dataset'],question=n['question'],target=n['reference'],official_rephrase=t['fit_questions'][0],image_path=mapped(n['image_path']),relative_image_path=n['original_image_path'],formal_sequence_position=i+1,question_type='FORMAL_NATIVE_SOURCE',router_positive_source='SOURCE_ONLY')
    assert all(Path(q['image_path']).is_file() for q in queries.values())
    assert raw_counts==dict(T0=8,T1G=32,T1L=49,T2G=32,T2L=2) and eligible_counts==dict(T0=8,T1G=29,T1L=0,T2G=32,T2L=0)
    write('private/MANIFEST.json',dict(tasks=panels,queries=queries,Base_correctness={aliases[k]:ledger['Base_correctness'][k] for k in aliases}))
    write('private/BINDINGS.json',selected)
    write('public/PANEL.json',dict(tasks=[dict(alias=t['alias'],panel=t['panel']) for t in panels],raw_counts=raw_counts,eligible_counts=eligible_counts,unique_base_inputs=len(queries),post_consumers=sum(len(t['panel']) for t in panels),cohort='FORMAL_BENCHMARK_FIRST8_NOT_DIAGNOSTIC_S01_S08'))
    audit=json.loads((batch8/'public/DATA_AUDIT.json').read_text())
    audit=dict(status='PASS',cohort='user_authorized_formal_benchmark_first8',samples=8,diagnostic_batch_not_reused=True,evaluation_targets_not_used_for_training=True)
    write('public/DATA_AUDIT.json',audit)
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
    config['hard_deadline_epoch']=time.time()+3600
    config['deadline_epoch']=config['hard_deadline_epoch']-120
    write('private/config.json',config)
    entry = entry_root/'d.py'
    entry.write_text('import json,os\nfrom experiments.vigor_medical_single_image_v1.editperf8 import main\nmain(json.load(open(os.environ["CONFIG_FILE"])))\n')
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
    from .editperf8 import AUTH
    payload = dict(authorization=AUTH,files={str(p.relative_to(root)):p.read_text() for p in files},
        frozen_targets=json.loads((root/'reports/vigor_editperf8_v1_20261019/FROZEN_TARGETS.json').read_text()),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    code=inspect.getsource(remote_editperf8)+'\nprint(json.dumps(remote_editperf8(json.loads(sys.stdin.readline()))))'
    return subprocess.run(['ssh','pro5000',"python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
        input=json.dumps(code)+'\n'+json.dumps(payload)+'\n',text=True,capture_output=True)
