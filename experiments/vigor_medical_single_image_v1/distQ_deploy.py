"""Single frozen offline distribution-sketch qualification deployment."""
import ast,inspect,json,subprocess
from pathlib import Path

def remote_launch(p):
 import os,json,time,tempfile,shutil,subprocess
 from pathlib import Path
 outputs=Path('/data/bmw/Knowledge_editing/outputs');old=outputs/'vigor-revision3-v1-20261020';master=outputs/'vigor-factF3-v1-20261020';root=outputs/'vigor-distributionQ-v1-20261020'
 if root.exists():raise FileExistsError('Already installed, no replay')
 if time.time()>=p['clock']['start_epoch']+600:raise TimeoutError('Insufficient original GPU work-clock reserve')
 mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(outputs)],text=True))['filesystems'][0]
 if mount['target']!='/data' or shutil.disk_usage(outputs).free<9*1024**3:raise OSError('Storage reserve')
 gpu=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
 if gpu[0]!='GPU-4b56ab28-a018-652c-b8f6-2d9dc3dc45d2' or int(gpu[1])<48*1024:raise RuntimeError('GPU2 memory margin')
 host_available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 if host_available<32*1024**3:raise MemoryError('Host memory admission')
 if sum(len(s.encode()) for s in p['files'].values())>10485760:raise OSError('Transfer ceiling')
 root.mkdir(parents=True)
 for n in ['private','public','logs','implementation']:(root/n).mkdir()
 def wr(n,v):(root/n).write_text(json.dumps(v,indent=2)+'\n')
 probe=root/'private/.probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink()
 wr('private/STORAGE_BASELINE.json',dict(root_existed=False,baseline_bytes=0,free_bytes=shutil.disk_usage(outputs).free,root_hard_limit=251658240,reserved_other_bytes=16777216,estimated_peak=268435456,host_available_bytes=host_available,GPU_free_MiB=int(gpu[1]),mount_target=mount['target']))
 for name,s in p['files'].items():
  q=root/'implementation'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(s)
 c=json.loads((master/'run/private/config.json').read_text());c.update(output=str(root),master=str(master),teacher_prior=str(outputs/'vigor-ftmq3-v1-20261020'),teacher_locality_prior=str(outputs/'vigor-teacherLocality3-v1-20261020'),authorization=p['authorization'],implementation_commit=p['commit'])
 env=dict(os.environ)
 for line in Path('/data/bmw/env.sh').read_text().splitlines():
  if line.startswith('export '):k,v=line[7:].split('=',1);env[k]=v
 parent=outputs/'full-context-summary8-v1-20261018/run';prev=json.loads((parent/'private/LAUNCH_ENV.json').read_text())
 for k in ['M3BENCH_MODEL_PATH','M3BENCH_VISION_PATH']:env[k]=prev[k];assert (Path(env[k])/'config.json').is_file()
 official=str(parent/'private/official_llava')
 env.update(PYTHONPATH=str(root/'implementation')+':'+official,CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='2',M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES='2',M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES='2',M3BENCH_FORMAL_EXPECTED_GPU_UUID=gpu[0],M3BENCH_LLAVA_SOURCE=official,M3BENCH_EXPECTED_LLAVA_SOURCE=official,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CONFIG_FILE=str(root/'private/config.json'))
 tmp=Path(tempfile.mkdtemp(prefix='p.',dir='/data/bmw/tmp'));cpu=tmp/'c.py';cpu.write_text('import json\nfrom experiments.vigor_medical_single_image_v1.distQ_ops import run_checks\nprint(json.dumps(run_checks()))\n')
 start=p['clock']['start_epoch'];wr('EXPERIMENT_CLOCK.json',p['clock'])
 checked=subprocess.run(['/data/bmw/envs/v0/bin/python',str(cpu)],env=env,capture_output=True,text=True,timeout=60)
 wr('private/CPU_RESPONSE.json',dict(returncode=checked.returncode,stdout=checked.stdout,stderr=checked.stderr))
 if checked.returncode:raise RuntimeError('CPU check failed')
 wr('public/CPU_CHECK.json',json.loads(checked.stdout.strip().splitlines()[-1]))
 c['hard_deadline_epoch']=start+14100;c['kill_deadline_epoch']=start+14400;c['deadline_epoch']=start+13800
 wr('private/config.json',c)
 entry=tmp/'d.py';entry.write_text('import os,json\nfrom experiments.vigor_medical_single_image_v1.distQ import main\nmain(json.load(open(os.environ["CONFIG_FILE"])))\n')
 with (root/'logs/worker.log').open('x') as f:proc=subprocess.Popen(['/data/bmw/envs/v0/bin/python','-u',str(entry)],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 receipt=dict(pid=proc.pid,epoch=time.time(),root=str(root),physical_gpu=2,implementation_commit=p['commit'],hard_deadline_epoch=c['hard_deadline_epoch'])
 wr('private/LAUNCH.json',receipt)
 watch=tmp/'w.py';watch.write_text('import os,time,signal\nfrom pathlib import Path\npid='+str(proc.pid)+'\np=Path("/proc/"+str(pid)+"/stat")\ntry: born=p.read_text().split()[21]\nexcept FileNotFoundError: raise SystemExit\ntime.sleep(max(0,'+repr(c['hard_deadline_epoch'])+'-time.time()))\ntry:\n if p.read_text().split()[21]==born: os.killpg(pid,signal.SIGTERM)\nexcept (ProcessLookupError,FileNotFoundError): pass\ntime.sleep(300)\ntry:\n if p.read_text().split()[21]==born: os.killpg(pid,signal.SIGKILL)\nexcept (ProcessLookupError,FileNotFoundError): pass\n')
 with (root/'logs/watchdog.log').open('x') as f:w=subprocess.Popen(['/data/bmw/envs/v0/bin/python',str(watch)],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
 wr('private/WATCHDOG.json',dict(pid=w.pid,deadline_epoch=c['hard_deadline_epoch'],process_group=proc.pid))
 receipt['argv']=subprocess.check_output(['ps','-p',str(proc.pid),'-o','pid=,args='],text=True).strip();time.sleep(8);receipt['gpu_snapshot']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 wr('private/STARTUP_CHECK.json',receipt);return receipt

def deploy():
 if json.loads((Path(__file__).resolve().parents[2]/'reports/vigor_distribution_static_v1_20261020/PUBLICATION.json').read_text())['anonymous_http']!=200:raise RuntimeError('Static audit must be delivered first')
 r=Path(__file__).resolve().parents[2];tree=ast.parse((r/'experiments/vigor_medical_single_image_v1/distQ_ops.py').read_text());node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='AUTH' for t in n.targets));auth=eval(compile(ast.Expression(node),'<AUTH>','eval'),{'MODULE':'model.layers.6.mlp.down_proj'})
 files=list((r/'experiments/vigor_medical_single_image_v1').glob('*.py'));files.append(r/'experiments/medtrace_stage12_v2_20260911/m3bench_repro/editors/llava_runtime.py')
 payload=dict(authorization=auth,files={str(x.relative_to(r)):x.read_text() for x in files},clock=json.loads((r/'.local_run/distQ_clock.json').read_text()),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip())
 code=inspect.getsource(remote_launch)+'\nprint(json.dumps(remote_launch(json.loads(sys.stdin.readline()))))'
 return subprocess.run(['ssh','pro5000',"python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=json.dumps(code)+'\n'+json.dumps(payload)+'\n',capture_output=True,text=True,timeout=90)
if __name__=='__main__':
 p=deploy();print(p.stdout);print(p.stderr);raise SystemExit(p.returncode)
