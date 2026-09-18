"""Finite Stage20 execution chain, not a scheduled monitor."""
from pathlib import Path
from collections import Counter
import base64,fcntl,json,os,shlex,subprocess,sys,time,urllib.request
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage19_fasttrack_budget import write,remaining
from scripts.medtrace.stage20_closeout import judge,outputs,score_endpoint,report


def publish(root):
    root=Path(root);prefix='experiments/medtrace_stage20_20260918';reports='reports/medtrace_stage20_20260918'
    names=['stage20_prepare','stage20_reference_review','stage20_admit','stage20_runtime','stage20_closeout','stage20_controller','test_stage20','stage19_fasttrack','stage19_fasttrack_budget']
    files={prefix+'/scripts/medtrace/'+n+'.py':(ROOT/'scripts/medtrace'/(n+'.py')).read_bytes() for n in names}
    for p in (root/'public').iterdir():
        if p.suffix not in ('.json','.md','.csv','.png') or p.name in ('CONTROLLER_STATUS.json','PUBLICATION_RECEIPT.json'):continue
        data=p.read_bytes()
        if p.name=='BUDGET_LEDGER.json':
            value=read(p)
            for item in value['judgment_batches']:
                for key in ('runtime_warnings','stderr','error','thread_id'):item.pop(key,None)
            data=(json.dumps(value,indent=2)+'\n').encode()
        if p.suffix!='.png' and any(s in data for s in (b'"raw_answer"',b'"raw_token_ids"',b'"image_path"',b'"source_qid"',b'/Users/',b'/remote-home/',b'/root/',b'"canonical_edit_id"')):raise ValueError('Private content in public artifact '+p.name)
        files[reports+'/'+p.name]=data
    files[prefix+'/README.md']=b'# Stage20 pure-stream extension\n\nApply the previously published Stage15-19 source overlays, then these files. The Stage20 finite controller uses the frozen original training and isolated source-agreement Judge. Run `python -m scripts.medtrace.test_stage20` for CPU checks. Authorized private source manifests, models and images are required and are not distributed. Natural routing, full-precision experts and independent resource ledgers are retained. Final writers, router, W0 and bindings remain private on the authorized data disk; no final-bank cleanup is invoked. No patient-independence or clinical-validation claim. See reports for actual common N, coverage and unfinished items.\n'
    repo='DLwbm123/Knowledge_editing-public';head=json.loads(subprocess.check_output(['gh','api','repos/'+repo+'/commits/main'],text=True))['sha']
    payload=dict(query='mutation($input:CreateCommitOnBranchInput!){createCommitOnBranch(input:$input){commit{oid url}}}',variables=dict(input=dict(branch=dict(repositoryNameWithOwner=repo,branchName='main'),expectedHeadOid=head,message=dict(headline='Deliver bounded Stage20 pure-stream and new-source evidence'),fileChanges=dict(additions=[dict(path=k,contents=base64.b64encode(v).decode()) for k,v in sorted(files.items())]))))
    path=root/'private/PUBLICATION_PAYLOAD.json';write(path,payload)
    response=json.loads(subprocess.check_output(['gh','api','graphql','--input',str(path)],text=True))
    if response.get('errors'):raise RuntimeError(response['errors'])
    receipt=response['data']['createCommitOnBranch']['commit'];sha=receipt['oid']
    if json.loads(subprocess.check_output(['gh','api','repos/'+repo+'/commits/main'],text=True))['sha']!=sha:raise ValueError('Public head verification failed')
    url='https://raw.githubusercontent.com/'+repo+'/'+sha+'/'+reports+'/ADVISOR_UPDATE_ZH.md'
    with urllib.request.urlopen(url,timeout=30) as r:
        if r.status!=200 or r.read()!=files[reports+'/ADVISOR_UPDATE_ZH.md']:raise ValueError('Anonymous artifact verification failed')
    receipt.update(anonymous_access=True,files=len(files),private_material_published=False);write(root/'PUBLICATION_RECEIPT.json',receipt)
    return receipt


def run(cfg):
    from scripts.medtrace.stage20_admit import prepare_base,finalize
    root=Path(cfg['local_run']);p=root/'private';public=root/'public';remote=cfg['remote_run'];status=public/'CONTROLLER_STATUS.json'
    lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',cfg['ssh_socket'],'-p',str(cfg['port']),cfg['host']];transport=shlex.join(ssh[:-1])
    def rpc(cmd):return subprocess.check_output(ssh+['bash -s'],input=cmd+'\n',text=True)
    def push(rel):subprocess.run(['rsync','-a','-e',transport,str(root/rel),cfg['host']+':'+remote+'/'+rel],check=True)
    def pull(rel):
        target=root/rel;target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-a','-e',transport,cfg['host']+':'+remote+'/'+rel,str(target)],check=True)
    def start(phase,basecfg):
        gpu=rpc('nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits -i 0').strip().split(',')
        if gpu[0].strip()!=basecfg['gpu_uuid'] or int(gpu[1])<22000:raise RuntimeError('GPU startup identity/memory guard failed')
        config=dict(basecfg,phase=phase);write_new(p/('SUPERVISOR_'+phase+'.json'),config);push('private/SUPERVISOR_'+phase+'.json')
        launch=remote+'/private/LAUNCH_'+phase
        cmd='set -e\ntest ! -e '+launch+'\ntouch '+launch+'\nJOB_CONFIG='+shlex.quote(remote+'/private/SUPERVISOR_'+phase+'.json')+' JOB_ARGV='+shlex.quote(json.dumps([cfg['remote_code']+'/scripts/medtrace/stage19_fasttrack_budget.py']))+' nohup '+cfg['remote_python']+' '+cfg['remote_entry']+' main > '+remote+'/private/'+phase+'.log 2>&1 < /dev/null &\necho $!'
        pid=int(rpc(cmd).strip());write(p/('LAUNCH_'+phase+'.json'),dict(supervisor_pid=pid,code=config['code_commit']))
        write(status,dict(phase=phase.upper()+'_RUNNING',GPU_started=True,supervisor_pid=pid));return pid
    def exit_ready(phase):return rpc('test ! -f '+remote+'/public/EXIT_'+phase+'.json || echo yes').strip()=='yes'
    def finish(phase):
        pull('public/EXIT_'+phase+'.json');pull('public/GPU_BUDGET_LEDGER.json');pull('private/DISPATCH_'+phase+'.json')
        ledger=read(public/'BUDGET_LEDGER.json');gpu=read(public/'GPU_BUDGET_LEDGER.json');ledger.update(gpu_seconds_used=sum(s['seconds'] for s in gpu['sessions']),gpu_sessions=gpu['sessions']);write(public/'BUDGET_LEDGER.json',ledger)
        if read(public/('EXIT_'+phase+'.json'))['exit_code']!=0:
            pull('private/'+phase+'.log');raise RuntimeError('GPU phase failed; preserved '+phase)
    def pull_small():
        subprocess.run(['rsync','-a','--exclude=BUDGET_LEDGER.json','--exclude=CONTROLLER_STATUS.json','-e',transport,cfg['host']+':'+remote+'/public/',str(public)+'/'],check=True)
        subprocess.run(['rsync','-a','--include=*/','--include=*.json','--include=*.jsonl','--include=BANKS.pt','--exclude=*','-e',transport,cfg['host']+':'+remote+'/private/',str(p)+'/'],check=True)
    if cfg.get('resume_after_train'):
        if read(public/'EXIT_base.json')['exit_code']!=0 or read(public/'EXIT_train.json')['exit_code']!=0:raise ValueError('Completed GPU phases required for closeout recovery')
        previous=read(p/'DISPATCH_train.json');stream=read(p/'STREAM.json')
        if previous['code_commit']!=cfg['worker_config']['code_commit'] or previous['stream_binding']!=digest(stream):raise ValueError('GPU recovery lineage changed')
        finish('train');pull_small();done=read(public/'GENERATED.json');basecfg=read(p/'SUPERVISOR_train.json')
        rows=[r for r in outputs(root) if r['mode']=='endpoint' and r['prefix']==done['N']]
        if read(p/('final'+str(done['N'])+'_SCORE_RECEIPT.json'))['consumer_binding']!=digest(rows):raise ValueError('Scored endpoint changed')
        write(status,dict(phase='REPORT_RECOVERY',N=done['N'],GPU_retraining=False,semantic_rejudging=False))
    else:
        while not all((p/n).exists() for n in ('SOURCE_QUALIFIED_CANDIDATES.json','NEW_SOURCE_PANEL.json','BASE_QUEUE.json')):
            os.kill(int((p/'reference_review.pid').read_text()),0)
            time.sleep(30)
        if not (p/'BASE_QUEUE_SOURCE_REVIEW.json').exists():queue=prepare_base(root)
        else:queue=read(p/'BASE_QUEUE.json')
        gpu=rpc('nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits -i 0').strip().split(',')
        auth=read(public/'RUN_AUTHORIZATION.json')
        if gpu[0].strip()!=auth['gpu_uuid'] or int(gpu[1])<22000:raise RuntimeError('Authorized GPU identity/memory guard failed')
        # Never overwrite a remote ledger after any worker launch, even after a local controller failure.
        if rpc('find '+remote+'/private -maxdepth 1 -name "LAUNCH_*" -print').strip():raise ValueError('Existing launch requires evidence-bound recovery; no ledger reset')
        remote_ledger=rpc('cat '+remote+'/public/GPU_BUDGET_LEDGER.json 2>/dev/null || true').strip()
        if remote_ledger and json.loads(remote_ledger)['sessions']:raise ValueError('Existing GPU charges must be preserved')
        for rel in ('public/RUN_AUTHORIZATION.json','public/GPU_BUDGET_LEDGER.json','private/BASE_QUEUE.json'):push(rel)
        basecfg=dict(cfg['worker_config'],authorization_binding=digest(auth),base_queue_binding=digest(queue['rows']))
        start('base',basecfg)
        while not exit_ready('base'):time.sleep(30)
        finish('base');pull('private/FRESH_BASE_OUTPUTS.json')
        write(status,dict(phase='BASE_SEMANTIC_QUALIFICATION',GPU_started=True))
        judge(root,read(p/'FRESH_BASE_OUTPUTS.json')['records'],'qualification','qualification')
        summary=finalize(root);push('private/STREAM.json');basecfg['stream_binding']=summary['stream_binding']
        start('train',basecfg);scored=set();inserted_scored=0
        while True:
            state=rpc('for n in 011 019 032 050 100; do test ! -f '+remote+'/public/PREFIX_$n.json || echo $n; done; test ! -f '+remote+'/public/EXIT_train.json || echo exit; test ! -f '+remote+'/private/OUTPUTS.jsonl || echo outputs').split()
            if 'outputs' in state:
                pull('private/OUTPUTS.jsonl');rows=[r for r in outputs(root) if r['mode']=='insertion'];pairs=Counter(r['prefix'] for r in rows)
                completed=max((n for n,c in pairs.items() if c==2),default=0)
                if completed>inserted_scored:
                    batch=[r for r in rows if inserted_scored<r['prefix']<=completed];judge(root,batch,f'insertion_{inserted_scored+1}_{completed}');inserted_scored=completed
            for s in state:
                if not s.isdigit() or int(s) in scored:continue
                n=int(s);pull('public/PREFIX_'+s+'.json')
                score_endpoint(root,n,final=n==summary['N']);scored.add(n);report(root)
            if 'exit' in state:break
            time.sleep(45)
        finish('train');pull_small();done=read(public/'GENERATED.json')
    score_endpoint(root,done['N'],final=True);report(root)
    # Resource/support gate uses only saved compute receipts, never scores.
    prereg=read(public/'ABLATION_PREREGISTRATION.json');times=[]
    for i in range(1,12):
        f=p/'edits'/f'e{i:03d}'/'C_FACT/TRAINING.json'
        if f.exists():times.append(read(f)['session_seconds'])
    estimate=2*sum(times)*1.25+240 if len(times)==11 else float('inf')
    recovery=cfg.get('ablation_recovery_config')
    if recovery:
        if read(public/'ABLATION_TOKEN_SUPPORT.json')['EXTRA_full11_supported']:raise ValueError('NO_H-only recovery lacks support exclusion')
        estimate=sum(times)*1.25+240 if len(times)==11 else float('inf')
    available=remaining(read(public/'GPU_BUDGET_LEDGER.json'))
    supported=prereg['all_H_G_structurally_legal'] and done['N']>=11
    execute=supported and estimate<=min(3600,available-600)
    write(public/'ABLATION_RESOURCE_GATE.json',dict(execute=execute,support=supported,estimated_seconds=estimate if len(times)==11 else None,available_seconds=available,cap_seconds=3600,uses_scores=False))
    if execute:
        phase='ablation_noh' if recovery else 'ablation'
        if recovery:basecfg.update(recovery)
        start(phase,basecfg)
        while not exit_ready(phase):time.sleep(45)
        finish(phase);pull_small()
        abdone=read(p/'ablation/public/GENERATED.json')
        write(public/'ABLATION_EXECUTION.json',dict(actual_N=abdone['N'],target_N=11,status='GENERATED_SCORING_PENDING' if abdone['N']==11 else 'RESOURCE_BOUNDARY_INCOMPLETE',final_banks_retained=True))
        if abdone['N']==11:
            score_endpoint(root,11,ablation=True)
            write(public/'ABLATION_EXECUTION.json',dict(actual_N=11,target_N=11,status='SCORED_NO_H_EXTRA_UNSUPPORTED' if recovery else 'SCORED',arms=abdone.get('arms',['A','B']),final_banks_retained=True))
    report(root,final=True)
    # Retain final writers, W0, routing and bindings for the declared consumers.
    consumers=read(public/'CHECKPOINT_CONSUMERS.json');consumers.update(final_banks_retained=True,automatic_final_cleanup=False,main_N=done['N'],ablation_executed=execute);write(public/'CHECKPOINT_CONSUMERS.json',consumers)
    manifest=read(public/'RUN_MANIFEST.json');manifest.update(status='SCORED_REPORTED_PUBLICATION_PENDING',report_code=cfg.get('report_recovery_code',cfg['worker_config']['code_commit']),main_GPU_code=read(p/'DISPATCH_train.json')['code_commit'],ablation_status=read(public/'ABLATION_DEV11.json')['status']);write(public/'RUN_MANIFEST.json',manifest)
    if not read(public/'REPORT_STATUS.json')['all_requested_outputs_scored']:raise ValueError('Unscored main outputs; publication deferred')
    receipt=publish(root);write(status,dict(phase='COMPLETE',N=done['N'],public_commit=receipt['oid'],final_banks_retained=True))


if __name__=='__main__':
    cfg=read(os.environ['JOB_CONFIG'])
    try:run(cfg)
    except Exception as e:
        write(Path(cfg['local_run'])/'public/CONTROLLER_STATUS.json',dict(phase='FAILED_PRESERVED',error=str(e)));raise
