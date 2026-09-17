"""Local phase controller: detached GPU jobs, isolated Judge, paired closeout, source-only publication."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage19_fasttrack_budget import write


def publish(root):
    root=Path(root);public=root/'public';prefix='experiments/medtrace_stage19_fasttrack_20260917';report='reports/medtrace_stage19_fasttrack_20260917'
    names=['stage19_fasttrack','stage19_fasttrack_budget','stage19_fasttrack_prepare','stage19_fasttrack_closeout','stage19_fasttrack_controller','test_stage19_fasttrack','stage15','stage18_cfact','stage18_support']
    files={prefix+'/scripts/medtrace/'+name+'.py':(ROOT/'scripts/medtrace'/(name+'.py')).read_bytes() for name in names}
    for p in public.iterdir():
        if p.suffix not in ('.json','.md','.csv','.png'):continue
        if p.name in ('PUBLICATION_RECEIPT.json','CONTROLLER_STATUS.json'):continue
        data=p.read_bytes()
        if p.suffix!='.png' and any(value in data for value in (b'"raw_answer"',b'"raw_token_ids"',b'"image_path"',b'"source_qid"',b'/Users/',b'/remote-home/')):
            raise ValueError('Private content in public report: '+p.name)
        files[report+'/'+p.name]=data
    files[prefix+'/README.md']=b'''# Stage19 FASTTRACK source overlay\n\nApply the prior Stage15-19 published source overlays, then this overlay. Reuses the frozen FP16 LLaVA-Med runtime, FP32 writers, HSIC Top1 and original BalancEdit recipe. Private source manifests and authorized model/image files are required; none are distributed.\n\nRun CPU checks with `python -m scripts.medtrace.test_stage19_fasttrack` and the existing Stage18/19 checks. The supervisor charges every GPU worker attempt against one 28800-second ledger. The local controller performs frozen source-agreement scoring and publishes only aggregate reports. Two arms share actual background experts in track S; total bank N is not FACT coverage K. See the corresponding report directory for realized N, coverage, resource usage and limitations.\n'''
    repository='DLwbm123/Knowledge_editing-public'
    head=json.loads(subprocess.check_output(['gh','api','repos/'+repository+'/commits/main'],text=True))['sha']
    payload=dict(query='mutation($input:CreateCommitOnBranchInput!){createCommitOnBranch(input:$input){commit{oid url}}}',variables=dict(input=dict(branch=dict(repositoryNameWithOwner=repository,branchName='main'),expectedHeadOid=head,
        message=dict(headline='Report paired Stage19 fasttrack experiment with actual coverage and resource bounds'),fileChanges=dict(additions=[dict(path=k,contents=base64.b64encode(v).decode()) for k,v in sorted(files.items())]))))
    path=root/'private/PUBLICATION_PAYLOAD.json';write(path,payload)
    response=json.loads(subprocess.check_output(['gh','api','graphql','--input',str(path)],text=True))
    if response.get('errors'):raise RuntimeError(response['errors'])
    receipt=response['data']['createCommitOnBranch']['commit']
    remote=json.loads(subprocess.check_output(['gh','api','repos/'+repository+'/commits/main'],text=True))['sha']
    if remote!=receipt['oid']:raise ValueError('Published head not verified')
    url='https://raw.githubusercontent.com/'+repository+'/'+remote+'/'+report+'/ADVISOR_BRIEF_ZH.md'
    with urllib.request.urlopen(url,timeout=30) as result:
        if result.status!=200 or result.read()!=files[report+'/ADVISOR_BRIEF_ZH.md']:raise ValueError('Anonymous publication check failed')
    receipt.update(file_count=len(files),anonymous_access=True,private_material_published=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(root/'PUBLICATION_RECEIPT.json',receipt)
    return receipt


def run(cfg):
    from scripts.medtrace.stage19_fasttrack_prepare import finalize
    from scripts.medtrace.stage19_fasttrack_closeout import qualification,score_endpoint,report
    root=Path(cfg['local_run']);remote=cfg['remote_run'];ssh=['ssh','-S',cfg['ssh_socket'],'-p',str(cfg['port']),cfg['host']]
    transport=' '.join(ssh[:-1]);host=cfg['host'];alias=Path(cfg['local_alias'])
    status=root/'public/CONTROLLER_STATUS.json'
    def remote_call(command):return subprocess.check_output(ssh+['bash -s'],input=command+'\n',text=True)
    def pull(path):
        target=alias/path;target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-a','-e',transport,host+':'+remote+'/'+path,str(target)],check=True)
    def push(path):
        subprocess.run(['rsync','-a','-e',transport,str(alias/path),host+':'+remote+'/'+path],check=True)
    def wait_exit(phase):
        while remote_call('test -f '+remote+'/public/EXIT_'+phase+'.json && echo ready || true').strip()!='ready':
            time.sleep(45)
        pull('public/EXIT_'+phase+'.json');pull('public/GPU_BUDGET_LEDGER.json')
        ledger=read(root/'public/BUDGET_LEDGER.json');gpu=read(root/'public/GPU_BUDGET_LEDGER.json')
        ledger.update(gpu_seconds_used=sum(s['seconds'] for s in gpu['sessions']),gpu_sessions=gpu['sessions'])
        write(root/'public/BUDGET_LEDGER.json',ledger)
        if read(root/'public'/('EXIT_'+phase+'.json'))['exit_code']!=0:
            pull('private/'+phase+'.log');raise RuntimeError('GPU phase failed: '+phase)
    def start(phase):
        path='private/SUPERVISOR_'+phase+'.json';push(path)
        remote_call('JOB_CONFIG='+remote+'/'+path+' JOB_ARGV=\''+json.dumps([cfg['remote_code']+'/scripts/medtrace/stage19_fasttrack_budget.py'])+'\' nohup '+cfg['remote_python']+' '+cfg['remote_entry']+' main > '+remote+'/private/'+phase+'.log 2>&1 < /dev/null & echo $!')
    write(status,dict(phase='MIGRATION_RUNNING'))
    while True:
        migration=remote_call('cat '+str(Path(remote).parent)+'/MIGRATION_EXIT.txt 2>/dev/null || true').strip()
        if migration:break
        time.sleep(45)
    if migration!='0':raise RuntimeError('Migration tool failed; GPU has not been started')
    gpu=remote_call('nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits -i 0').strip().split(',')
    if gpu[0].strip()!='GPU-ccc69848-6553-ad18-65b4-09227b59cfa0' or int(gpu[1])<22000:
        raise RuntimeError('Authorized GPU identity/free-memory startup guard failed')
    if cfg.get('resume_train'):
        summary=read(root/'public/STREAM_MANIFEST.json');config=read(root/'private/SUPERVISOR_train.json')
        if (config['stream_binding']!=summary['stream_binding'] or not config.get('resume_audit')
                or read(root/'public/EXIT_base.json')['exit_code']!=0):
            raise ValueError('Training recovery authorization/evidence missing')
    else:
        write(status,dict(phase='BASE_RUNNING'))
        start('base');wait_exit('base');pull('private/FRESH_BASE_OUTPUTS.json')
        write(status,dict(phase='BASE_QUALIFICATION'))
        qualification(root);summary=finalize(root)
        config=read(root/'private/SUPERVISOR_base.json');config.update(phase='train',mode='STAGE19_FASTTRACK_TWO_ARMS',stream_binding=summary['stream_binding'])
        write(root/'private/SUPERVISOR_train.json',config);push('private/STREAM.json')
    write(status,dict(phase='TRAINING',N=summary['N'],track=summary['track']))
    start('train');scored=set()
    while True:
        state=remote_call('for n in 011 050 100; do test ! -f '+remote+'/public/PREFIX_$n.json || echo $n; done; test ! -f '+remote+'/public/EXIT_train.json || echo exit').split()
        for token in state:
            if token=='exit':continue
            n=int(token)
            if n in scored or n==summary['N']:continue
            pull('private/OUTPUTS.jsonl');score_endpoint(root,n);scored.add(n)
        if 'exit' in state:break
        time.sleep(45)
    wait_exit('train');pull('private/OUTPUTS.jsonl');pull('private/DISPATCH_train.json')
    subprocess.run(['rsync','-a','--exclude=BUDGET_LEDGER.json','--exclude=CONTROLLER_STATUS.json','-e',transport,host+':'+remote+'/public/',str(alias/'public/')],check=True)
    done=read(root/'public/GENERATED.json');score_endpoint(root,done['N'],final=True)
    write(status,dict(phase='REPORTING',N=done['N']));report(root)
    # Preserve small reproducibility material before the completed weight lifecycle ends.
    subprocess.run(['rsync','-a','--include=*/','--include=*.json','--include=*.jsonl','--include=BANKS.pt','--exclude=*','-e',transport,host+':'+remote+'/private/',str(alias/'private/')],check=True)
    cleanup=read(root/'private/DISPATCH_train.json');cleanup['phase']='cleanup';write(root/'private/DISPATCH_cleanup.json',cleanup);push('private/DISPATCH_cleanup.json')
    remote_call('JOB_CONFIG='+remote+'/private/DISPATCH_cleanup.json JOB_ARGV=\''+json.dumps([cfg['remote_code']+'/scripts/medtrace/stage19_fasttrack.py'])+'\' '+cfg['remote_python']+' '+cfg['remote_entry']+' run')
    pull('public/CHECKPOINT_CLEANUP.json');pull('private/CHECKPOINT_CLEANUP.json')
    receipt=publish(alias);write(status,dict(phase='COMPLETE',N=done['N'],public_commit=receipt['oid'],url=receipt['url']))


if __name__=='__main__':
    cfg=read(os.environ['JOB_CONFIG'])
    try:run(cfg)
    except Exception as error:
        write(Path(cfg['local_run'])/'public/CONTROLLER_STATUS.json',dict(phase='FAILED_PRESERVED',error=str(error)))
        raise
