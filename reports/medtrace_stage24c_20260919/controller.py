"""Finite generation->isolated scoring->public report; no automatic next experiment."""
import os,sys,time,json,subprocess,shlex,fcntl,base64,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from scripts.medtrace.stage19_fasttrack_budget import read,write
from scripts.medtrace.stage20_closeout import outputs
from scripts.medtrace.stage18_score import score_key
from reports.medtrace_stage22_20260919.scoring import judge
from report import report


def run(cfg):
    root=Path(cfg['local_run']);common=Path(cfg['common_local_run']);dest=Path(__file__).parent;lock=(root/'private/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',cfg['ssh_socket'],'-p','30177','root@hb01-ssh.gpuhome.cc'];transport=shlex.join(ssh[:-1]);started=time.time()
    while True:
        subprocess.run(['rsync','-a','-e',transport,ssh[-1]+':'+cfg['remote_run']+'/public/',str(root/'public')+'/', '--exclude=CONTROLLER_STATUS.json','--exclude=RESULTS.json','--exclude=REPORT_STATUS.json','--exclude=ADVISOR_UPDATE_ZH.md'],check=True)
        exists=subprocess.run(ssh+['test -f '+shlex.quote(cfg['remote_run']+'/private/OUTPUTS.jsonl')],stdout=subprocess.DEVNULL).returncode==0
        if exists:subprocess.run(['rsync','-a','-e',transport,ssh[-1]+':'+cfg['remote_run']+'/private/OUTPUTS.jsonl',str(root/'private/OUTPUTS.jsonl')],check=True)
        rows=outputs(root)
        for n in sorted({r['prefix'] for r in rows if r['arm']=='R_H' and r['mode']=='insertion'}):
            rr=[r for r in rows if r['arm']=='R_H' and r['mode']=='insertion' and r['prefix']==n];name=f'24C_R_H_INSERT_{n:03d}'
            if not (common/'private'/f'{name}_SCORE_RECEIPT.json').exists():judge(common,rr,name,category='24C')
        for arm in ('E0','E2','R_H'):
            rr=[r for r in rows if r['arm']==arm and r['mode']=='DEV79'];name='24C_DEV79_'+arm
            if len(rr)==79 and not (common/'private'/f'{name}_SCORE_RECEIPT.json').exists():
                cache=read(common/'private/QUALIFIED_SCORE_CACHE.json')['scores'];assert all(score_key(r['source'],r['Base']) in cache for r in rr);judge(common,rr,name,category='24C')
        status=report(root,common);write(root/'public/CONTROLLER_STATUS.json',dict(phase='RUNNING_OR_SCORING',status=status))
        exitfile=root/'public/EXIT_24C.json'
        if exitfile.exists():
            if read(exitfile)['exit_code']!=0:raise RuntimeError('GPU stopped; retain charged partial run, no automatic retry')
            assert status['complete'];break
        if time.time()-started>14400:raise TimeoutError('Finite four-hour controller window reached')
        time.sleep(45)
    for n in ('RESULTS.json','REPORT_STATUS.json','ADVISOR_UPDATE_ZH.md','ENVIRONMENT_ACCEPTANCE.json','GENERATED.json','RESOURCE_PROFILE.json','EXIT_24C.json'):(dest/n).write_bytes((root/'public'/n).read_bytes())
    subprocess.run(['rsync','-a','-e',transport,ssh[-1]+':/root/rivermind-data/job-522/run/public/GPU_BUDGET_LEDGER.json',str(common/'public/GPU_BUDGET_LEDGER.json')],check=True)
    ledger=read(common/'public/GPU_BUDGET_LEDGER.json');write(dest/'GPU_ACCOUNT.json',dict(cumulative_seconds=sum(x['seconds'] for x in ledger['sessions']),limit=28800,this_phase_seconds=sum(x['seconds'] for x in ledger['sessions'] if x['phase']=='24C')))
    jl=read(common/'public/BUDGET_LEDGER.json');batches=[b for b in jl['judgment_batches'] if b['category']=='24C'];receipts=[read(f) for f in (common/'private').glob('24C_*_SCORE_RECEIPT.json')]
    write(dest/'JUDGE_ACCOUNT.json',dict(cumulative=jl['new_judgment_items_dispatched'],this_phase=jl['24C_items_dispatched'],usage={k:sum((b.get('usage') or {}).get(k,0) for b in batches) for k in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')},exact_reuse_consumers=sum(r['reused'] for r in receipts),cost_status='Provider currency billing unavailable; never treated as zero',actual_cost=None,cache_count_unit='consumer batches, not independent QA'))
    subprocess.run(['git','-C',str(ROOT),'add',str(dest)],check=True);subprocess.run(['git','-C',str(ROOT),'commit','-m','Report completed pure19 R2 HSIC comparison and full DEV coverage'],check=True)
    files=subprocess.check_output(['git','-C',str(ROOT),'ls-files',str(dest.relative_to(ROOT))],text=True).splitlines();assert all('/private/' not in f for f in files)
    sha=subprocess.check_output(['gh','api','repos/DLwbm123/Knowledge_editing-public/git/ref/heads/main','--jq','.object.sha'],text=True).strip()
    payload=dict(query='mutation($input:CreateCommitOnBranchInput!){createCommitOnBranch(input:$input){commit{oid url}}}',variables=dict(input=dict(branch=dict(repositoryNameWithOwner='DLwbm123/Knowledge_editing-public',branchName='main'),expectedHeadOid=sha,message=dict(headline='Stage24C pure19 R2 HSIC DEV results'),fileChanges=dict(additions=[dict(path=f,contents=base64.b64encode((ROOT/f).read_bytes()).decode()) for f in files]))))
    packet=root/'private/PUBLICATION_PAYLOAD.json';write(packet,payload);response=json.loads(subprocess.check_output(['gh','api','graphql','--input',str(packet)],text=True));write(root/'private/PUBLICATION_RESPONSE.json',response);assert not response.get('errors');commit=response['data']['createCommitOnBranch']['commit'];url='https://raw.githubusercontent.com/DLwbm123/Knowledge_editing-public/'+commit['oid']+'/'+str(dest.relative_to(ROOT))+'/ADVISOR_UPDATE_ZH.md'
    with urllib.request.urlopen(url,timeout=30) as r:assert r.status==200 and '固定R2'.encode() in r.read()
    with urllib.request.urlopen(commit['url'],timeout=30) as r:assert r.status==200
    remote=subprocess.check_output(['gh','api','repos/DLwbm123/Knowledge_editing-public/git/ref/heads/main','--jq','.object.sha'],text=True).strip();assert remote==commit['oid']
    write(root/'private/PUBLICATION_RECEIPT.json',dict(**commit,anonymous_access_verified=True,remote_head_verified=True));write(root/'public/CONTROLLER_STATUS.json',dict(phase='COMPLETE_SCORED_PUBLISHED',public_commit=commit['oid'],Stage25='NOT_STARTED'))

if __name__=='__main__':
    cfg=read(os.environ['JOB_CONFIG'])
    try:run(cfg)
    except Exception as e:
        write(Path(cfg['local_run'])/'public/CONTROLLER_STATUS.json',dict(phase='NEEDS_BOUNDED_REVIEW',error=str(e),no_semantic_retry=True));raise
