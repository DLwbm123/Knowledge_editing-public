"""Finish, freeze and publish the bounded whole-source revision audit."""
import collections
import fcntl
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
W=Path(os.environ['WORKSPACE']);S=W/'.local_run/revision2';D=W/'reports/purew_cpol146_v1_20261013/revision2'
BRANCH='experiment/purew-cpol146-v1-20261013'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
    lock=(S/'TAIL.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (S/'PUBLICATION.json').exists():return
    worker=read(S/'QUALIFICATION_START.json');deadline=worker['started_epoch']+6*3600
    while not (S/'SOURCE_QUALIFICATION.json').exists() and not (S/'QUALIFICATION_FAILURE.json').exists():
        p=subprocess.run(['ps','-ww','-p',str(worker['pid']),'-o','lstart=,args='],text=True,capture_output=True)
        assert p.returncode==0 and worker['lstart'] in p.stdout and worker['entry'] in p.stdout,'Worker missing; no blind retry'
        assert time.time()<deadline,'Finite qualification deadline expired'
        time.sleep(30)
    if (S/'SOURCE_QUALIFICATION.json').exists():
        code='import os,runpy;runpy.run_path(os.environ["WORKSPACE"]+"/experiments/purew_cpol146_v1/collision_revision2.py",run_name="__main__")'
        subprocess.run([sys.executable,'-c',code],env=dict(os.environ,WORKSPACE=str(W)),check=True)
        result=read(S/'QUALIFICATION_FINAL.json')
    else:
        result=dict(status='BLOCKED_INCOMPLETE_QUALIFICATION',failure_record_retained=True,training_launched=False)
        write(D/'QUALIFICATION_RESULT.json',result)
    styles=collections.defaultdict(collections.Counter)
    for p in (S/'qualification').glob('review_*/RESPONSE.json'):
        for r in read(p)['rows']:
            for k,v in r.items():
                if k=='order':continue
                styles[k]['reviewed']+=1;styles[k]['accepted']+=int(all(v[x] for x in ('equivalent','answer_compatible','no_added_information')))
                for check in ('equivalent','answer_compatible','no_added_information'):styles[k]['failed_'+check]+=int(not v[check])
    ev=[read(p) for p in (S/'qualification').glob('*/EVIDENCE.json')];usage=collections.Counter()
    for e in ev:usage.update({k:v for k,v in (e.get('usage') or {}).items() if isinstance(v,int)})
    old=read(D.parent/'QUALIFICATION_COSTS.json')
    cost=dict(revision=2,requests_attempted=len(ev),author_calls=sum(e['stage']=='author' for e in ev),review_calls=sum(e['stage']=='review' for e in ev),failed_calls=sum(e['status']!='FORMAT_VALID' for e in ev),usage=dict(usage),summed_call_seconds=sum(e.get('finished',e['started'])-e['started'] for e in ev),GPU_seconds=0,formal_updates=0,prior_revision=old,cumulative_requests=old['requests_attempted']+len(ev),cumulative_usage=dict(collections.Counter(old['usage'])+usage),no_old_request_retries=True)
    write(D/'STYLE_AUDIT.json',dict(styles));write(D/'QUALIFICATION_COSTS.json',cost)
    protocol=read(D/'PROTOCOL.json')
    protocol['qualification']['status']=result['status']
    if result.get('candidate_binding'):protocol['qualification']['candidate_binding']=result['candidate_binding']
    protocol['status']='FROZEN_REVISION2_DATA_QUALIFIED_GPU_IMPLEMENTATION_PENDING' if result['status']=='PASS' else 'FROZEN_REVISION2_DATA_BLOCKED'
    protocol['qualification']['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    write(D/'PROTOCOL.json',protocol)
    (D/'PROTOCOL.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>C/P/O/L 修订版2</title><style>body{max-width:1000px;margin:40px auto;font:16px system-ui}pre{white-space:pre-wrap}</style><h1>整批源端修订版2</h1><pre>'+html.escape(json.dumps(protocol,ensure_ascii=False,indent=2))+'</pre></html>')
    lines=['# 整批训练改写：修订版2资格结果','',f'状态：**{result["status"]}**。新GPU正式更新0；原146连续写入和旧版全部记录保留。','', '| 类别 | 审查条数 | 语义通过 |','|---|---:|---:|']
    for k,v in styles.items():lines.append(f'| {k} | {v["reviewed"]} | {v["accepted"]} |')
    lines+=['','语义通过还需核心问句变化、候选互异及独立重合准入。作者/审核者仅见原训练源問答及新候选；没有评价题/答案、旧候选、旧失败或重合项身份。','', '```json',json.dumps(result,ensure_ascii=False,indent=2),'```','', f'本版尝试{len(ev)}次请求，累计含旧版{cost["cumulative_requests"]}次；旧请求不重试，全部成本保留。','', '只在整批资格PASS后冻结可训练数据绑定。仍须完成GPU执行/评分/交付代码并验证，等待原146GPU全部阶段结束、正常退出和无活动租约，再进行新轮真实模型准入；本报告不是GPU已排队或准入已通过。失败不得依据评价重合反馈改词或选子集。C的既有11项缺评分保留上下界。']
    (D/'QUALIFICATION_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    paths=[str((D/n).relative_to(W)) for n in ['QUALIFICATION_RESULT.json','QUALIFICATION_COSTS.json','STYLE_AUDIT.json','PROTOCOL.json','PROTOCOL.html','QUALIFICATION_REPORT_ZH.md']]
    env=dict(os.environ,HTTP_PROXY='http://127.0.0.1:7897',HTTPS_PROXY='http://127.0.0.1:7897',http_proxy='http://127.0.0.1:7897',https_proxy='http://127.0.0.1:7897',NO_PROXY='',no_proxy='')
    assert subprocess.check_output(['git','branch','--show-current'],cwd=W,text=True).strip()==BRANCH
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=W,text=True).strip()
    subprocess.run(['git','add','--',*paths],cwd=W,check=True)
    subprocess.run(['git','commit','-m','Freeze revision2 qualification outcome and cumulative audit costs'],cwd=W,check=True)
    subprocess.run(['git','-c','http.proxy=http://127.0.0.1:7897','push','origin',BRANCH],cwd=W,env=env,check=True,timeout=120)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897','ls-remote','origin','refs/heads/'+BRANCH],cwd=W,env=env,text=True,timeout=60).split()[0];assert sha==remote
    url='https://raw.githubusercontent.com/DLwbm123/Knowledge_editing-public/'+sha+'/'+str((D/'QUALIFICATION_RESULT.json').relative_to(W))
    raw=subprocess.check_output(['curl','--fail','--silent','--show-error','--proxy','http://127.0.0.1:7897',url],env=env,text=True,timeout=60);assert json.loads(raw)==result
    write(S/'PUBLICATION.json',dict(status='PUBLISHED',commit=sha,remote_sha=remote,anonymous_access=True,result=result['status'],GPU_started=False,epoch=time.time()))
    print(json.dumps(dict(status='PUBLISHED',result=result['status'],commit=sha)),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:write(S/'TAIL_FAILURE.json',dict(error=repr(e),epoch=time.time(),no_GPU_launched=True));raise
