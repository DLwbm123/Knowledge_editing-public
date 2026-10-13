"""Finite completion of the already-authorized data audit; never launches GPU work."""
import collections
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
W=Path(os.environ['WORKSPACE']);S=W/'.local_run';D=W/'reports/purew_cpol146_v1_20261013'
BRANCH='experiment/purew-cpol146-v1-20261013'

def read(p):return json.loads(p.read_text())
def write(p,d):
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
    import fcntl
    handle=(S/'QUALIFICATION_TAIL.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (S/'QUALIFICATION_PUBLICATION.json').exists():return
    deadline=time.time()+6*3600
    worker=read(S/'QUALIFICATION_START.json')
    while not (S/'SOURCE_QUALIFICATION.json').exists() and not (S/'QUALIFICATION_FAILURE.json').exists():
        p=subprocess.run(['ps','-ww','-p',str(worker['pid']),'-o','lstart=,args='],text=True,capture_output=True)
        assert p.returncode==0 and worker['lstart'] in p.stdout and worker['entry'] in p.stdout,'Qualification worker missing; retain state, no blind retry'
        assert time.time()<deadline,'Finite data audit wait expired'
        time.sleep(30)
    if (S/'SOURCE_QUALIFICATION.json').exists():
        env=dict(os.environ,WORKSPACE=str(W))
        subprocess.run([sys.executable,'-c','import os,runpy;runpy.run_path(os.environ["WORKSPACE"]+"/experiments/purew_cpol146_v1/collision_screen.py",run_name="__main__")'],env=env,check=True)
        result=read(S/'QUALIFICATION_FINAL.json')
    else:
        result=dict(status='BLOCKED_INCOMPLETE_QUALIFICATION',failure_record_retained=True,training_launched=False,source_review_complete=False)
        write(D/'QUALIFICATION_RESULT.json',result)
    styles=collections.defaultdict(collections.Counter)
    for p in (S/'qualification').glob('review_*/RESPONSE.json'):
        for r in read(p)['rows']:
            for k,v in r.items():
                if k=='order':continue
                styles[k]['reviewed']+=1
                styles[k]['accepted']+=int(all(v[x] for x in ('equivalent','answer_compatible','no_added_information')))
                for criterion in ('equivalent','answer_compatible','no_added_information'):
                    styles[k]['failed_'+criterion]+=int(not v[criterion])
    ev=[read(p) for p in (S/'qualification').glob('*/EVIDENCE.json')]
    tokens=collections.Counter()
    for e in ev:
        tokens.update({k:v for k,v in (e.get('usage') or {}).items() if isinstance(v,int)})
    cost=dict(requests_attempted=len(ev),author_calls=sum(e['stage']=='author' for e in ev),review_calls=sum(e['stage']=='review' for e in ev),provider_or_schema_failed=sum(e['status']!='FORMAT_VALID' for e in ev),usage=dict(tokens),summed_call_seconds=sum(e.get('finished',e['started'])-e['started'] for e in ev),GPU_seconds=0,formal_updates=0,engineering_classifier_failure=dict(retained=True,replayed_requests=0,original_response_reused=True))
    write(D/'QUALIFICATION_COSTS.json',cost);write(D/'STYLE_AUDIT.json',dict(styles))
    protocol=read(D/'PROTOCOL.json')
    protocol['qualification']['status']=result['status']
    if result.get('candidate_binding'):protocol['qualification']['candidate_binding']=result['candidate_binding']
    protocol['status']='DESIGN_FROZEN_DATA_QUALIFIED_GPU_IMPLEMENTATION_PENDING' if result['status']=='PASS' else 'DESIGN_FROZEN_DATA_BLOCKED'
    write(D/'PROTOCOL.json',protocol)
    (D/'PROTOCOL.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>C/P/O/L 冻结设计</title><style>body{max-width:1000px;margin:40px auto;font:16px system-ui}pre{white-space:pre-wrap}</style><h1>C/P/O/L 有限机制对照</h1><p>'+html.escape(protocol['status'])+'</p><pre>'+html.escape(json.dumps(protocol,ensure_ascii=False,indent=2))+'</pre></html>')
    lines=['# 训练改写资格审查结果','',f'状态：**{result["status"]}**。新 GPU 正式更新为0；当前原146连续写入未被修改。','', '| 改写类别 | 已审查 | 语义通过 |','|---|---:|---:|']
    for k,v in styles.items():lines.append(f'| {k} | {v["reviewed"]} | {v["accepted"]} |')
    lines+=['','语义通过不等于最终数据准入：还需通过核心问题变化、四种候选互异，以及全评价题的独立重合检查。','', '```json',json.dumps(result,ensure_ascii=False,indent=2),'```','',f'共尝试 {len(ev)} 次作者/独立审核请求，全部调用、失败与费用计数保留。资格审查不使用GPU。首批本地事件分类误报已经保留；直接复用原响应，未重请求。','', 'C/P/O/L的机制与逐步范数预算保持冻结。数据版本未通过时不取通过子集、不依据评价重合信息修补候选，不把评价题用于训练，不启动P/O/L。若数据资格通过，仍须完成并验证GPU执行器、评分和交付消费者，再等待原146 GPU完成及新轮真实模型准入；本报告不等于GPU排队成功。','', 'C的11项永久缺评分不重试，最终对比保留共享缺分上下界。保护按逐题零新增损伤判定，不能用平均提升补偿。']
    (D/'QUALIFICATION_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    paths=[str((D/n).relative_to(W)) for n in ['QUALIFICATION_RESULT.json','QUALIFICATION_COSTS.json','STYLE_AUDIT.json','PROTOCOL.json','PROTOCOL.html','QUALIFICATION_REPORT_ZH.md']]
    env=dict(os.environ,HTTP_PROXY='http://127.0.0.1:7897',HTTPS_PROXY='http://127.0.0.1:7897',http_proxy='http://127.0.0.1:7897',https_proxy='http://127.0.0.1:7897',NO_PROXY='',no_proxy='')
    assert subprocess.check_output(['git','branch','--show-current'],cwd=W,text=True).strip()==BRANCH
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=W,text=True).strip(),'Unrelated staged changes; do not publish'
    subprocess.run(['git','add','--',*paths],cwd=W,check=True)
    subprocess.run(['git','commit','-m','Record complete source-only rewrite qualification and retained failures'],cwd=W,check=True)
    subprocess.run(['git','-c','http.proxy=http://127.0.0.1:7897','push','origin',BRANCH],cwd=W,env=env,check=True,timeout=120)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897','ls-remote','origin','refs/heads/'+BRANCH],cwd=W,env=env,text=True,timeout=60).split()[0]
    assert sha==remote
    url='https://raw.githubusercontent.com/DLwbm123/Knowledge_editing-public/'+sha+'/'+str((D/'QUALIFICATION_RESULT.json').relative_to(W))
    raw=subprocess.check_output(['curl','--fail','--silent','--show-error','--proxy','http://127.0.0.1:7897',url],env=env,text=True,timeout=60)
    assert json.loads(raw)==result
    write(S/'QUALIFICATION_PUBLICATION.json',dict(status='PUBLISHED',commit=sha,remote_sha=remote,anonymous_access=True,result=result['status'],GPU_started=False,epoch=time.time()))
    print(json.dumps(dict(publication='PUBLISHED',result=result['status'],commit=sha)),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        write(S/'QUALIFICATION_TAIL_FAILURE.json',dict(error=repr(e),epoch=time.time(),no_GPU_launched=True));raise
