"""Idempotent local dispatch after MEMIT generation, scoring and closeout."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def remote(code):
    p=subprocess.run(['ssh','pro5000','/data/bmw/envs/v0/bin/python','-'],input=code,text=True,capture_output=True,timeout=180)
    if p.returncode:raise RuntimeError(p.stderr[-4000:])
    return json.loads(p.stdout)


def main():
    cfg=read(os.environ['DISPATCH_CONFIG']);statefile=Path(cfg['state']);state=read(statefile)
    prefix='import os,json,pathlib,sys\nq=pathlib.Path('+repr(cfg['neutral'])+')\nos.environ.update(json.loads((q/\'ENV.json\').read_text()))\nr=pathlib.Path(os.environ[\'EXPERIMENT_ROOT\'])\np=pathlib.Path(os.environ[\'REFERENCE_RUN\'])\no=p/\'MEMIT_MM\'\n'
    snapshot=remote(prefix+'''
import sqlite3,time
read=lambda p:json.loads(p.read_text())
def live(pid,ticks):
 f=pathlib.Path('/proc',str(pid),'stat')
 return f.exists() and f.read_text().split(') ',1)[1].split()[19]==str(ticks)
identities=__IDENTITIES__
identities += [read(f) for f in (o/'parallel/prefix146_completion').glob('*/START.json')]
active=[v['pid'] for v in identities if live(v['pid'],v['start_ticks'])]
terminal=(o/'INCOMPLETE_GENERATION.json').exists() or (o/'GENERATION_COMPLETE.json').exists()
generation_terminal=terminal and not active and (o/'WORKER_EXIT.json').exists() and (o/'GUARD_EXIT.json').exists()
if generation_terminal:
 assert read(o/'WORKER_EXIT.json')['code']==read(o/'GUARD_EXIT.json')['code']==0
 terminal_receipt=read(o/'INCOMPLETE_GENERATION.json') if (o/'INCOMPLETE_GENERATION.json').exists() else read(o/'GENERATION_COMPLETE.json')
 assert terminal_receipt['counts']['inserted']==146 and terminal_receipt['counts']['edits']==292
 assert terminal_receipt['counts']['formal_consumers']+len(terminal_receipt.get('aborted_consumers',[]))==7494
else:terminal_receipt=None
db=sqlite3.connect('file:'+str(p/'judge_v131/queue.sqlite')+'?mode=ro',uri=True)
states=dict(db.execute('SELECT state,count(*) FROM payload GROUP BY state'))
consumers=db.execute("SELECT count(*) FROM consumer WHERE method='MEMIT_MM'").fetchone()[0]
health=read(o/'GUARD_HEALTH.json')
print(json.dumps(dict(time=time.time(),generation_terminal=generation_terminal,active_pids=active,queue_states=states,
 memit_scored_consumers=consumers,prior_managed_bytes=health['new_bytes'],terminal_receipt=terminal_receipt,
 med_dispatched=(r/'GUARD_ONCE').exists())))
'''.replace('__IDENTITIES__',repr(cfg['dependency_identities'])))
    write(Path(cfg['local'])/'DEPENDENCY_SNAPSHOT.json',snapshot)
    if snapshot['med_dispatched']:
        print(json.dumps(dict(status='ALREADY_DISPATCHED_NO_DUPLICATE')));return
    scorer=read(cfg['MEMIT_scorer_config']);scorer_end=Path(scorer['local'])/'SCORER_EXIT.json'
    scorer_terminal=scorer_end.exists() and read(scorer_end)['code']==0
    ready=snapshot['generation_terminal'] and scorer_terminal and not any(snapshot['queue_states'].get(k,0) for k in ('PENDING','RESERVED'))
    if not ready:
        state.update(launch_readiness='READY_WAITING_FOR_MEMIT_TERMINAL',last_dependency_snapshot=snapshot)
        write(statefile,state);print(json.dumps(dict(status='WAITING_FOR_MEMIT_TERMINAL',generation_terminal=snapshot['generation_terminal'],scoring_terminal=scorer_terminal)));return
    expected=snapshot['terminal_receipt']['counts']['formal_consumers'];assert snapshot['memit_scored_consumers']==expected
    report=remote(prefix+"sys.path.insert(0,str(r/'tools'))\nimport memit_closeout\nmemit_closeout.main()")
    public=Path(cfg['workspace'])/'reports/med146_single_sequential_v133_20261027'
    write(public/'MEMIT_TERMINAL_RESULTS.json',report)
    (public/'MEMIT_TERMINAL_REPORT_ZH.md').write_text('# MEMIT 终态收尾\n\n'+report['status']+'。全部146次插入已完成，正式生成'+str(report['durable_generated'])+'/7494；保留'+str(report['known_generation_UNKNOWN'])+'条已声明UNKNOWN，未重试。\n\n评分采用 Astra high 的原二元来源一致性协议；不混作新MED共同CORE/FULL epoch。详见匿名JSON。\n')
    dependency=dict(generation_terminal=True,scoring_terminal=True,prior_managed_bytes=snapshot['prior_managed_bytes'],
        known_generation_UNKNOWN=report['known_generation_UNKNOWN'],time=time.time(),source='Identity-bound actual terminal receipts and queue')
    commit=subprocess.check_output(['git','-C',cfg['workspace'],'rev-parse','HEAD'],text=True).strip()
    assert subprocess.check_output(['git','-C',cfg['workspace'],'branch','--show-current'],text=True).strip()==cfg['branch']
    receipt=remote(prefix+'import subprocess,time\n'+
        "assert not (r/'GUARD_ONCE').exists()\n"+
        "(r/'MEMIT_DEPENDENCY_TERMINAL.json').write_text("+repr(json.dumps(dependency,indent=2))+" )\n"+
        "(r/'SOURCE_VERSION.json').write_text(json.dumps(dict(commit="+repr(commit)+",branch="+repr(cfg['branch'])+",protocol='MED146_SINGLE_SEQUENTIAL_V133')))\n"+
        "log=(r/'guard.log').open('a')\nchild=subprocess.Popen([sys.executable,str(q/'q.py')],env=dict(os.environ,ENTRY_MODE='guard'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nlog.close()\n"+
        "ticks=pathlib.Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split()[19]\nvalue=dict(pid=child.pid,start_ticks=ticks,time=time.time(),GPU=0,source_commit="+repr(commit)+")\n(r/'DISPATCH.json').write_text(json.dumps(value,indent=2))\nprint(json.dumps(value))")
    time.sleep(2)
    health=remote(prefix+"print(json.dumps(dict(clock=(r/'CLOCK.json').exists(),exit=json.loads((r/'GUARD_EXIT.json').read_text()) if (r/'GUARD_EXIT.json').exists() else None,worker=json.loads((r/'WORKER_IDENTITY.json').read_text()) if (r/'WORKER_IDENTITY.json').exists() else None)))")
    if not (health['clock'] and health['worker'] and health['exit'] is None):
        state.update(status='MED_GPU_ADMISSION_PENDING_OR_FAILED',last_guard_startup=health,MED_guard=receipt)
        write(statefile,state)
        raise RuntimeError('MED did not pass GPU guard startup; inspect guard.log, preserve the original clock if present')
    local=Path(cfg['local']);local.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,SCORING_CONFIG=cfg['MED_scoring_config'],JUDGE_SHARED_ROOT=str(local),
        LOCAL_NEUTRAL_ENTRY=cfg['MED_score_entry'],LOCAL_ENTRY_MODE='guard')
    with (local/'score.log').open('a') as log:
        child=subprocess.Popen([sys.executable,cfg['MED_score_entry']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(local/'SCORER_DISPATCH.json',dict(pid=child.pid,time=time.time()))
    state.update(status='MED_STARTED_GPU0_ONLY',launch_readiness='DISPATCHED',MED_guard=receipt,MED_worker=health['worker'],MED_local_guard_pid=child.pid,MED_original_clock=remote(prefix+"print((r/'CLOCK.json').read_text())"))
    write(statefile,state);print(json.dumps(dict(status='STARTED_GPU0_ONLY',guard=receipt,worker=health['worker'],local_scorer=child.pid)))

if __name__=='__main__':main()
