"""One authorized J continuation in the existing run; never install or reset budgets."""
import json,os,subprocess,sys,time
from pathlib import Path
W=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
STATE=W/'.local_run/ecw_consistent3_v1_20261015/j_only_revision1';OUT=W/'reports/purew_ecw_consistent3_v1_20261015/j_only_revision1'

REMOTE=r'''
import json,os,shutil,sys,tempfile,time,subprocess,sqlite3
from pathlib import Path
x=json.loads(sys.stdin.readline());r=Path('/data/bmw/Knowledge_editing/outputs/purew-ecw-consistent3-v1-20261015/run');read=lambda p:json.loads(p.read_text())
a=x['amendment'];m=read(r/'RUN_MANIFEST.json');ledger=read(r/'RESOURCE_LEDGER.json')
assert m['starting_epoch']==a['activation_epoch'] and m['deadline_epoch']==a['deadline_epoch'] and time.time()<m['deadline_epoch']
assert all(s.get('ended_epoch') for s in ledger['gpu_sessions']) and ledger['Judge_attempts']==0
for n in ('START_CONTROLLER','START_calibration'):
 row=read(r/'private'/(n+'.json'));p=Path('/proc')/str(row['pid'])
 assert not p.exists() or (p/'stat').read_text().rsplit(')',1)[1].split()[19]!=str(row['start_ticks'])
assert not list((r/'private/steps').glob('*/*/*.json')) and not (r/'private/GENERATION_COMPLETE.json').exists()
assert sum(read(p).get('committed_updates',0)+read(p).get('attempt_slots',0) for p in (r/'private/attempts').glob('*.json'))==0
assert read(r/'private/ADMISSION_STOP_REVIEW.json')['reason']=='BASE_DECISION_MARGIN_NOT_ADMITTED'
assert x['CPU']['status']=='PASS' and x['reuse']['status']=='PASS' and a['active_arms']==['J']
db=sqlite3.connect('file:'+str(r/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True)
assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0 and not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone();db.close()
for name,text in x['files'].items():compile(text,name,'exec')
old=Path(a['calibration']['origin_run']);source=old/a['calibration']['path'];assert source.stat().st_size==x['reuse']['bytes']
mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(r)],text=True))['filesystems'][0]
assert mount['target']=='/data' and shutil.disk_usage(r).free>=m['min_free_bytes']+m['owned_weight_limit_bytes']
mem=int(next(z.split()[1] for z in Path('/proc/meminfo').read_text().splitlines() if z.startswith('MemAvailable:')))*1024;assert mem>=96*1024**3
probe=r/'private/.j_probe';probe.write_text('read_write_probe');assert probe.read_text()=='read_write_probe';probe.unlink()
archive=r/'private/engineering_j_decouple1';archive.mkdir(exist_ok=False);(archive/'public').mkdir();(archive/'failures').mkdir();(archive/'tools').mkdir()
for name in ['FROZEN_PROTOCOL.json','GPU_SOURCE_VERSION.json','START_CONTROLLER.json','START_calibration.json','ADMISSION_STOP_REVIEW.json','REPORT_COMPLETE.json','PUBLICATION.json']:
 p=r/'private'/name
 if p.exists():shutil.copy2(p,archive/name)
for p in (r/'private/failures').glob('*.json'):shutil.move(str(p),archive/'failures'/p.name)
for name in ['RESULTS.json','REPORT_ZH.md','PROGRESS.json','ADMISSION_STOP_REVIEW.json','STARTUP_VERIFICATION.json']:
 p=r/'public'/name
 if p.exists():shutil.copy2(p,archive/'public'/name)
for name,text in x['files'].items():
 p=r/'private/tools'/name
 if p.exists():shutil.copy2(p,archive/'tools'/name)
 p.write_text(text)
for name in ('REPORT_COMPLETE.json','PUBLICATION.json'):
 p=r/'private'/name
 if p.exists():p.unlink() # Small terminal markers are retained above, not discarded.
cal=r/'private/calibration/Base_plain.pt';assert not cal.exists();cal.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,cal);assert cal.stat().st_size==source.stat().st_size
proto=read(r/'private/FROZEN_PROTOCOL.json');proto.update(active_arms=['J'],execution_revision=a['revision'],user_validation_hold=a['later_user_GPU2_hold'])
def write(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
write(r/'private/FROZEN_PROTOCOL.json',proto);write(r/'public/EFFECTIVE_RUNTIME_PROTOCOL.json',proto)
write(r/'public/J_ONLY_PROTOCOL_AMENDMENT.json',a);write(r/'public/J_CONTINUATION_CPU.json',x['CPU']);write(r/'public/J_PLAIN_BASE_REUSE.json',dict(x['reuse'],fresh_outputs_this_continuation=0,inherited_parameter_factor_only=True,current_Base3_required=True))
write(r/'public/BASE_CALIBRATION.json',dict(status='PASS',mode='REUSED_VERIFIED_ORIGINAL_PLAIN_PARAMETER_FACTOR',rows=16,inherited_native_outputs_exact=8,new_calibration_model_calls=0,no_normalized_gap_gate=True))
write(r/'private/GPU_SOURCE_VERSION.json',dict(commit=x['commit']))
ledger['weights_observed_peak_bytes']=max(ledger.get('weights_observed_peak_bytes',0),cal.stat().st_size);write(r/'RESOURCE_LEDGER.json',ledger)
assert read(r/'RUN_MANIFEST.json')==m
write(r/'public/RESULTS.json',dict(status='J_CONTINUATION_WAIT_USER_GPU2_VALIDATION',formal_slots=0,new_Judge=0,N='NOT_STARTED_THEORY_PENDING',NJ='NOT_STARTED_THEORY_PENDING',historical_partial_retained=True))
(r/'public/REPORT_ZH.md').write_text('# J独立接续等待\n\n旧校准失败、partial报告和全部成本保留于engineering_j_decouple1；原激活/截止/额度不重置。仅接续J，复用plain参数统计，不复用病例缓存；N/NJ仅理论。用户正在验证GPU2，等待观测验证占用后退出并连续60秒空闲，或收到后续明确释放授权；等待期间不加载模型、不占GPU、不发Judge。\n')
tmp=tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp');env=read(r/'private/LAUNCH_ENV.json');env.update(RUN_ROOT=str(r),TMPDIR=tmp,ACTION='controller_J_revision1',ARM='J',TASK_FILE=str(r/'private/tools/controller.py'))
fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=tmp);os.close(fd);Path(entry).write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
write(archive/'CONTINUATION_AUTHORIZATION.json',a)
write(r/'private/LAUNCH_ENV.json',env)
child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',entry],env=env,stdout=(r/'logs/controller_J_revision1.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
argv=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','args='],text=True).strip();assert not any(z in argv for z in ('wangbomin','Knowledge_editing','purew','ecw'))
start=dict(pid=child.pid,start_ticks=Path('/proc').joinpath(str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19],entry=entry,argv=argv,epoch=time.time(),commit=x['commit'],activation_epoch=m['starting_epoch'],deadline_epoch=m['deadline_epoch'],new_installation=False,only_active_arm='J',initial_state='WAIT_USER_GPU2_VALIDATION')
write(r/'private/START_CONTROLLER.json',start);write(r/'private/START_J_CONTINUATION.json',start);print(json.dumps(start))
'''

def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
    STATE.mkdir(parents=True,exist_ok=True)
    old=STATE.parent
    for name in ('SCORERS','FINITE_TAIL'):
        rows=json.loads((old/(name+'.json')).read_text());rows=rows.get('workers',[rows])
        for row in rows:
            p=subprocess.run(['ps','-ww','-p',str(row['pid']),'-o','lstart=,args='],capture_output=True,text=True)
            assert not (p.returncode==0 and p.stdout.split()[:5]==row['identity'].split()[:5] and row['entry'] in p.stdout),'Existing consumer remains alive'
    payload=dict(amendment=json.loads((OUT/'PROTOCOL_AMENDMENT.json').read_text()),CPU=json.loads((OUT/'CPU_VALIDATION.json').read_text()),reuse=json.loads((OUT/'PLAIN_BASE_REUSE_VALIDATION.json').read_text()),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),files={p.name:p.read_text() for p in HERE.glob('*.py') if p.name in ('controller.py','worker.py','runtime.py','queue_revision.py','report_revision.py')})
    with (STATE/'CONTINUATION_ATTEMPT.json').open('x') as f:json.dump(dict(epoch=time.time(),commit=payload['commit'],new_installation=False),f)
    try:
        p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',"/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=json.dumps(REMOTE)+'\n'+json.dumps(payload)+'\n',text=True,capture_output=True,timeout=100)
    except subprocess.TimeoutExpired as exc:
        (STATE/'CONTINUATION_UNCERTAIN.json').write_text(json.dumps(dict(epoch=time.time(),stdout=str(exc.stdout),stderr=str(exc.stderr),no_blind_retry=True)));raise
    (STATE/'CONTINUATION_RESPONSE.json').write_text(json.dumps(dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)));assert p.returncode==0,p.stderr
    start=json.loads(p.stdout);(STATE/'STARTUP.json').write_text(json.dumps(start,indent=2))
    os.environ.update(WORKSPACE=str(W),CONTINUATION_STATE=str(STATE),PUBLIC_REPORT_SUBDIR='j_only_revision1')
    from consumers import start_consumers
    start_consumers(start)

if __name__=='__main__':main()
