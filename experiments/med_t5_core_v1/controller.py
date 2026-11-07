"""Upload-dependent T5 Base qualification, admitted MDU execution and closeout."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def read(p):return json.loads(Path(p).read_text())


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(p)


def ssh(cfg,code,timeout=90):
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',cfg['ssh_host'],cfg['remote_python'],'-'],
        input=code,text=True,capture_output=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr[-5000:])
    return json.loads(result.stdout)


def remote_state(cfg,names):
    return ssh(cfg,'from pathlib import Path\nimport json\nr=Path('+repr(cfg['remote_root'])+')\nprint(json.dumps({n:json.loads((r/n).read_text()) if (r/n).exists() else None for n in '+repr(names)+'}))')


def wait_phase(cfg,complete,exit_file,failures):
    local=Path(cfg['control']);clock=read(local/'clock.json')
    while True:
        assert time.time()<clock['hard_end']-660
        state=remote_state(cfg,[complete,exit_file]+failures)
        for failure in failures:
            assert not state[failure],state[failure]
        if state[complete] and state[exit_file]:
            assert state[exit_file]['code']==0
            return state
        write(local/'PROGRESS.json',dict(phase='WAIT_'+complete,time=time.time()))
        time.sleep(30)


def packet(cfg,values,history):
    sys.path.insert(0,cfg['common_source'])
    from med133_score_contract import PROTOCOL,digest
    fields=('core_correct','full_supported','unsupported_expansion','extra_contradiction')
    def semantic(row):return digest(dict(protocol=PROTOCOL,model='gpt-6-astra',reasoning='medium',
        question=row['question'],gold_answer=row['gold_answer'],raw_base_answer=row['raw_base_answer']))
    payloads,consumers={},[]
    for value in values:
        b=value.get('judge_input',value.get('binding',{}).get('judge_input'));answer=value.get('answer',value.get('R0'))
        row=dict(question=b['question'],gold_answer=b['reference'],raw_base_answer=answer['raw_answer'])
        key=semantic(row);payloads[key]=dict(row,opaque_query_id=key)
        consumers.append({**{k:value[k] for k in ('owner','role','query_id','consumer_id')},'mode':value.get('mode','Base'),
            'route_ON':value.get('route',{}).get('activated'),'key':key})
    votes={};failed=set()
    for folder in history:
        folder=Path(folder)
        if not (folder/'PACKET_PRIVATE.json').exists() or not (folder/'DECISIONS.json').exists():continue
        old,dec=read(folder/'PACKET_PRIVATE.json'),read(folder/'DECISIONS.json')
        for oldkey,raw in old['payloads'].items():
            row=raw.get('record',raw)
            if not {'question','gold_answer','raw_base_answer'}<=set(row):continue
            key=semantic(row)
            if key not in payloads:continue
            if oldkey in dec['decisions']:votes.setdefault(key,set()).add(tuple(dec['decisions'][oldkey][f] for f in fields))
            elif oldkey in dec.get('unknown',[]):failed.add(key)
    known,pending,unknown={},[],[]
    for key in sorted(payloads):
        choices=votes.get(key,set())
        if len(choices)>1 or not choices and key in failed:unknown.append(key)
        elif choices:known[key]=dict(opaque_query_id=key,**dict(zip(fields,next(iter(choices)))))
        else:pending.append(key)
    return dict(payloads=payloads,consumers=consumers,known_valid=known,pending=pending,
        inherited_unknown=unknown,contract=read(cfg['accepted_contract']))


def score(cfg,folder,stage,values,history):
    control=Path(cfg['control']);folder.mkdir()
    payload=packet(cfg,values,history);write(folder/'PACKET_PRIVATE.json',payload)
    clock=read(control/'clock.json')
    settings=dict(root=str(folder),private_root=cfg['private_root'],clock=str(control/'clock.json'),
        worktree=cfg['worktree'],source=cfg['source'],common_source=cfg['common_source'],
        cli=cfg['cli'],report=cfg['report'],neutral_entry=str(control/'s.py'),stage=stage,
        t5_source=str(Path(cfg['worktree'])/'experiments/med_t5_core_v1'),base_mask=str(control/'base/BASE_MASK_FROZEN.json'))
    write(folder/'config.json',settings)
    if stage=='t5_base':
        events=ssh(cfg,'from pathlib import Path\nimport json\nr=Path('+repr(cfg['remote_root'])+')\nprint(json.dumps(json.loads((r/"SOURCE_INPUTS.private.json").read_text())["events"]))')
        write(folder/'EVENTS_FROZEN.private.json',dict(events=events))
    env=dict(os.environ,T4_CONFIG=str(folder/'config.json'),T4_MODE='guard',CURRENT_SOURCE=str(Path(cfg['worktree'])/'experiments/med_t4_core_v1/score.py'))
    with (folder/'guard.log').open('x') as log:
        result=subprocess.run([sys.executable,str(control/'s.py')],env=env,cwd=folder,stdout=log,stderr=subprocess.STDOUT,
            timeout=max(1,int(clock['hard_end']-time.time()-60)),start_new_session=True)
    assert result.returncode==0 and read(folder/'GUARD_EXIT.json')['code']==0
    durable=Path(cfg['private_root'])/folder.name;durable.mkdir(exist_ok=True)
    for name in ['PACKET_PRIVATE.json','DECISIONS.json','BASE_MASK_FROZEN.json','COMPLETE.json','GUARD_EXIT.json','COST.json','BATCH_MANIFEST.json']:
        if (folder/name).exists():shutil.copy2(folder/name,durable/name)
    return read(folder/'COMPLETE.json')


def publish(cfg):
    sys.path.insert(0,str(Path(cfg['worktree'])/'experiments/med_t5_data_v1'))
    from job import publish as public_delivery
    public_delivery(cfg)


def deployment_code(root,config,sources,clock):
    code='CONFIG='+repr(config)+'\nSOURCES='+repr(sources)+'\nCLOCK='+repr(clock)+'\nROOT='+repr(root)+'\n'+r'''
from pathlib import Path
import json,sys,os
cfg=CONFIG;r=Path(ROOT);n=Path(cfg['neutral'])
assert not r.exists() and not n.exists()
r.mkdir(parents=True,mode=0o700);n.mkdir(mode=0o700);(r/'tools').mkdir()
p=r/'.probe';p.write_text('ok');assert p.read_text()=='ok';p.unlink()
fs=os.statvfs(r);assert fs.f_bavail*fs.f_frsize>8*1024**3
for name in ['tmp','cuda','torch','rpc']:(n/name).mkdir()
for name,source in SOURCES.items():(r/'tools'/name).write_text(source)
entry="import os,sys,runpy\nsys.path.insert(0,os.environ['T5_TOOLS'])\nrunpy.run_path(os.environ['CURRENT_SOURCE'],run_name='__main__')\n"
for name in ['c.py','b.py','g.py','w.py','e.py']:(n/name).write_text(entry)
(r/'CLOCK.json').write_text(json.dumps(CLOCK))
(r/'CONFIG.private.json').write_text(json.dumps(cfg))
os.environ.update(json.loads(Path(cfg['environment']).read_text()))
os.environ['T5_REMOTE_ROOT']=str(r);os.environ['T5_TOOLS']=str(r/'tools')
sys.path.insert(0,str(r/'tools'))
import remote_ops
print(json.dumps(remote_ops.base(r)))
'''
    compile(code,'remote_t5_deployment','exec')
    return code


def main(cfg):
    control=Path(cfg['control']);private=Path(cfg['private_root']);w=Path(cfg['worktree'])
    with (control/'CONTROLLER_ONCE').open('x'):pass
    transfer_control=Path(cfg['transfer_control']);transfer_report=Path(cfg['transfer_report'])
    while not (transfer_control/'DATA_COMPLETE.json').exists() or not (transfer_report/'TRANSFER_COMPLETE.json').exists():
        if (transfer_control/'FAILURE.json').exists():raise RuntimeError('Dataset transfer failed; no T5 model work started')
        assert time.time()<cfg['dependency_hard_end']-120
        write(control/'PROGRESS.json',dict(phase='WAIT_VERIFIED_DATA_TRANSFER',time=time.time()))
        time.sleep(15)
    assert read(transfer_report/'TRANSFER_COMPLETE.json')['status']=='COMPLETE'
    from sources import build
    sources=build(w);sources['remote_ops.py']=(w/'experiments/med_t5_core_v1/remote_ops.py').read_text()
    clock=dict(original_start=time.time(),hard_end=time.time()+23*3600,
        scope='T5 input admission, Base generation/CORE qualification, fixed MDU training and evaluation; no clock reset')
    write(control/'clock.json',clock);write(private/'CLOCK.json',clock)
    remote_cfg=dict(cfg['remote_config'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip())
    code=deployment_code(cfg['remote_root'],remote_cfg,sources,clock)
    started=ssh(cfg,code,1500);write(private/'BASE_START.private.json',started)
    public_admission={k:v for k,v in started['source_admission'].items() if k!='excluded_case_ids'}
    write(Path(cfg['report'])/'INPUT_ADMISSION.json',public_admission)
    write(control/'PROGRESS.json',dict(phase='BASE_GENERATION_RUNNING',time=time.time()))
    wait_phase(cfg,'BASE_GENERATION_COMPLETE.json','BASE_CHAIN_EXIT.json',['BASE_CHAIN_FAILURE.json'])
    values=ssh(cfg,'from pathlib import Path\nimport json\nr=Path('+repr(cfg['remote_root'])+')\nprint(json.dumps([json.loads(p.read_text()) for p in sorted((r/"base_outputs").glob("*.json"))]))',300)
    write(private/'BASE_OUTPUTS.private.json',values)
    score(cfg,control/'base','t5_base',values,cfg['history_roots'])
    mask=read(control/'base/BASE_MASK_FROZEN.json')
    if not mask['confirmed_usable_cases']:
        write(Path(cfg['report'])/'RESULTS.json',dict(status='NO_CONFIRMED_USABLE_PRIORS',time=time.time(),Base_qualification=read(control/'base/COMPLETE.json')))
        publish(cfg);write(control/'COMPLETE.json',dict(status='COMPLETE_NO_ADMITTED_EDITING',time=time.time()));return
    command='from pathlib import Path\nimport json,sys,os\nr=Path('+repr(cfg['remote_root'])+')\ncfg=json.loads((r/"CONFIG.private.json").read_text())\n(r/"BASE_MASK_FROZEN.json").write_text('+repr(json.dumps(mask))+')\nos.environ["T5_TOOLS"]=str(r/"tools");sys.path.insert(0,str(r/"tools"))\nimport remote_ops\nprint(json.dumps(remote_ops.train(r)))'
    started=ssh(cfg,command,900);write(private/'TRAIN_START.private.json',started)
    write(Path(cfg['report'])/'TRAINING_ADMISSION.json',started['admission'])
    wait_phase(cfg,'PIPELINE_COMPLETE.json','GUARD_EXIT.json',['PIPELINE_FAILURE.json'])
    values=ssh(cfg,'from pathlib import Path\nimport json\nr=Path('+repr(cfg['remote_root'])+')\nprint(json.dumps([json.loads(p.read_text()) for p in sorted((r/"outputs").glob("*.json"))]))',300)
    assert len(values)==started['admission']['total_consumers']
    write(private/'METHOD_OUTPUTS.private.json',values)
    score(cfg,control/'method','t5_method',values,[str(control/'base')]+cfg['history_roots'])
    state=remote_state(cfg,['GUARD_EXIT.json','EVAL_GENERATION_COMPLETE.json','BASE_GENERATION_COMPLETE.json'])
    write(Path(cfg['report'])/'EXECUTION_AUDIT.json',dict(status='COMPLETE',time=time.time(),clock=clock,
        source_model_generation_precision='FP32 for both Base and MDU; archived generation dtype remains historical metadata',
        stages=state,main150_modified=False,transfer_artifacts_read_only=True,source_raw_images_unchanged=True))
    publish(cfg);write(control/'COMPLETE.json',dict(status='T5_CORE_CHAIN_COMPLETE',time=time.time()))


if __name__=='__main__':
    config=read(os.environ['T5_CONFIG'])
    try:main(config)
    except BaseException:
        write(Path(config['control'])/'FAILURE.json',dict(time=time.time(),error=traceback.format_exc()));raise
