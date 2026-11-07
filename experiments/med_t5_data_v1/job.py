"""Guard a bounded local T5 build and upload through neutral process paths."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def read(p):return json.loads(Path(p).read_text())


def write(p,value):
    p=Path(p);temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(p)


def owned_bytes(root):
    if not root.exists():return 0
    size=0
    for p in root.rglob('*'):
        try:
            if p.is_file() and not p.is_symlink():size+=p.stat().st_size
        except FileNotFoundError:pass
    return size


def remote(cfg,code,timeout=45):
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',cfg['ssh_host'],'python3','-'],
        input=code,text=True,capture_output=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr[-2000:])
    return json.loads(result.stdout)


def publish(cfg):
    w=Path(cfg['worktree']);report=Path(cfg['report']);relative=str(report.relative_to(w))
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=w,text=True).strip()
    subprocess.run(['git','add',relative],cwd=w,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=w,text=True).splitlines()
    assert staged and all(p.startswith(relative+'/') for p in staged),'Publication scope changed concurrently'
    subprocess.run(['git','commit','-m','Report '+report.name+' results and execution evidence'],cwd=w,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=w,text=True).strip()
    env=dict(os.environ,HTTP_PROXY=cfg['proxy'],HTTPS_PROXY=cfg['proxy'],ALL_PROXY=cfg['proxy'],
        NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1')
    subprocess.run(['git','-c','http.proxy='+cfg['proxy'],'push','origin','HEAD:refs/heads/'+branch],cwd=w,env=env,check=True)
    actual=subprocess.check_output(['git','-c','http.proxy='+cfg['proxy'],'ls-remote','origin','refs/heads/'+branch],cwd=w,env=env,text=True).split()[0]
    assert actual==head
    artifact=next(name for name in ('RESULTS.json','SUMMARY.json','INPUT_ADMISSION.json') if (report/name).is_file())
    url=cfg['public_raw_base']+'/'+head+'/'+relative+'/'+artifact
    result=subprocess.run(['curl','--proxy',cfg['proxy'],'--connect-timeout','8','--max-time','30','-sS',
        '-o','/dev/null','-w','%{http_code}',url],text=True,capture_output=True,env=env)
    assert result.returncode==0 and result.stdout=='200'
    write(Path(cfg['private_root'])/'PUBLICATION_RECEIPT.json',dict(head=head,branch=branch,remote_verified=True,anonymous_HTTP=200))


def worker(cfg):
    control=Path(cfg['control']);package=Path(cfg['package'])
    with (control/'WORKER_ONCE').open('x'):pass
    write(control/'PROGRESS.json',dict(phase='BUILD',time=time.time()))
    result=subprocess.run([sys.executable,cfg['build_entry']],env=dict(os.environ,CURRENT_SOURCE=cfg['build_source']),cwd=control)
    assert result.returncode==0,'Data construction failed'
    summary=read(package/'SUMMARY.json');write(Path(cfg['report'])/'SUMMARY.json',summary)
    for phase,source,target,extra in [('MAIN_UPLOAD','archives/',cfg['remote_neutral']+'/incoming/',
            ['--include=PadChest_GR.zip.[0-9][0-9][0-9]','--exclude=*']),
        ('T5_UPLOAD','dataset/',cfg['remote_neutral']+'/dataset/',['--exclude=._*'])]:
        write(control/'PROGRESS.json',dict(phase=phase,time=time.time()))
        result=subprocess.run(['rsync','-rt','--partial','--partial-dir=.partial']+extra+[source,cfg['ssh_host']+':'+target],cwd=control)
        assert result.returncode==0,(phase,result.returncode)
        write(control/(phase+'_EXIT.json'),dict(code=0,time=time.time()))
    # One low-cost target check: expected paths, sizes, manifests and case count.
    expected={p.name:p.stat().st_size for p in (Path(cfg['source'])/'Padchest_GR_files').glob('PadChest_GR.zip.[0-9][0-9][0-9]')}
    expected_images={p.name:p.stat().st_size for p in (package/'images').iterdir() if p.suffix=='.png' and not p.name.startswith('._')}
    check='from pathlib import Path\nimport json\nr=Path('+repr(cfg['remote_main'])+');d=Path('+repr(cfg['remote_package'])+')\nexpected='+repr(expected)+'\nimages='+repr(expected_images)+'\nassert all((r/n).is_file() and (r/n).stat().st_size==size for n,size in expected.items())\nassert all((d/"images"/n).is_file() and (d/"images"/n).stat().st_size==size for n,size in images.items())\nrows=[json.loads(x) for x in (d/"T5_CANONICAL.private.jsonl").read_text().splitlines()]\nassert len(rows)=='+str(summary['canonical_cases'])+'\nassert all((d/t[k]["relative_image_path"]).is_file() for t in rows for k in ["edit_case","eval_case"])\nprint(json.dumps({"archive_parts":len(expected),"archive_bytes":sum(expected.values()),"images":len(images),"image_bytes":sum(images.values()),"cases":len(rows),"path_and_size_check":"PASS","byte_identical_not_asserted":True}))'
    receipt=remote(cfg,check)
    write(Path(cfg['private_root'])/'TRANSFER_RECEIPT.private.json',receipt)
    write(Path(cfg['report'])/'TRANSFER_COMPLETE.json',dict(status='COMPLETE',time=time.time(),**receipt,
        download_required=False,existing_DataP_archive_reused=True,new_model_or_Judge_calls=0))
    write(control/'DATA_COMPLETE.json',dict(time=time.time(),status='DATA_BUILT_AND_UPLOADED'))
    try:publish(cfg)
    except Exception:
        write(control/'PUBLICATION_FAILURE.json',dict(time=time.time(),error=traceback.format_exc()))
        raise
    write(control/'COMPLETE.json',dict(time=time.time(),status='DATA_AND_PUBLIC_DELIVERY_COMPLETE'))


def guard(cfg):
    control=Path(cfg['control']);clock=cfg['clock']
    with (control/'GUARD_ONCE').open('x'):pass
    with (control/'worker.log').open('x') as log:
        child=subprocess.Popen([sys.executable,cfg['entry']],env=dict(os.environ,T5_MODE='worker'),cwd=control,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(control/'WORKER_IDENTITY.json',dict(pid=child.pid,time=time.time()))
    try:
        while child.poll() is None:
            local=owned_bytes(Path(cfg['package']))+owned_bytes(control)
            fs=os.statvfs(Path(cfg['package']).parent)
            assert local+1024**3<cfg['new_local_bytes_limit'] and fs.f_bavail*fs.f_frsize>8*1024**3
            assert time.time()<clock['hard_end']-90
            check='from pathlib import Path\nimport os,json\nr=Path('+repr(cfg['remote_main'])+');d=Path('+repr(cfg['remote_package'])+')\nsize=sum(p.stat().st_size for base in [r,d] for p in base.rglob("*") if p.is_file() and not p.is_symlink())\ns=os.statvfs(r)\nprint(json.dumps({"bytes":size,"free":s.f_bavail*s.f_frsize}))'
            usage=remote(cfg,check)
            assert usage['bytes']+2*1024**3<cfg['new_remote_bytes_limit'] and usage['free']>8*1024**3
            write(control/'GUARD_HEALTH.json',dict(time=time.time(),local_new_bytes=local,remote_new_bytes=usage['bytes'],hard_end=clock['hard_end']))
            time.sleep(15)
    except BaseException:
        write(control/'STOP.json',dict(time=time.time(),error=traceback.format_exc()))
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.wait();write(control/'GUARD_EXIT.json',dict(code=code,time=time.time(),clock=clock));raise SystemExit(code)


if __name__=='__main__':
    config=read(os.environ['T5_CONFIG'])
    if os.environ.get('T5_MODE')=='worker':
        try:worker(config)
        except BaseException:
            write(Path(config['control'])/'FAILURE.json',dict(time=time.time(),error=traceback.format_exc()));raise
    else:guard(config)
