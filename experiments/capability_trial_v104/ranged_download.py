"""Resume official immutable weight ranges without rewriting existing prefixes."""
import concurrent.futures
import json
import os
from pathlib import Path
import threading
import time


def main():
    import requests
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    from huggingface_hub._local_folder import get_local_download_paths, write_download_metadata
    root=Path(os.environ['TRIAL_ROOT'])
    cfg=json.loads((root/'CONFIG.json').read_text())
    credential=Path(cfg['credential_file'])
    token=credential.read_text().strip()
    local=Path(cfg['candidate_path'])
    manifest_path=root/'RANGE_MANIFEST.json'
    assert not manifest_path.exists(), 'Range takeover is once-only; do not infer progress from sparse length'
    event=threading.Event()
    sessions=threading.local()
    files={}
    def save():
        tmp=manifest_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(original_start=cfg['original_start'],hard_end=cfg['hard_end'],
            chunk_bytes=8*1024*1024,max_workers=4,files=files,time=time.time()),indent=2))
        tmp.replace(manifest_path)
    def check():
        assert not event.is_set() and time.time()<cfg['hard_end']-15
        assert not (root/'STOP.json').exists()
    def fetch(task):
        name,start,end=task
        item=files[name]
        if not hasattr(sessions,'http'):
            sessions.http=requests.Session()
        for attempt in range(2):
            check()
            written=0
            try:
                with sessions.http.get(item['url'],headers={'Authorization':'Bearer '+token,
                        'Range':f'bytes={start}-{end}'},stream=True,timeout=(15,30)) as response:
                    expected=f'bytes {start}-{end}/{item["expected_bytes"]}'
                    if response.status_code!=206 or response.headers.get('content-range')!=expected:
                        raise ValueError('Official range response did not match requested immutable bytes')
                    fd=os.open(item['partial_path'],os.O_WRONLY)
                    try:
                        for block in response.iter_content(256*1024):
                            check()
                            if not block:
                                continue
                            if written+len(block)>end-start+1:
                                raise ValueError('Range response exceeded declared length')
                            offset=start+written
                            assert offset>=item['preserved_prefix_bytes']
                            done=0
                            while done<len(block):
                                count=os.pwrite(fd,block[done:],offset+done)
                                if count<=0:
                                    raise OSError('Short positional write')
                                done+=count
                            written+=len(block)
                    finally:
                        os.close(fd)
                if written!=end-start+1:
                    raise OSError('Incomplete range response')
                return name,start,end,attempt+1
            except (requests.RequestException,OSError):
                if attempt:
                    raise
        raise RuntimeError('Range attempts exhausted')
    try:
        tasks=[]
        for name,size in cfg['download_sizes'].items():
            if not name.endswith('.safetensors'):
                continue
            check()
            url=hf_hub_url(cfg['candidate_id'],name,revision=cfg['revision'],endpoint='https://huggingface.co')
            meta=get_hf_file_metadata(url,token=token,timeout=20)
            assert meta.commit_hash==cfg['revision'] and meta.size==size
            paths=get_local_download_paths(local,name)
            if paths.file_path.exists():
                assert paths.file_path.stat().st_size==size
                continue
            partial=paths.incomplete_path(meta.etag)
            assert partial.exists(), 'Expected existing contiguous prefix missing'
            prefix=partial.stat().st_size
            assert 0<prefix<size
            files[name]=dict(url=url,partial_path=str(partial),final_path=str(paths.file_path),
                expected_bytes=size,preserved_prefix_bytes=prefix,etag=meta.etag,completed=[])
            tasks.extend((name,start,min(start+8*1024*1024,size)-1) for start in range(prefix,size,8*1024*1024))
        assert files and tasks
        tasks.sort(key=lambda t:(t[1]-files[t[0]]['preserved_prefix_bytes'],t[0]))
        save()
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=4)
        futures=[pool.submit(fetch,task) for task in tasks]
        try:
            for future in concurrent.futures.as_completed(futures):
                name,start,end,attempts=future.result()
                files[name]['completed'].append(dict(start=start,end=end,attempts=attempts))
                save()
                complete=sum(f['preserved_prefix_bytes']+sum(c['end']-c['start']+1 for c in f['completed']) for f in files.values())
                (root/'RANGE_PROGRESS.json').write_text(json.dumps(dict(completed_weight_bytes=complete,
                    total_weight_bytes=sum(cfg['download_sizes'][n] for n in cfg['download_sizes'] if n.endswith('.safetensors')),
                    completed_ranges=sum(len(f['completed']) for f in files.values()),time=time.time()),indent=2))
        except BaseException:
            event.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
        for name,item in files.items():
            position=item['preserved_prefix_bytes']
            for part in sorted(item['completed'],key=lambda x:x['start']):
                assert part['start']==position
                position=part['end']+1
            assert position==item['expected_bytes']
            partial=Path(item['partial_path'])
            assert partial.stat().st_size==position
            partial.replace(item['final_path'])
            write_download_metadata(local,name,cfg['revision'],item['etag'])
        for name in cfg['download_files']:
            if not name.endswith('.safetensors'):
                check()
                hf_hub_download(cfg['candidate_id'],name,revision=cfg['revision'],local_dir=local,token=token)
        actual={name:(local/name).stat().st_size for name in cfg['download_files']}
        assert actual==cfg['download_sizes']
        (root/'DOWNLOAD_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),sizes=actual,
            revision=cfg['revision'],verification='preserved contiguous prefixes; exact HTTP206 ranges and complete coverage; known sizes; no added hashes',
            range_manifest='RANGE_MANIFEST.json'),indent=2))
    except BaseException as exc:
        (root/'RANGE_FAILURE.json').write_text(json.dumps(dict(error_type=type(exc).__name__,time=time.time()),indent=2))
        raise RuntimeError('Bounded range download failed; private range ledger retained') from None
    finally:
        credential.unlink(missing_ok=True)


if __name__=='__main__':
    main()
