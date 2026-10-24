"""CPU fault checks with synthetic prefixes and mocked HTTP; no credentials/network."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import huggingface_hub as hub
from huggingface_hub._local_folder import get_local_download_paths
import requests


def test(mode):
    with tempfile.TemporaryDirectory(dir=os.environ.get('TEST_PARENT')) as tmp:
        root=Path(tmp)
        local=root/'model'
        size=8*1024*1024+51
        prefix=b'preserved-prefix!'
        names=['a.safetensors','b.safetensors']
        paths={name:get_local_download_paths(local,name) for name in names}
        for path in paths.values():
            path.incomplete_path('synthetic').write_bytes(prefix)
        credential=root/'credential'
        credential.write_text('synthetic-only')
        cfg=dict(original_start=time.time(),hard_end=time.time()+60,credential_file=str(credential),
            candidate_path=str(local),candidate_id='synthetic',revision='fixed',
            download_sizes={name:size for name in names},download_files=names)
        (root/'CONFIG.json').write_text(json.dumps(cfg))
        class Response:
            status_code=206
            def __init__(self, start, end):
                self.length=end-start+1
                self.headers={'content-range':f'bytes {start}-{end}/{size}' if mode!='bad_range' else 'bytes 0-1/2'}
            def __enter__(self):
                return self
            def __exit__(self,*args):
                return False
            def iter_content(self, chunk):
                remaining=self.length-(1 if mode=='short' else 0)
                while remaining:
                    count=min(remaining,chunk)
                    yield b'z'*count
                    remaining-=count
        class Session:
            def get(self,url,headers,**kwargs):
                start,end=map(int,headers['Range'].split('=')[1].split('-'))
                assert start>=len(prefix)
                return Response(start,end)
        spec=importlib.util.spec_from_file_location('range_under_test',Path(__file__).with_name('ranged_download.py'))
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(os.environ,{'TRIAL_ROOT':tmp}), patch.object(requests,'Session',Session), \
             patch.object(hub,'hf_hub_url',lambda *a,**k:'https://synthetic.invalid'), \
             patch.object(hub,'get_hf_file_metadata',lambda *a,**k:SimpleNamespace(commit_hash='fixed',size=size,etag='synthetic')):
            failed=False
            try:
                module.main()
            except RuntimeError:
                failed=True
        assert failed==(mode!='normal')
        assert not credential.exists()
        assert (root/'DOWNLOAD_COMPLETE.json').exists()==(mode=='normal')
        manifest=json.loads((root/'RANGE_MANIFEST.json').read_text())
        for name,path in paths.items():
            actual=path.file_path if mode=='normal' else path.incomplete_path('synthetic')
            with actual.open('rb') as stream:
                assert stream.read(len(prefix))==prefix
                if mode=='normal':
                    assert stream.read()==b'z'*(size-len(prefix))
            assert manifest['files'][name]['preserved_prefix_bytes']==len(prefix)
        if mode=='normal':
            progress=json.loads((root/'RANGE_PROGRESS.json').read_text())
            assert progress['completed_weight_bytes']==2*size and progress['completed_ranges']==4
        return dict(case=mode,passed=True,synthetic_only=True)


if __name__=='__main__':
    print(json.dumps([test(mode) for mode in ['normal','bad_range','short']]))
