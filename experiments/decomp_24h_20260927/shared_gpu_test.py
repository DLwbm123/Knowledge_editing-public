"""CPU check: launch still rejects insufficient memory and accepts shared capacity."""
import ast, os, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
source=Path(__file__).with_name('controller.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='launch')
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);(root/'logs').mkdir()
 sub=SimpleNamespace(STDOUT=-2,check_output=Mock(return_value='29999'),Popen=Mock(return_value=SimpleNamespace(pid=123)))
 ns=dict(subprocess=sub,os=os,ROOT=root,time=SimpleNamespace(time=lambda:0),log_limit=lambda:None)
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<launch>','exec'),ns)
 gpu=dict(index=6,uuid='test-device')
 assert ns['launch'](gpu) is None
 sub.Popen.assert_not_called()
 sub.check_output.return_value='60113'
 assert ns['launch'](gpu)['pid']==123
 assert sub.Popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES']=='test-device'
 assert all('--query-compute-apps=gpu_uuid' not in c.args[0] for c in sub.check_output.call_args_list)
print('PASS shared-capacity launch and insufficient-memory rejection')
