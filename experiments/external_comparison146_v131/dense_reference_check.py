"""Compare actual diagnostic keys with the native direct MEMIT solve on CPU."""
import json
import os
from pathlib import Path
import signal
import time
import traceback
import torch

root = Path(os.environ['V131_ROOT'])
out = Path(os.environ['AUDIT_OUTPUT'])


def write(name, value):
    path = out/name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def main():
    started = time.time()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('CPU reference limit')))
    signal.alarm(600)
    torch.set_num_threads(8)
    records = torch.load(out/'SOLVE_RECORDS.pt',map_location='cpu',weights_only=True)
    assert {(v['owner'],v['layer']) for v in records} == {(i,l) for i in (1,6) for l in range(4,9)}
    assert len(records) == 10 and not (out/'DENSE_REFERENCE_RESULT.json').exists()
    rows = []
    for record in records:
        layer,key,adjusted = record['layer'],record['key'],record['adjusted']
        covariance = torch.load(root/'STATISTICS'/f'covariance_{layer}.pt',
                                map_location='cpu',weights_only=True).double()
        matrix = 15000 * covariance + torch.outer(key,key)
        expected = torch.linalg.solve(matrix,key)
        relative = float((expected-adjusted).norm()/expected.norm())
        residual = float((matrix@adjusted-key).norm()/key.norm())
        rows.append(dict(owner=record['owner'],layer=layer,relative_adjusted_key_error=relative,
                         max_abs=float((expected-adjusted).abs().max()),
                         relative_equation_residual=residual,passed=relative<=1e-8 and residual<=1e-5))
        write('DENSE_REFERENCE_PROGRESS.json',dict(time=time.time(),completed=len(rows),rows=rows))
        print('DENSE_REFERENCE_COMPLETE',len(rows),flush=True)
        del covariance,matrix,expected
    write('DENSE_REFERENCE_RESULT.json',dict(status='PASS' if all(r['passed'] for r in rows)
          else 'NUMERICAL_DISCREPANCY',rows=rows,seconds=time.time()-started,GPU_calls=0,Judge_calls=0))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write('DENSE_REFERENCE_FAILURE.json',dict(time=time.time(),error=traceback.format_exc()))
        raise
