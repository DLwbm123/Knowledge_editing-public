"""Connection retries cannot call a model or reserve a queue item."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import interim20_recovery_scorer as scorer


def main():
    failed=SimpleNamespace(returncode=35,stdout='000',stderr='TLS failure')
    success=SimpleNamespace(returncode=0,stdout='401',stderr='')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        judge=SimpleNamespace(profile=lambda *args:'sandbox')
        for attempts,responses in [('1',[failed]),('0',[failed]*4+[success])]:
            with patch.dict(os.environ,CONNECTION_PREFLIGHT_ATTEMPTS=attempts,CONNECTION_PREFLIGHT_DEADLINE='9999999999'),patch.object(scorer.subprocess,'run',side_effect=responses) as run,patch.object(scorer.time,'sleep'):
                try:scorer.connection_check(root,root,root,root,judge)
                except AssertionError:assert attempts=='1'
                else:assert attempts=='0'
                assert run.call_count==len(responses)
                assert all(call.args[0][3]=='/usr/bin/curl' for call in run.call_args_list)
                saved=json.loads((root/'CONNECTION_PREFLIGHT.json').read_text())
                assert len(saved['checks'])==len(responses) and all(not r['model_call'] for r in saved['checks'])
    print('PASS: unlimited connection-only retries, original single check preserved, evidence retained')


if __name__=='__main__':main()

# Exercise failed model attempt -> charged retry -> valid result, without inference.
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);bundle=root/'bundle';work=root/'work';work.mkdir()
    for folder in ('execution_evidence','responses'):(bundle/'operator'/folder).mkdir(parents=True)
    calls=[];charges=[];published=[]
    def run_batch(bundle,batch,work,sibling,output_operator=None):
        calls.append(1);valid=len(calls)==2
        ev=dict(status='FORMAT_VALID' if valid else 'FAILED_NO_RETRY',isolation_checks={'own':True},tool_event_types=[])
        (output_operator/'execution_evidence/b.json').write_text(json.dumps(ev))
        if valid:(output_operator/'responses/b.json').write_text('{}')
        else:raise RuntimeError('synthetic network failure')
    judge=SimpleNamespace(run_batch=run_batch)
    fake=SimpleNamespace(judge=judge,SOURCE=root,remote=lambda op:charges.append(op),publish_file=lambda *args:published.append(args))
    fake.call=lambda b,ba,w,s:judge.run_batch(b,ba,w,s)
    scorer.enable_model_retries(fake)
    with patch.object(scorer,'connection_check'),patch.object(scorer.time,'sleep'):
        fake.call(bundle,{'batch_id':'b'},work,root)
    assert len(calls)==2 and len(charges)==1 and charges[0]['action']=='retry_charge'
    assert len(published)==2 and json.loads((bundle/'operator/execution_evidence/b.json').read_text())['status']=='FORMAT_VALID'
    assert (bundle/'operator/attempts/0/execution_evidence/b.json').exists()
print('PASS: failed attempts retained, one charge per retry, stop immediately on valid result')
