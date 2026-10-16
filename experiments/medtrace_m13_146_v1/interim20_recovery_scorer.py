"""Serial sealed scorer with an unauthenticated TLS check before charging each batch."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import traceback


def connection_check(work, sibling, bundle, source, judge):
    sandbox = work/'connection.sb'
    sandbox.write_text(judge.profile(work, [work, sibling], bundle, source))
    env = dict(os.environ)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env[key] = 'http://127.0.0.1:7897'
    env.update(NO_PROXY='', no_proxy='')
    attempts = int(os.environ.get('CONNECTION_PREFLIGHT_ATTEMPTS', '1'))
    assert attempts in (0, 1, 3)
    deadline = float(os.environ['CONNECTION_PREFLIGHT_DEADLINE']) if attempts == 0 else float('inf')
    history = []
    while attempts == 0 or len(history) < attempts:
        assert time.time() < deadline, 'Connection wait reached project deadline'
        if history:
            time.sleep(min(30, 2 ** min(len(history), 5)))
        result = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(sandbox), '/usr/bin/curl',
            '--proxy', 'http://127.0.0.1:7897', '--connect-timeout', '8', '--max-time', '15',
            '-sS', '-o', '/dev/null', '-w', '%{http_code}',
            'https://chatgpt.com/backend-api/codex/models'], env=env, capture_output=True, text=True)
        history.append(dict(epoch=time.time(), exit_code=result.returncode, HTTP=result.stdout,
            errors=result.stderr[-1000:], model_call=False))
        receipt = dict(history[-1], checks=history, maximum_checks=attempts,
            limit='Transport reachability only; not authenticated inference readiness')
        (work/'CONNECTION_PREFLIGHT.json').write_text(json.dumps(receipt, indent=2)+'\n')
        if result.returncode == 0 and result.stdout in ('200', '401', '403'):
            return
    raise AssertionError('Proxy/TLS preflight failed before reservation')



def enable_model_retries(scorer):
    original_call = scorer.call
    original_runner = scorer.judge.run_batch
    def call(bundle, batch, work, sibling):
        attempt = 0
        while True:
            operator = bundle/'operator'/'attempts'/str(attempt)
            (operator/'execution_evidence').mkdir(parents=True)
            (operator/'responses').mkdir()
            attempt_work = work/('attempt_'+str(attempt)); attempt_work.mkdir()
            if attempt:
                connection_check(attempt_work, sibling, bundle, scorer.SOURCE, scorer.judge)
                scorer.remote(dict(action='retry_charge', batch_id=batch['batch_id'], attempt=attempt, evidence=evidence))
            def runner(*args, **kwargs):
                return original_runner(*args, **kwargs, output_operator=operator)
            scorer.judge.run_batch = runner
            try:
                original_call(bundle, batch, attempt_work, sibling)
            except Exception:
                if not (operator/'execution_evidence'/(batch['batch_id']+'.json')).exists():
                    raise
            finally:
                scorer.judge.run_batch = original_runner
            ep = operator/'execution_evidence'/(batch['batch_id']+'.json')
            evidence = json.loads(ep.read_text())
            scorer.publish_file('attempts/'+batch['batch_id']+'/'+str(attempt)+'.json', ep.read_bytes())
            if evidence['status']=='FORMAT_VALID':
                for folder in ('execution_evidence','responses'):
                    shutil.copyfile(operator/folder/(batch['batch_id']+'.json'),bundle/'operator'/folder/(batch['batch_id']+'.json'))
                return
            assert evidence['status']=='FAILED_NO_RETRY' and not evidence.get('tool_event_types')
            assert all(evidence['isolation_checks'].values())
            attempt += 1
            time.sleep(min(30, 2 ** min(attempt, 5)))
    scorer.call = call


def main():
    spec = importlib.util.spec_from_file_location('sealed_scorer', os.environ['RECOVERY_SCORER_FILE'])
    scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
    if os.environ.get('CM_MODEL_RETRY')=='1':
        enable_model_retries(scorer)
    original = scorer.prepare
    def prepare(rows):
        value = original(rows)
        _, bundle, work, sibling = value
        connection_check(work, sibling, bundle, scorer.SOURCE, scorer.judge)
        return value
    scorer.prepare = prepare
    try:
        scorer.main()
    except BaseException as error:
        failure = dict(error=repr(error), traceback=traceback.format_exc(), epoch=time.time(),
                       automatic_retry=False)
        (scorer.LOCAL/'SCORER_FAILURE.json').write_text(json.dumps(failure, indent=2)+'\n')
        scorer.publish_file('SCORER_FAILURE.json', json.dumps(failure).encode())
        raise


if __name__ == '__main__': main()
