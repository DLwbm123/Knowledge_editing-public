"""Serial sealed scorer with an unauthenticated TLS check before charging each batch."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def connection_check(work, sibling, bundle, source, judge):
    sandbox = work/'connection.sb'
    sandbox.write_text(judge.profile(work, [work, sibling], bundle, source))
    env = dict(os.environ)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env[key] = 'http://127.0.0.1:7897'
    env.update(NO_PROXY='', no_proxy='')
    result = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(sandbox), '/usr/bin/curl',
        '--proxy', 'http://127.0.0.1:7897', '--connect-timeout', '8', '--max-time', '15',
        '-sS', '-o', '/dev/null', '-w', '%{http_code}',
        'https://chatgpt.com/backend-api/codex/models'], env=env, capture_output=True, text=True)
    receipt = dict(epoch=time.time(), exit_code=result.returncode, HTTP=result.stdout,
        errors=result.stderr[-1000:], model_call=False,
        limit='Transport reachability only; not authenticated inference readiness')
    (work/'CONNECTION_PREFLIGHT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    assert result.returncode == 0 and result.stdout in ('200', '401', '403'), 'Proxy/TLS preflight failed before reservation'


def main():
    spec = importlib.util.spec_from_file_location('sealed_scorer', os.environ['RECOVERY_SCORER_FILE'])
    scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
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
