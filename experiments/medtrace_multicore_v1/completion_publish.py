"""One completion consumer; publish only reviewed aggregate report types."""
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

FILES = ('CPU_CHECKS.json', 'MECHANISM_RESULTS.json', 'EXECUTION_AUDIT.json',
         'ROOT_FIX_FULL_RESULTS.json', 'ROOT_FIX_FULL_REPORT_ZH.md',
         'MULTI_FULL_RESULTS.json', 'MULTI_FULL_REPORT_ZH.md',
         'PAIRED_FULL_RESULTS.json', 'FINAL_REPORT_ZH.md')


def public_check(text, name):
    assert not re.search(r'/data/|/Users/|/home/|core9q_|sk-[A-Za-z0-9]|Bearer ', text)
    if name.endswith('.json'):
        def visit(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    assert key.lower() not in {'query_id', 'image_path', 'raw_answer', 'answer',
                        'patient', 'weights', 'credential', 'per_owner', 'cells', 'paths'}
                    if key.lower() == 'tokens': assert isinstance(item, int) and item >= 0
                    visit(item)
            elif isinstance(value, list):
                for item in value: visit(item)
        visit(json.loads(text))


def main():
    public_check('{"fisher":{"tokens":1514},"count":8}', 'check.json')
    for fixture in ('{"query_id":"private"}', '{"tokens":[1,2]}', '{"paths":["private"]}'):
        try: public_check(fixture, 'check.json')
        except AssertionError: pass
        else: raise AssertionError('Publication boundary self-check failed')
    settings = json.loads(Path(os.environ['DELIVERY_SETTINGS']).read_text())
    local = Path(settings['state']); local.mkdir(exist_ok=True)
    with (local / 'ONCE').open('x'): pass
    checkout = Path(settings['checkout']); run = settings['run']
    env = dict(os.environ, HTTP_PROXY='http://127.0.0.1:7897', HTTPS_PROXY='http://127.0.0.1:7897',
               ALL_PROXY='http://127.0.0.1:7897', NO_PROXY='', no_proxy='')
    def command(args, **kw):
        return subprocess.run(args, cwd=checkout, env=env, text=True, capture_output=True,
                              check=True, timeout=300, **kw).stdout.strip()
    def remote(code):
        return json.loads(command(['ssh', 'pro5000', 'python3 -'], input=code))
    header = 'from pathlib import Path\nimport json\nr=Path(' + repr(run) + ')\n'
    while True:
        state = remote(header + "print(json.dumps(dict(complete=(r/'private/COMPLETE.json').exists(),"
            "exit=json.loads((r/'private/EXIT1.json').read_text()) if (r/'private/EXIT1.json').exists() else None,"
            "failures=[p.name for p in (r/'private').glob('FAILURE_*')],"
            "deadline=json.loads((r/'RUN_MANIFEST.json').read_text())['deadline_epoch'])))")
        assert not state['failures'], 'Execution failure: retain private state and require audit'
        assert time.time() <= state['deadline'] + 600, 'Finite completion deadline exceeded'
        if state['exit'] is not None:
            assert state['exit']['exit_code'] == 0 and state['complete']
            break
        time.sleep(30)
    assert command(['git', 'branch', '--show-current']) == settings['branch']
    assert not command(['git', 'status', '--porcelain']), 'Concurrent checkout changes; do not publish'
    data = remote(header + 'names=' + repr(FILES) + '\n' +
        "print(json.dumps({n:(r/'public'/n).read_text() for n in names if (r/'public'/n).exists()}))")
    assert {'CPU_CHECKS.json', 'MECHANISM_RESULTS.json', 'EXECUTION_AUDIT.json', 'FINAL_REPORT_ZH.md'} <= data.keys()
    mechanism = json.loads(data['MECHANISM_RESULTS.json'])
    if mechanism['mechanism_gate']: assert set(FILES) == data.keys()
    else: assert 'PAIRED_FULL_RESULTS.json' not in data
    assert json.loads(data['EXECUTION_AUDIT.json'])['status'] == 'PASS'
    for name, text in data.items(): public_check(text, name)
    folder = checkout / settings['report_folder']
    for name, text in data.items(): (folder / name).write_text(text)
    command(['git', 'add', *[str(folder / name) for name in data]])
    command(['git', 'commit', '-m', 'Publish completed frozen paired multicore experiment results'])
    command(['git', '-c', 'http.proxy=http://127.0.0.1:7897', '-c', 'http.version=HTTP/1.1',
             'push', 'origin', 'HEAD:refs/heads/' + settings['branch']])
    sha = command(['git', 'rev-parse', 'HEAD'])
    remote_sha = command(['git', '-c', 'http.proxy=http://127.0.0.1:7897', 'ls-remote', 'origin',
                          'refs/heads/' + settings['branch']]).split()[0]
    assert sha == remote_sha
    url = 'https://raw.githubusercontent.com/' + settings['repository'] + '/' + sha + '/' + settings['report_folder'] + '/FINAL_REPORT_ZH.md'
    status = command(['curl', '--proxy', 'http://127.0.0.1:7897', '-sSL', '-o', '/dev/null', '-w', '%{http_code}', url])
    assert status == '200'
    full = json.loads(data['PAIRED_FULL_RESULTS.json']) if mechanism['mechanism_gate'] else None
    result = '完整性能兼顾条件通过' if full and full['scientific_success'] else '未建立完整性能兼顾'
    body = local / 'pr_body.md'
    body.write_text('同一八冻结状态上完成数值边界修复与根核/同秩多核配对验证。外部0.1C预算、90%纯提案收益、步幅规则、seed和7168参数保持不变。\n\n'
        + '新增真实配对见证：' + str(mechanism['additional_real_paired_trials']) + '；预定首选改善：' + str(mechanism['improved_first_selected_owners']) + '。'
        + ('两臂全部八例160步和2316消费者已完成；' + result + '。' if full else '机制门槛未通过，完整训练和新评分未启动。')
        + '\n\n全部阴性、缺失、计数及继承成本保留。历史M1/M3/M13/E比较及匿名聚合区间见公开中文报告。CPU检查与真实执行审计PASS，remote SHA匹配，匿名报告HTTP200。全部DEV，无独立确认或医学安全结论。\n\n'
        + '[中文报告](' + url + ')\n')
    command(['gh', 'pr', 'edit', str(settings['PR']), '--repo', settings['repository'],
             '--title', 'Completed frozen paired root and multicore validation', '--body-file', str(body)])
    receipt = dict(commit=sha, remote_SHA_match=True, anonymous_HTTP=200, PR=settings['PR'],
                   epoch=time.time(), report=url, source_execution=settings['source_commit'])
    remote(header + "(r/'private/DELIVERY1.json').write_text(" + repr(json.dumps(receipt)) + ")\n"
        + "p=r/'public/PROGRESS.json';v=json.loads(p.read_text());v.update(publication_complete=True,whole_task_complete=True);p.write_text(json.dumps(v)+'\\n');print(json.dumps({'status':'DELIVERED'}))")
    (local / 'DELIVERY1.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    try: main()
    except BaseException:
        state = json.loads(Path(os.environ['DELIVERY_SETTINGS']).read_text())['state']
        Path(state, 'DELIVERY_BLOCKED.json').write_text(json.dumps(dict(error=traceback.format_exc(), retry=False)))
        raise
