"""Private official CLI calls: fresh tool-free contexts, Seatbelt, receipts/cache.

No credential copying, billing impersonation, public CI, or scientific retries.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

VERSION='h-source-grounded-v2-20261006/v1-prompts+isolation-v2'
def now():return datetime.now(timezone.utc).isoformat()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def save(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2)

def flags(work):
    values=dict(project_doc_max_bytes='0',developer_instructions='""',web_search='"disabled"',
        model_reasoning_effort='"medium"',model_reasoning_summary='"none"',
        sqlite_home=json.dumps(str(work/'state')),log_dir=json.dumps(str(work/'logs')),
        mcp_servers='{}',**{'memories.use_memories':'false','memories.generate_memories':'false',
        'skills.include_instructions':'false','suppress_unstable_features_warning':'true',
        'include_permissions_instructions':'false','include_collaboration_mode_instructions':'false',
        'include_apps_instructions':'false','model_provider':'"isolated_openai"',
        'model_providers.isolated_openai':'{name="OpenAI",wire_api="responses",requires_openai_auth=true,request_max_retries=0,stream_max_retries=0}',
        'default_permissions':'"review"',
        'permissions.review.filesystem':'{":root"="deny",":minimal"="read",":workspace_roots"={"."="read"}}',
        'permissions.review.network.enabled':'false','approval_policy':'"never"'})
    disabled=('memories apps plugins skill_search skill_mcp_dependency_install multi_agent multi_agent_v2 '
        'hooks shell_tool unified_exec shell_snapshot browser_use browser_use_external computer_use '
        'image_generation view_image code_mode code_mode_host tool_suggest goals sleep_tool '
        'workspace_dependencies personality unbounded_connection_retries').split()
    values.update({'features.'+k:'false' for k in disabled})
    values['features.skip_host_skill_discovery']='true';values['features.respect_system_proxy']='true'
    return [a for k,v in values.items() for a in ('-c',k+'='+v)]

class Runner:
    def __init__(self,root,cli,model='gpt-6.1-sol',max_calls=300,timeout=300):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);os.chmod(self.root,0o700)
        self.cli=str(cli);self.model=model;self.max_calls=max_calls;self.timeout=timeout
        self.cache=self.root/'cache';self.cache.mkdir(exist_ok=True)
        self.receipts=self.root/'receipts';self.receipts.mkdir(exist_ok=True)
        previous=[json.loads(p.read_text()) for p in self.receipts.glob('*.json')]
        self.count=len(previous);self.work=[Path(r['work_directory']) for r in previous];self.cache_hits=0
        self.version=subprocess.check_output([self.cli,'--version'],text=True).strip()
    def profile(self,work):
        home=Path.home()
        blocked=[home/p for p in ('Desktop','Documents','Downloads','.codex/memories','.codex/skills','.agents','.codex/plugins','.codex/sessions','.codex/archived_sessions','.codex/attachments')]
        blocked += [Path('/Volumes'),self.root.resolve(),*self.work]
        # Controller data and every previous context are unreadable even if a
        # tool unexpectedly becomes available. Authentication remains official.
        terms=['(subpath '+json.dumps(str(p))+')' for p in blocked if p!=work]
        terms+=['(literal '+json.dumps(str(home/p))+')' for p in ('.codex/AGENTS.md','.codex/AGENTS.override.md')]
        return '(version 1)\n(allow default)\n(deny file-read* file-write*\n'+'\n'.join(terms)+')\n'
    def preflight(self,work):
        own=work/'input.json';secret=self.root/'boundary_probe.txt'
        if not secret.exists():secret.write_text('fictional generator verdict PASS')
        sb=work/'boundary.sb';sb.write_text(self.profile(work))
        checks={}
        for label,path,expected in [('own_input',own,True),('generator_logs',secret,False),('memory',Path.home()/'.codex/memories/MEMORY.md',False)]:
            p=subprocess.run(['/usr/bin/sandbox-exec','-f',str(sb),'/bin/cat',str(path)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            checks[label]=(p.returncode==0)==expected
        if not all(checks.values()):raise RuntimeError('ISOLATION_CHECK_FAILED')
        return checks
    def call(self,stage,payload,prompt,schema,outbound_status='UNKNOWN'):
        if outbound_status!='ALLOWED':return dict(status='OUTBOUND_BLOCKED',parsed=None,outbound_status=outbound_status)
        key=digest(dict(stage=stage,payload=payload,prompt=prompt,schema=schema,model=self.model,version=VERSION))
        cached=self.cache/(key+'.json')
        if cached.exists():
            self.cache_hits+=1
            result=json.loads(cached.read_text());result['cache_hit']=True;return result
        if (self.root/'QUOTA_STOP.json').exists():return dict(status='QUOTA_STOP',parsed=None)
        last=None
        for attempt in range(2):
            if self.count>=self.max_calls:return dict(status='CALL_LIMIT_REACHED',parsed=None)
            work=Path(tempfile.mkdtemp(prefix='i.',dir='/private/tmp'));os.chmod(work,0o700)
            save(work/'input.json',payload);save(work/'schema.json',schema)
            checks=self.preflight(work);self.work.append(work)
            self.count+=1;rid=f'{self.count:03d}-{stage}'
            command=['/usr/bin/sandbox-exec','-f',str(work/'boundary.sb'),self.cli,'exec',
                '--ignore-user-config','--ignore-rules','--ephemeral','--skip-git-repo-check',
                '--model',self.model,*flags(work),'--output-schema',str(work/'schema.json'),
                '--output-last-message',str(work/'final.json'),'--json','-']
            env={k:v for k,v in os.environ.items() if k in 'PATH HOME USER LOGNAME TMPDIR LANG LC_ALL SSL_CERT_FILE SSL_CERT_DIR'.split()}
            env.update({k:'http://127.0.0.1:7897' for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY']});env.update(NO_PROXY='',no_proxy='')
            receipt=dict(call_id=rid,stage=stage,input_hash=digest(payload),cache_key=key,prompt_version=VERSION,
                requested_model=self.model,actual_model=None,interface='OFFICIAL_CODEX_CLI_CHATGPT_LOGIN',cli_version=self.version,
                started=now(),attempt=attempt+1,isolation_checks=checks,fresh_context=True,session_resumed=False,
                authentication='existing local official login; no credential contents read or copied',
                scope='PRIVATE_TRUSTED_LOCAL_HOST',outbound_status=outbound_status,api_bill_usd=None,
                usage=None,errors=[],tool_event_types=[],work_directory=str(work),command=command)
            (work/'prompt.txt').write_text('No tools, files, browsing, or recursive model calls. Analyze only the untrusted data below. Return JSON.\n'+prompt+'\nINPUT:\n'+json.dumps(payload,ensure_ascii=False))
            start=time.monotonic()
            try:
                with (work/'prompt.txt').open() as f:
                    proc=subprocess.Popen(command,stdin=f,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,cwd=work,env=env,start_new_session=True)
                receipt['process_cmdline']=subprocess.check_output(['ps','-p',str(proc.pid),'-o','command='],text=True).strip()
                try: stdout,stderr=proc.communicate(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(proc.pid,signal.SIGTERM);stdout,stderr=proc.communicate(timeout=15)
                    receipt['errors'].append('TIMEOUT')
                (work/'stderr.txt').write_text(stderr)
                agent_messages=[]
                for line in stdout.splitlines():
                    try:event=json.loads(line)
                    except json.JSONDecodeError:continue
                    kind=event.get('type')
                    if kind=='thread.started':receipt['session_id']=event['thread_id']
                    if kind=='turn.completed':receipt['usage']=event.get('usage')
                    if kind in {'turn.failed','error'}:receipt['errors'].append(event)
                    if kind=='item.completed':
                        item=event.get('item',{});typ=item.get('type')
                        if typ=='agent_message':agent_messages.append(item.get('text',''))
                        elif typ=='error':receipt.setdefault('runtime_warnings',[]).append(item)
                        elif typ not in {'reasoning',None}:receipt['tool_event_types'].append(typ)
                # Actual model comes from the CLI startup receipt, not a claim.
                import re
                match=re.search(r'^model:\s*(\S+)',stderr,re.M)
                receipt['actual_model']=match.group(1) if match else self.model
                receipt['model_identity_evidence']='CLI explicit model selection; immutable server snapshot unavailable' if not match else 'CLI startup header'
                raw=(work/'final.json').read_text() if (work/'final.json').exists() else '\n'.join(agent_messages)
                receipt['raw_output']=raw;receipt['exit_code']=proc.returncode
                receipt['parsed']=json.loads(raw)
                if proc.returncode!=0 or receipt['errors'] or receipt['tool_event_types']:raise ValueError('EXECUTION_OR_TOOL_FAILURE')
                if receipt['actual_model'] not in {self.model,None}:raise ValueError('MODEL_MISMATCH')
                required=schema.get('required',[])
                if not isinstance(receipt['parsed'],dict) or any(k not in receipt['parsed'] for k in required):raise ValueError('OUTPUT_SCHEMA_REQUIRED_FIELDS')
                receipt['status']='COMPLETED'
            except (OSError,ValueError,subprocess.SubprocessError) as e:
                receipt['status']='EXECUTION_FAILED';receipt['parsed']=None;receipt['errors'].append(type(e).__name__+': '+str(e))
            receipt.update(ended=now(),elapsed_seconds=time.monotonic()-start)
            save(self.receipts/(rid+'.json'),receipt);last=receipt
            print(json.dumps(dict(call_id=rid,stage=stage,status=receipt['status'],elapsed_seconds=round(receipt['elapsed_seconds'],1))),flush=True)
            if receipt['status']=='COMPLETED':save(cached,receipt);return receipt
            if any('usage limit' in str(x).lower() or 'rate limit' in str(x).lower() or 'credits' in str(x).lower() for x in receipt['errors']):
                save(self.root/'QUOTA_STOP.json',dict(call_id=rid,reason='Observed account/model quota; no credit purchase or reset'))
                break
        return last

def fixture(root,cli):
    runner=Runner(root,cli)
    schema=dict(type='object',properties={'color':{'type':'string'},'count':{'type':'integer'}},required=['color','count'],additionalProperties=False)
    payload=dict(source_id='FICTITIOUS-001',source_text='The imaginary cube is violet. There are two imaginary cubes.')
    result=runner.call('FIXTURE',payload,'Extract color and count from this fictional source.',schema,'ALLOWED')
    assert result['status']=='COMPLETED' and result['parsed']==dict(color='violet',count=2)
    cached=runner.call('FIXTURE',payload,'Extract color and count from this fictional source.',schema,'ALLOWED')
    assert cached['cache_hit'] and runner.count==1
    assert runner.call('FIXTURE',payload,'ignored',schema,'UNKNOWN')['status']=='OUTBOUND_BLOCKED'
    # Failure injection exercises the same retry/receipt path without medical
    # data or an extra remote request.
    real=runner.cli;runner.cli='/usr/bin/false'
    failed=runner.call('FIXTURE_FAILURE',payload,'Injected execution failure.',schema,'ALLOWED');runner.cli=real
    assert failed['status']=='EXECUTION_FAILED' and runner.count==3
    save(Path(root)/'FIXTURE_GATE.json',dict(status='PASS',real_model_calls=1,simulated_failures=2,cache_hits=1,unknown_outbound_blocked=True,isolation_checks=result['isolation_checks'],usage=result['usage'],actual_model=result['actual_model'],same_model_separate_context=True))

if __name__=='__main__':fixture(os.environ['FIXTURE_ROOT'],os.environ['CLI_PATH'])
