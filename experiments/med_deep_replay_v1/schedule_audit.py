"""Count the fixed 160-step order using identity metadata, not FIT content."""
from pathlib import Path

path = Path(__file__).with_name('identity_audit.py')
source = path.read_text()
start = source.index("print(json.dumps({'records':len(rows)")
end = source.index("\n'''", start)
replacement = """allowed=[x for x in rows if x not in reserve and x not in panel and x not in checks and x not in historic]
first=allowed[:160]
print(json.dumps({'allowed_records':len(allowed),'allowed_groups':len(set(allowed)),
                  'scheduled_records':len(first),'scheduled_groups':len(set(first)),
                  'unreached_records':len(allowed)-len(first),'unreached_groups':len(set(allowed)-set(first)),
                  'qa_decoded':0,'content_admitted':False}))"""
source = source[:start] + replacement + source[end:]
exec(compile(source, str(path), 'exec'), {'__file__': str(path), '__name__': '__main__'})
