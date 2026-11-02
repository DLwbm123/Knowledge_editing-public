"""One guarded source-only token coverage pass; no images or evaluation data."""
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time

ROOT = Path(os.environ['TASK_INPUT_ROOT'])
OUT = Path(os.environ['TASK_PRIVATE_OUT'])
MODEL = Path(os.environ['TASK_TOKENIZER'])
OFFICIAL = Path(os.environ['TASK_OFFICIAL_SOURCE'])


class MappingUnknown(Exception):
    pass


def main():
    sys.path.insert(0, str(OFFICIAL))
    from llava.conversation import conv_templates
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    if type(tokenizer).__name__ != 'LlamaTokenizerFast' or tokenizer.eos_token_id != 2:
        raise ValueError('Tokenizer class or EOS mismatch')
    adm = json.loads((ROOT / 'ROLE_ADMISSION.json').read_text())
    entries = json.loads((ROOT / 'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
    if adm['status'] != 'TRAINING_SOURCE_ROLE_ADMITTED' or not 1 <= len(entries) <= 150:
        raise ValueError('Role admission mismatch')
    if [str(x['task']['order']) for x in entries] != [str(x) for x in adm['owners']]:
        raise ValueError('Owner identity mismatch')
    terms = {
        'anatomy': 'brain lung lungs heart liver kidney kidneys spleen pancreas stomach intestine bowel bladder breast chest abdomen abdominal pelvis pelvic spine spinal head neck'.split(),
        'observation': 'lesion mass nodule tumor fracture opacity effusion hemorrhage bleeding infection'.split(),
    }
    pats = {k: re.compile(r'(?<!\w)(?:' + '|'.join(map(re.escape, sorted(v, key=len, reverse=True))) + r')(?!\w)', re.I) for k, v in terms.items()}
    def prompt(question, answer):
        conv = conv_templates['mistral_instruct'].copy()
        conv.append_message(conv.roles[0], '<image>\n' + question)
        conv.append_message(conv.roles[1], answer)
        return conv.get_prompt()
    def encode(p):
        chunks = p.split('<image>')
        if len(chunks) != 2:
            raise ValueError('Unexpected image placeholder count')
        left = tokenizer(chunks[0], return_offsets_mapping=True)
        right = tokenizer(chunks[1], return_offsets_mapping=True)
        if not left.input_ids or left.input_ids[0] != tokenizer.bos_token_id or right.input_ids[0] != tokenizer.bos_token_id:
            raise ValueError('Unexpected image-token encoding')
        start = len(chunks[0]) + len('<image>')
        ids = left.input_ids + [-200] + right.input_ids[1:]
        offsets = list(left.offset_mapping) + [(0, 0)] + [(a+start,b+start) if b>a else (0,0) for a,b in right.offset_mapping[1:]]
        return ids, offsets
    counts = collections.Counter()
    locks = []
    for entry in entries:
        rows = entry['source_rows']
        if len(rows) != 5 or [r['audit_role'] for r in rows] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']:
            raise ValueError('Training source roles mismatch')
        answers = [r['reference'].strip() for r in rows]
        if len(set(answers)) != 1:
            raise ValueError('Training references differ')
        answer = answers[0]
        matched = [pats[k].search(answer) for k in ('anatomy','observation')]
        record = {'order':entry['task']['order'], 'status':'FALLBACK'}
        if not all(matched):
            counts['missing_category'] += 1
            locks.append(record)
            continue
        counts['two_category'] += 1
        common = None
        per_view = []
        try:
            for row in rows:
                prefix = prompt(row['question'], None)
                full = prompt(row['question'], answer)
                if not full.startswith(prefix):
                    raise ValueError('Prompt prefix mismatch')
                answer_start = full.find(answer, len(prefix))
                if answer_start < 0 or full[len(prefix):answer_start].strip():
                    raise ValueError('Prompt answer position does not bind')
                pref_ids, _ = encode(prefix)
                ids, offsets = encode(full)
                if ids[:len(pref_ids)] != pref_ids:
                    raise ValueError('Full prompt does not preserve prefix tokens')
                answer_end = answer_start + len(answer)
                positions = [i for i,(a,b) in enumerate(offsets) if i>=len(pref_ids) and b>a and a<answer_end and answer_start<b]
                if any(full[a:min(b,answer_start)].strip() or full[max(a,answer_end):b].strip() for i in positions for a,b in [offsets[i]]):
                    raise MappingUnknown('Answer boundary mixes nonspace content')
                if not positions or positions != list(range(positions[0],positions[-1]+1)):
                    raise ValueError('Answer token positions noncontiguous or empty')
                target_ids = [ids[i] for i in positions]
                if common is None: common = target_ids
                if common != target_ids: raise ValueError('Answer tokenization differs across views')
                spans = []
                for m in matched:
                    start, end = answer_start+m.start(), answer_start+m.end()
                    sel = [j for j,i in enumerate(positions) if offsets[i][0] < end and start < offsets[i][1]]
                    if not sel or sel != list(range(sel[0], sel[-1]+1)):
                        raise MappingUnknown('Span mapping is not contiguous')
                    if offsets[positions[sel[0]]][0] > start or offsets[positions[sel[-1]]][1] < end:
                        raise MappingUnknown('Span mapping incomplete')
                    spans.append((sel[0],sel[-1]+1))
                if set(range(*spans[0])) & set(range(*spans[1])):
                    raise MappingUnknown('Mapped source spans cross')
                per_view.append({'answer_start_label':positions[0], 'answer_end_label':positions[-1]+1, 'medical':spans})
            if any(x['medical'] != per_view[0]['medical'] for x in per_view):
                raise MappingUnknown('Target span differs between views')
        except MappingUnknown:
            counts['span_mapping_unknown'] += 1
            locks.append(record)
            continue
        controls=[]
        for k,(start,end) in enumerate(per_view[0]['medical']):
            n=end-start
            windows=[(i,i+n) for i in range(len(common)-n+1)]
            seed=str(entry['task']['seed'])
            windows.sort(key=lambda w:hashlib.sha256(f'{seed}:{k}:{w[0]}'.encode()).digest())
            pick=next((w for w in windows if all(not (w[0]<z[1] and z[0]<w[1]) for z in controls)),None)
            if pick is None: break
            controls.append(pick)
        if len(controls)!=2:
            counts['control_missing'] += 1
            locks.append(record)
            continue
        medical=per_view[0]['medical']
        counts['matched_edits']+=1
        counts['groups']+=2
        counts['equal_groups']+=sum(a==b for a,b in zip(medical,controls))
        counts['overlapping_groups']+=sum(bool(set(range(*a)) & set(range(*b))) for a,b in zip(medical,controls))
        counts['medical_tokens']+=sum(b-a for a,b in medical)
        counts['control_tokens']+=sum(b-a for a,b in controls)
        record.update(status='WEAK_TOKEN_MATCH',views=per_view,controls=controls,answer_tokens=len(common))
        locks.append(record)
    if counts['medical_tokens'] != counts['control_tokens'] or len(locks)!=len(entries):
        raise ValueError('Coverage accounting mismatch')
    fd=os.open(OUT,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f: json.dump({'status':'WEAK_TOKEN_PROXY_ONLY','records':locks},f,separators=(',',':'))
    print(json.dumps({'edits':len(entries),'counts':dict(counts),'training_rows':len(entries)*5,'eval_question_decodes':0,'image_pixels':0,'model_calls':0,'judge_calls':0}),flush=True)

if __name__=='__main__':
    # The independently waiting parent controls a child and enforces a hard deadline.
    started=time.time(); pid=os.fork()
    if pid==0:
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE,(10_000_000,10_000_000))
            main()
        except BaseException as e:
            print(json.dumps({'status':'FAIL','error_type':type(e).__name__,'reason':str(e) if str(e) in ('Prompt prefix mismatch','Prompt answer position does not bind','Full prompt does not preserve prefix tokens','Answer token positions noncontiguous or empty','Answer tokenization differs across views','Span mapping is not contiguous','Span mapping incomplete','Mapped source spans cross','Target span differs between views','Training source roles mismatch','Training references differ','Unexpected image-token encoding') else 'OTHER'}),flush=True)
            os._exit(1)
        os._exit(0)
    deadline=float(os.environ['TASK_DEADLINE'])
    while True:
        got,status=os.waitpid(pid,os.WNOHANG)
        if got:
            code=os.waitstatus_to_exitcode(status)
            print(json.dumps({'guard_exit':code,'elapsed_seconds':round(time.time()-started,3)}),flush=True)
            sys.exit(code)
        if time.time()>=deadline:
            os.kill(pid,signal.SIGKILL); os.waitpid(pid,0)
            print(json.dumps({'guard_exit':'TIMEOUT','elapsed_seconds':round(time.time()-started,3)}),flush=True)
            sys.exit(124)
        time.sleep(.2)
