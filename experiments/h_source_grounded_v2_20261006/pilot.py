"""A cached blind extraction, B construction, C clean-room review; no training."""
import collections
import json
import os
from pathlib import Path
import re
from audit import read,write,save
from runner import Runner,digest

def obj(fields):return dict(type='object',properties=fields,required=list(fields),additionalProperties=False)
S={'type':'string'};B={'type':'boolean'};LS={'type':'array','items':S}
Q=obj({x:S for x in ['body_part','laterality','time','count','uncertainty','protocol','referent']})
FACT=obj(dict(fact_id=S,entity=S,relation=S,value=S,polarity=S,qualifiers=Q,evidence_ids=LS,evidence_spans=LS,evidence_origin=S,scope=S,limitations=LS))
SCHEMAS={
 'A':obj(dict(source_id=S,facts={'type':'array','items':FACT},insufficient_evidence=B)),
 'B':obj(dict(edit_id=S,source_id=S,candidate_verdict=S,construction_type=S,h_question=S,h_answer=S,same_question_operator=B,qualifier_alignment=obj(dict(native_full_question_preserved=B,review=S)),source_supports_h_answer=S,target_relation=S,edit_should_transfer=S,evidence_ids=LS,evidence_spans=LS,decisive_reason=S,requires_new_medical_inference=B,missing_evidence=LS,needs_specialist_review=B,suggested_evidence_level=S)),
 'C':obj(dict(edit_id=S,source_id=S,answer_supported=S,question_aligned=S,target_excluded=S,independent_verdict=S,failure_codes=LS,supporting_evidence_ids=LS,concise_reason=S,requires_specialist_review=B))}

def source_view(s):
    return {k:s[k] for k in ['source_id','source_dataset','source_text','evidence_ids','evidence_origin','image_binding']}
def edit_view(e):
    return {k:e[k] for k in ['edit_id','original_question','target_answer','attribute_relation','qualifiers_full_text']}
def audit_payload(edit,source,candidate):
    # A whitelist, not removal of a few known generator fields. No verdict,
    # confidence, reason, PASS label, quota, or A model interpretation is sent.
    return dict(edit_record=edit_view(edit),candidate_H={k:candidate[k] for k in ['h_question','h_answer','evidence_ids','evidence_spans']},source_record=source_view(source))
def valid_spans(spans,text):return bool(spans) and all(isinstance(x,str) and x and x in text for x in spans)
def score(e,s,old_ids):
    q=e['original_question'].lower();t=e['target_answer'].lower();text=s['source_text'].lower()
    tokens=set(re.findall(r'[a-z]+',q+' '+t))-set('what where is are the this that in of and seen image describe how would you there it on side body pathology lesion findings abnormal'.split())
    overlap=sum(w in text for w in tokens)
    keys=[k for k in ['ring','pleural','aortic','tributaries','hemorrhage','edema','omental','tip','posterior'] if k in q+' '+t]
    direct=sum(k in text for k in keys)
    negative=bool(re.search(r'\b(?:no|without|absence)\b',text))
    old=any(x['source_id'] in old_ids for x in s.get('original_QA_records',[]))
    return (direct*4+overlap+int(negative)*2,int(old),s['source_id'])

def queue(root,units,origin,per_edit=2):
    p=root/'private';selection=json.loads((p/'PILOT_SELECTION.json').read_text())
    v1=read(p/'v1_revised_candidates.jsonl');rows=[]
    previous=read(p/'pilot_results.jsonl') if (p/'pilot_results.jsonl').exists() else []
    used=collections.defaultdict(set)
    for r in previous:used[r['edit_id']].add(r['source_group_id'])
    for e in selection['edits']:
        old_ids={x['source_id'] for x in v1 if x['edit_id']==e['edit_id'] and x['construction_status']=='ENGINE_NOT_IMPLEMENTED'}
        ranked=sorted([s for s in units if s['assigned_role']=='FIT' and s['source_role_check']=='PASS' and s['outbound_status']=='ALLOWED' and s['source_group_id'] not in used[e['edit_id']] and (origin=='V1_CACHE_REUSE' or e['position'] in s['targeted_edit_positions'])],key=lambda s:score(e,s,old_ids),reverse=True)
        for s in ranked[:min(per_edit,10-len(used[e['edit_id']]))]:
            # Text overlap is retrieval only. Preserve weak semantic matches for
            # explicit UNKNOWN/FAIL; do not pretend they are accepted evidence.
            if score(e,s,old_ids)[0]<1:continue
            rows.append(dict(edit=e,source=s,origin=origin,old_candidate_reused=any(x['source_id'] in old_ids for x in s.get('original_QA_records',[]))))
            used[e['edit_id']].add(s['source_group_id'])
    save(p/('QUEUE_'+origin+'.json'),dict(items=rows,relationships=len(rows),selection_hash=digest(selection),locked_before_this_phase_model_calls=True))
    return rows

def main():
    root=Path(os.environ['V2_ROOT']);p=root/'private';origin=os.environ.get('PILOT_ORIGIN','V1_CACHE_REUSE')
    units=read(p/('v1_reused_units.jsonl' if origin=='V1_CACHE_REUSE' else 'new_units.jsonl'))
    queue_file=p/('QUEUE_'+origin+'.json')
    items=json.loads(queue_file.read_text())['items'] if queue_file.exists() else queue(root,units,origin)
    runner=Runner(p/'model_calls',os.environ['CLI_PATH'],max_calls=295)
    prompt_root=Path(os.environ['PROMPT_ROOT'])
    prompts={stage:(prompt_root/f'prompt_{i}.txt').read_text()+ '\nV2: Return all schema fields. Use empty strings for an unavailable candidate. Preserve the literal original question unless an exact proposition-preserving alignment is needed. No quota. Do not diagnose pixels; source text only. Key anatomical or clinical inference unavailable in the source must remain UNKNOWN.' for i,stage in enumerate('ABC',1)}
    # A audit record never contains an edit target. Cache per source unit across
    # every relation and both phases. Input is only whitelisted original data.
    prior=read(p/'pilot_results.jsonl') if (p/'pilot_results.jsonl').exists() else []
    done={(r['edit_id'],r['source_id']) for r in prior}
    for item in items:
        e=item['edit'];s=item['source'];eid=e['edit_id'];sid=s['source_id']
        if (eid,sid) in done:continue
        if len(prior)>=120:break
        a=runner.call('A',dict(source_record=source_view(s)),prompts['A'],SCHEMAS['A'],s['outbound_status'])
        row=dict(edit_id=eid,position=e['position'],pilot_question_type=e['pilot_question_type'],source_id=sid,source_group_id=s['source_group_id'],image_id=s['image_id'],image_path=s['image_path'],image_sha256=s['image_sha256'],assigned_role='FIT',origin=origin,old_candidate_reused=item['old_candidate_reused'],outbound_status=s['outbound_status'],A_receipt=a.get('call_id'),construction_status='EXECUTION_FAILED',evidence_status='NOT_YET_ASSESSED',evidence_level='UNKNOWN',final_status='UNKNOWN',model_relationship='SAME_MODEL_SEPARATE_CONTEXT',patient_independence='UNKNOWN',training_started=False)
        facts=a.get('parsed')
        if facts and facts.get('source_id')==sid and all(valid_spans(f['evidence_spans'],s['source_text']) and set(f['evidence_ids'])<=set(s['evidence_ids']) for f in facts['facts']):
            for f in facts['facts']:f['precise_locations']=[dict(start=s['source_text'].index(x),end=s['source_text'].index(x)+len(x)) for x in f['evidence_spans']]
            b=runner.call('B',dict(edit_record=edit_view(e),source_record=source_view(s),extracted_facts=facts,role_check='PASS_FIT_GROUP_FIXED'),prompts['B'],SCHEMAS['B'],s['outbound_status'])
            candidate=b.get('parsed');row['B_receipt']=b.get('call_id');row['generator_output']=candidate
            if candidate and candidate.get('edit_id')==eid and candidate.get('source_id')==sid:
                c=runner.call('C',audit_payload(e,s,candidate),prompts['C'],SCHEMAS['C'],s['outbound_status'])
                audit=c.get('parsed');row['C_receipt']=c.get('call_id');row['audit_output']=audit
                if audit and audit.get('edit_id')==eid and audit.get('source_id')==sid:
                    row['construction_status']='MODEL_REVIEW_COMPLETED'
                    spans_ok=valid_spans(candidate['evidence_spans'],s['source_text']) and set(candidate['evidence_ids'])<=set(s['evidence_ids'])
                    alignment=candidate['same_question_operator'] and audit['question_aligned']=='YES'
                    supported=candidate['source_supports_h_answer']=='YES' and audit['answer_supported']=='YES' and spans_ok
                    excluded=candidate['target_relation']=='CONTRADICTS' and audit['target_excluded']=='YES'
                    passed=candidate['candidate_verdict']=='SUPPORTED' and audit['independent_verdict']=='PASS' and alignment and supported and excluded and not candidate['requires_new_medical_inference'] and not candidate['needs_specialist_review'] and not audit['requires_specialist_review']
                    row.update(h_question=candidate['h_question'],h_answer=candidate['h_answer'],evidence_ids=candidate['evidence_ids'],evidence_spans=candidate['evidence_spans'],citation_check='PASS' if spans_ok else 'FAIL',source_supports_h_answer='YES' if supported else 'UNKNOWN',target_relation='CONTRADICTS' if excluded else 'COMPATIBLE' if audit['target_excluded']=='NO' else 'UNKNOWN',same_question_operator=bool(alignment),source_role_check='PASS',A_facts=facts,missing_evidence=candidate['missing_evidence'],specific_review_gap=candidate['decisive_reason']+' '+audit['concise_reason'])
                    row['evidence_status']='ANSWER_SUPPORTED_TARGET_EXCLUDED' if supported and excluded and alignment else 'ANSWER_SUPPORTED_TARGET_NOT_EXCLUDED' if supported and audit['target_excluded']=='NO' else 'ANSWER_EVIDENCE_MISSING' if audit['answer_supported']=='NO' or not spans_ok else 'REFERENT_OR_QUALIFIER_UNRESOLVED'
                    # Model consensus is still grounded; never auto-upgrade.
                    if passed:row.update(evidence_level='SOURCE_GROUNDED',final_status='ACCEPTED_GROUNDED_REVIEWED')
                    elif audit['independent_verdict']=='FAIL' and candidate['candidate_verdict']=='REFUTED':row.update(evidence_level='REJECTED',final_status='REJECTED')
                    row['generator_auditor_disagreement']={'SUPPORTED':'PASS','REFUTED':'FAIL','UNKNOWN':'UNKNOWN'}.get(candidate['candidate_verdict'])!=audit['independent_verdict']
                else:row['execution_error']='C_FAILED_OR_ID_MISMATCH'
            else:row['execution_error']='B_FAILED_OR_ID_MISMATCH'
        else:row['execution_error']='A_FAILED_OR_UNBOUND_EVIDENCE'
        with (p/'pilot_results.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        prior.append(row);done.add((eid,sid))
        print(json.dumps(dict(position=e['position'],status=row['final_status'],calls=runner.count)),flush=True)
        if a['status']=='CALL_LIMIT_REACHED':break
    print(json.dumps(dict(phase=origin,relations=len(prior),calls=runner.count,cache_hits=runner.cache_hits)),flush=True)

if __name__=='__main__':main()
