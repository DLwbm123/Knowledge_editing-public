"""After all 2148 predictions freeze, form candidate and original score payloads."""
import hashlib
import json
from pathlib import Path
import prior_binding
import stream_extract as stream

CONTRACT='MEDPROP_SMALL4_CORE_FULL_SOURCE_V1'


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def build(root, old150):
    root,old150=Path(root),Path(old150)
    frozen=json.loads((root/'PREDICTIONS_FROZEN.json').read_text())
    if frozen.get('status')!='ALL_PREDICTIONS_FROZEN' or frozen.get('events')!=2148:
        raise ValueError('predictions are not frozen')
    if json.loads((root/'EVAL_GUARD_EXIT.json').read_text()).get('code')!=0:
        raise ValueError('evaluation guard did not succeed')
    inputs=json.loads((root/'BANK_INPUTS_PRIVATE.json').read_text())
    if inputs.get('status')!='BANK_INPUTS_BOUND' or len(inputs['records'])!=2148:
        raise ValueError('bank input coverage mismatch')
    predictions={}
    for file in (root/'predictions').glob('*.json'):
        v=json.loads(file.read_text());cid=v['consumer_id']
        if cid in predictions:raise ValueError('duplicate prediction')
        predictions[cid]=v
    if len(predictions)!=2148:raise ValueError('prediction coverage mismatch')
    records={};consumers=[]
    fields={'consumer_id':True,'binding':{'input':{'query_id':True,'question':True,'image_sha256':True,'reference':True}},
            'R0':{'raw_answer':True,'raw_token_ids':True},'EOS':True,'at_cap':True}
    for item in inputs['records']:
        cid=item['consumer_id'];pred=predictions.get(cid)
        if pred is None or pred['query_id']!=item['query']['query_id']:
            raise ValueError('missing or misbound candidate')
        with (old150/'eval_outputs'/item['original_filename']).open('rb') as f:
            old=prior_binding.select(f,stream.nonspace(f),fields)
        q=item['query'];oq=old.get('binding',{}).get('input',{})
        if old.get('consumer_id')!=cid or any(oq.get(k)!=q[k] for k in ('query_id','question','image_sha256')) or not isinstance(oq.get('reference'),str):
            raise ValueError('reference binding differs')
        old_answer=old.get('R0',{})
        if not isinstance(old_answer.get('raw_answer'),str) or not old_answer.get('raw_token_ids'):
            raise ValueError('invalid original answer')
        if pred.get('candidate'):
            if pred.get('original_route')!=item['route'] or any(pred.get(k)!=item['judge_input'][k] for k in ('question','image_sha256','prompt_ids','attention_mask','generation')):
                raise ValueError('candidate input or R0 mismatch')
            candidate={'raw_answer':pred['raw_answer'],'raw_token_ids':pred['raw_token_ids']}
        else:
            if pred.get('prediction_reused_from')!=item['original_filename'] or pred.get('route')!=item['route']:
                raise ValueError('reused prediction binding differs')
            candidate=old_answer
        for arm,answer in (('MED',old_answer),('MED_D',candidate)):
            semantic={'protocol':CONTRACT,'model':'gpt-6-astra','reasoning':'medium','question':q['question'],
                      'gold_answer':oq['reference'],'raw_base_answer':answer['raw_answer']}
            key=digest(semantic)
            legacy={'query_id':q['query_id'],'question':q['question'],'reference':oq['reference'],
                    'image_sha256':q['image_sha256'],'prompt_ids':item['judge_input']['prompt_ids'],'answer':answer}
            record={'opaque_query_id':key,'question':q['question'],'gold_answer':oq['reference'],'raw_base_answer':answer['raw_answer']}
            if key not in records:records[key]={'record':record,'legacy_keys':[]}
            if records[key]['record']!=record:raise ValueError('score payload collision')
            records[key]['legacy_keys'].append(digest(legacy))
            consumers.append({'consumer_id':cid,'query_id':q['query_id'],'arm':arm,'key':key,
                              'owner':item['owner'],'role':q['audit_role'],'route':item['route'],
                              'candidate_expert':bool(pred.get('candidate'))})
    if len(consumers)!=4296 or {x['consumer_id'] for x in consumers}!=set(predictions):
        raise ValueError('consumer coverage mismatch')
    packet={'status':'SCORE_PACKET_FROZEN','consumers':consumers,'payloads':records,'events':2148,'new_judge_authorized':False}
    path=root/'SCORE_PACKET_PRIVATE.json'
    with path.open('x') as f:json.dump(packet,f,ensure_ascii=False)
    path.chmod(0o600)
    return {'status':packet['status'],'consumers':len(consumers),'unique_payloads':len(records)}
