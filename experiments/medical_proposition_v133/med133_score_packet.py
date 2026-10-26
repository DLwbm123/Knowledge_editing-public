"""Read-only common-epoch answer packet; no historical regeneration."""
import hashlib
import json
import os
from pathlib import Path
import time


def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def gather(stage):
    root,parent,reference=map(lambda k:Path(os.environ[k]),('EXPERIMENT_ROOT','PARENT_RUN','REFERENCE_RUN'))
    f=json.loads((root/'INPUTS_FROZEN.json').read_text())
    assert (root/f'STAGE_{stage}_GENERATION_COMPLETE.json').exists()
    plan=f['plan'] if stage=='B' else [v for v in f['plan'] if v['mode']=='single']
    consumers=[];fields=('question','reference','image_sha256','prompt_ids','attention_mask','runtime','generation')
    bases={}
    for row in plan:
        cid=json.dumps([row[k] for k in ('method','mode','prefix','owner')]+[row['query']['query_id']],separators=(',',':'))
        med=json.loads((root/'outputs'/(digest(cid)+'.json')).read_text())
        legacyid=json.dumps(['M13P_160']+[row[k] for k in ('mode','prefix','owner')]+[row['query']['query_id']],separators=(',',':'))
        legacy=json.loads((reference/'M13P_160/outputs'/(digest(legacyid)+'.json')).read_text())
        qid=row['query']['query_id']
        if qid not in bases:
            bases[qid]=json.loads((parent/'private/outputs/BASE_base_p0/0'/(digest(qid)+'.json')).read_text())
        for arm,value in [('MED',med),('LEGACY',legacy),('BASE',bases[qid])]:
            binding=value['binding']['judge_input'];other=med['binding']['judge_input']
            assert all(binding[k]==other[k] for k in fields),('Common input/runtime mismatch',arm,cid)
            assert binding['question']==row['query']['question'] and binding['reference']==row['query']['reference']
            answer=value['R0'];assert answer['raw_token_ids']
            payload=dict(query_id=qid,question=binding['question'],reference=binding['reference'],
                image_sha256=binding['image_sha256'],prompt_ids=binding['prompt_ids'],answer=answer)
            consumers.append(dict(arm=arm,mode=row['mode'],prefix=row['prefix'],owner=row['owner'],query_id=qid,
                payload_key=digest(payload),question=binding['question'],reference=binding['reference'],
                image_sha256=binding['image_sha256'],answer=answer,EOS=answer['raw_token_ids'][-1]==2,
                at_cap=len(answer['raw_token_ids'])>=1024,active_target=med['active_target'],
                effective_expert=med['effective_expert'] if arm=='MED' else value.get('effective_expert'),
                prediction_reused=med['prediction_reused'] if arm=='MED' else True))
    assert len(consumers)==len(plan)*3
    return dict(tasks=f['tasks'],base_mask=f['base_mask'],consumers=consumers,
        clock=json.loads((root/'CLOCK.json').read_text()),stage=stage,
        input_binding=json.loads((root/'PROTOCOL_LOCK.json').read_text())['input_binding'])

if __name__=='__main__':print(json.dumps(gather(os.environ['SCORE_STAGE'])))
