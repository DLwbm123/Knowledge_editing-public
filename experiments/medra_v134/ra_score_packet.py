"""Common frozen four-arm answers, with no comparator regeneration."""
import json
import os
from pathlib import Path

from med133_score_contract import digest


def gather(stage):
    root,parent,reference,raw=map(lambda k:Path(os.environ[k]),('EXPERIMENT_ROOT','PARENT_RUN','REFERENCE_RUN','RAW_MED_ROOT'))
    read=lambda p:json.loads(p.read_text())
    inputs=read(root/'INPUTS_FROZEN.json')
    assert (root/f'STAGE_{stage}_GENERATION_COMPLETE.json').exists()
    plan=inputs['plan'] if stage=='B' else [v for v in inputs['plan'] if v['mode']=='single']
    hashes=read(root/'EXPERT_LIBRARY_FROZEN.json')['states']
    consumers=[];bases={}
    for row in plan:
        qid=row['query']['query_id']
        identity=lambda name:json.dumps([name]+[row[k] for k in ('mode','prefix','owner')]+[qid],separators=(',',':'))
        ra=read(root/'outputs'/(digest(identity('MED_RA'))+'.json'))
        med=read(raw/'outputs'/(digest(identity('MED'))+'.json'))
        legacy=read(reference/'M13P_160/outputs'/(digest(identity('M13P_160'))+'.json'))
        if qid not in bases:bases[qid]=read(parent/'private/outputs/BASE_base_p0/0'/(digest(qid)+'.json'))
        assert ra['effective_state_hash']==(hashes[ra['effective_expert']] if ra['effective_expert'] else 'UNEDITED_BASE')
        assert ra['original_router'] and ra['actual_Base_R0_evaluated']
        fields=('question','reference','image_sha256','prompt_ids','attention_mask','runtime','generation')
        for arm,value in [('RA',ra),('MED',med),('LEGACY',legacy),('BASE',bases[qid])]:
            binding=value['binding']['judge_input'];expected=ra['binding']['judge_input']
            assert all(binding[k]==expected[k] for k in fields),('Input/runtime mismatch',arm)
            assert binding['question']==row['query']['question'] and binding['reference']==row['query']['reference']
            answer=value['R0'];assert answer['raw_token_ids']
            key=digest(dict(query_id=qid,question=binding['question'],reference=binding['reference'],
                image_sha256=binding['image_sha256'],prompt_ids=binding['prompt_ids'],answer=answer))
            consumers.append(dict(arm=arm,mode=row['mode'],prefix=row['prefix'],owner=row['owner'],query_id=qid,
                payload_key=key,question=binding['question'],reference=binding['reference'],image_sha256=binding['image_sha256'],
                answer=answer,EOS=answer['raw_token_ids'][-1]==2,at_cap=len(answer['raw_token_ids'])>=1024,
                active_target=ra['active_target'],effective_expert=value.get('effective_expert'),
                prediction_reused=value.get('prediction_reused',False)))
    assert len(consumers)==len(plan)*4
    return dict(tasks=inputs['tasks'],base_mask=inputs['base_mask'],consumers=consumers,stage=stage,
                clock=read(root/'CLOCK.json'),input_binding=read(root/'PROTOCOL_LOCK.json')['input_binding'])
