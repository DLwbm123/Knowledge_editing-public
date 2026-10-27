"""Read existing V133 metadata only; no model loading or remote writes."""
import hashlib
import json
from pathlib import Path


def collect(root,reference):
    root,reference=Path(root),Path(reference)
    read=lambda p:json.loads(p.read_text())
    inputs=read(root/'INPUTS_FROZEN.json');tasks=inputs['tasks']
    frozen=read(root/'EXPERT_LIBRARY_FROZEN.json')
    routes=[]
    for path in sorted((root/'outputs').glob('*.json')):
        med=read(path);qid=med['binding']['input']['query_id']
        cid=json.dumps(['M13P_160',med['mode'],med['prefix'],med['owner'],qid],separators=(',',':'))
        identity=hashlib.sha256(json.dumps(cid,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        legacy=read(reference/'M13P_160/outputs'/(identity+'.json'))
        selected=med['effective_expert'];allowed=tasks[med['owner']-1:med['owner']] if med['mode']=='single' else tasks[:med['prefix']]
        assert selected is None or selected in {t['edit_id'] for t in allowed}
        assert med['effective_state_hash']==(frozen['states'][selected] if selected else 'UNEDITED_BASE')
        assert legacy['mode']==med['mode'] and legacy['prefix']==med['prefix'] and legacy['owner']==med['owner']
        fields=('question','reference','image_sha256','prompt_ids','attention_mask','runtime','generation')
        assert all(med['binding']['judge_input'][k]==legacy['binding']['judge_input'][k] for k in fields)
        routes.append(dict(mode=med['mode'],prefix=med['prefix'],owner=med['owner'],query_id=qid,
            MED_route=med['route'],M13P_route=legacy['route'],MED_expert=selected,
            M13P_expert=legacy['effective_expert'],MED_state_hash=med['effective_state_hash'],
            MED_reused=med['prediction_reused']))
    assert len(routes)==7494
    training=[]
    for task in tasks:
        owner=task['order'];v=read(root/'training'/f'MED_{owner}.json');c=read(root/'calibration'/f'{owner}.json')
        assert v['steps']==160 and v['save_load_exact'] and len(v['curve'])==160
        first,last=v['curve'][0],v['curve'][-1]
        training.append(dict(owner=owner,coefficient=c['coefficient'],C_gain=c['C_gain'],
            C_FULL_CTX_penalty=c['C_FULL_CTX_penalty'],first_logged_source_NLL=first['source_NLL'],
            last_logged_source_NLL=last['source_NLL'],last_logged_source_step=last['step'],
            last_logged_penalty_before=last['penalty_before'],first_penalty_before=first['penalty_before'],
            actual_operator_step_sum=sum(x['matching']['actual'] for x in v['curve']),
            last_operator_step=last['matching']['actual']))
    teacher=read(root/'CURVATURE_BINDING.json')['actual_inputs']
    for value in teacher:
        binding=value['actual_teacher_binding']
        expected=[[batch,i] for batch,labels in enumerate(binding['labels'])
                  for i in range(len(labels)-1) if labels[i+1]!=-100]
        assert binding['predictor_positions']==expected
    return dict(routes=routes,training=training,source=read(root/'SOURCE_VERSION.json'),
        mask_check=read(root/'ACTUAL_MASK_DERIVATIVE_CHECK.json'),curvature_check=read(root/'V132_CURVATURE_PARITY.json'),
        curvature_admission=read(root/'CURVATURE_ADMISSION.json'),
        mapped_alias_paths=len(teacher),mapped_alias_mask_positions=sum(len(v['actual_teacher_binding']['predictor_positions']) for v in teacher),
        saved_teacher_masks_checked=len(teacher),
        cache_exists={n:(root/'curvature'/n).exists() for n in ('MED.pt','FULL_CTX.pt','COMMON_FALLBACK.pt')},
        cleanup=read(root/'INTERMEDIATE_CLEANUP.private.json'),generation_complete=read(root/'GENERATION_COMPLETE.json'))
