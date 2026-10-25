"""Normalize existing primary AlphaEdit outputs without any new model execution."""
import json
import os
from pathlib import Path
import time
from reuse import consumer_id, name, read, write

ROOT = Path(os.environ['V131_ROOT'])
SOURCE = Path(os.environ['V131_ALPHA_SOURCE'])
OUT = ROOT / 'ALPHAEDIT_MM'


def main():
    assert not (OUT/'GENERATION_COMPLETE.json').exists()
    config = read(SOURCE/'CONFIG.private.json')
    assert read(SOURCE/'RESULT.json')['state'] == 'GENERATED_NOT_SCORED'
    assert config['arms']['multi_layer']['role'] == 'primary_AlphaEdit'
    assert config['arms']['multi_layer']['layers'] == [19,20,21,22,23]
    assert read(SOURCE/'multi_layer/STATUS.json')['single'] == 146
    assert read(SOURCE/'multi_layer/STATUS.json')['sequential'] == 146
    tasks = config['tasks']
    assert len(tasks) == 146 and [t['order'] for t in tasks] == list(range(1,147))
    assert read(SOURCE/'multi_layer/sequential/146.RECEIPT.json')['history_columns'] == {
        str(layer):146 for layer in [19,20,21,22,23]}
    raw = {phase:{} for phase in ('single','sequential')}
    for phase in raw:
        for path in (SOURCE/'multi_layer'/phase).glob('*.OUTPUT.private.json'):
            value = read(path)
            assert value['edit_id'] == tasks[value['index']-1]['edit_id']
            raw[phase][value['index']] = (path,value)
        assert len(raw[phase]) == 146
    plan = [r for r in read(ROOT/'EVAL_PLAN.json') if r['method']=='ALPHAEDIT_MM'
            and r['query'].get('audit_role') != 'FIT']
    assert len(plan) == 5722
    for row in plan:
        qid = row['query']['query_id']
        query = config['queries'][qid]
        assert all(query[k] == row['query'][k] for k in ('question','reference','image_sha256'))
        phase = 'single' if row['mode']=='single' else 'sequential'
        index = row['owner'] if phase=='single' else row['prefix']
        path, old = raw[phase][index]
        prediction = old['outputs'][qid]
        assert set(prediction) == {'raw_answer','raw_token_ids','eos_reached','hit_token_cap'}
        assert prediction['raw_token_ids'] and isinstance(prediction['raw_answer'],str)
        binding = dict(query_id=qid, question=query['question'], reference=query['reference'],
            image_sha256=query['image_sha256'], image_path=query['image_path'],
            prompt_ids=query['expected_prompt_ids'],
            runtime=dict(source_protocol=config['protocol_version'],source_commit=config['source_commit'],
                native_source=config['native_source'],model_path=config['model_path'],
                vision_path=config['vision_path'],dtype='float16'), generation=config['generation'])
        value = dict(method='ALPHAEDIT_MM',mode=row['mode'],prefix=row['prefix'],owner=row['owner'],
            consumer_id=consumer_id(row),binding=dict(input=row['query'],judge_input=binding),
            R0={k:prediction[k] for k in ('raw_answer','raw_token_ids')},
            at_cap=prediction['hit_token_cap'],EOS=prediction['eos_reached'],
            prediction_reused=True,new_GPU_generation=False,
            cumulative_parameter_state=row['mode']!='single',
            provenance=dict(kind='EXISTING_PRIMARY_ALPHAEDIT_MULTILAYER',prediction=str(path),
                source_query=qid,source_config=str(SOURCE/'CONFIG.private.json'),
                source_receipt=str(path.with_name(f'{index:03d}.RECEIPT.json'))))
        write(OUT/'outputs'/(name(consumer_id(row))+'.json'),value)
    write(OUT/'GENERATION_COMPLETE.json',dict(status='EXISTING_COMPLETE_GENERATION_NORMALIZED',
        original_raw_predictions=dict(single=2378,sequential=3355),
        current_nonFIT_consumers=5722,all146_single_and_actual_sequence=True,
        prefixes=[1,50,100,146],extra_FIT_not_measured_not_zero=1772,
        new_GPU_generations=0,new_training_updates=0,primary_arm_fixed_before_results=True,time=time.time()))


if __name__ == '__main__':
    main()
