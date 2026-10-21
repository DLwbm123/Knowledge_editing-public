"""Read-only retention on the already qualified, frozen adaptive endpoint."""
import json
import os
import time
from pathlib import Path
import torch
from .coefficient_runtime import native_weight
from .dual_execution import GenerationBudgetExhausted


def consume(runtime,layer,states,result,weight,config,budget,root):
    master=Path(os.environ['RUN_LOCALITY_MASTER'])
    panel=json.loads((master/'private/LOCALITY_PANEL.json').read_text())
    scores=json.loads((master/'private/BASE_SCORES.json').read_text())
    manifest=json.loads((root/'private/binding.json').read_text())
    bindings=json.loads(Path(os.environ['RUN_OLD_BINDINGS']).read_text())
    bound=next(iter(bindings.values()))
    if bound['generation']!=runtime.generation_config or bound['runtime']['model_snapshot_sha']!=manifest['model_snapshot']:
        raise PermissionError('Base cache runtime binding mismatch')
    rows=panel['rows']
    if [x['alias'] for x in rows]!=config['panel_aliases'] or not panel['based_only_on_Base']:
        raise PermissionError('Frozen Base-only panel required')
    qualification=json.loads((root/'private/SOURCE_QUALIFICATION.json').read_text())
    if qualification['arms']['ADAPT']['final_correct'] is not True or qualification['arms']['ADAPT']['protection_feasible'] is not True:
        raise PermissionError('Actual source qualification required')
    expected=json.loads((root/'private/EXPECTED_FINAL.json').read_text())
    if result['arms']['ADAPT']['final']!=expected:
        raise RuntimeError('Original successful endpoint metrics differ')
    eos=runtime.model.generation_config.eos_token_id
    eos=eos if isinstance(eos,(list,tuple)) else [eos]
    # References are only serialized for the post-run scorer; never used by the model.
    result['retention']=[]
    with native_weight(runtime.model,layer,weight(states['ADAPT'][1][0])):
        for item in rows:
            alias=item['alias'];query=item['query']
            cached=json.loads((master/f'basequalify/private/{alias}_GENERATION.json').read_text())
            if scores['scores'][alias] is not True or item['base_score'] is not True or cached['status']!='COMPLETE' or cached['raw_answer']!=item['base_answer'] or cached['raw_token_ids'][-1] not in eos:
                raise PermissionError('Base correct complete cache binding mismatch')
            packet=dict(alias=alias,question=query['question'],gold_answer=query['reference'],base_answer=item['base_answer'],base_correct=True)
            with (root/f'private/{alias}_RESERVED.json').open('x') as f:json.dump(dict(epoch=time.time()),f)
            budget.begin_generation()
            try:
                raw=runtime.adapter.prepare_inputs(Path(query['image_path']),query['question'],None)
                if raw['image_sha256']!=query['image_sha256']:
                    raise PermissionError('Frozen panel image mismatch')
                with torch.inference_mode():output=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                ids=list(output.raw_token_ids)
                complete=bool(ids) and ids[-1] in eos and len(ids)<=config['max_new_tokens']
                packet.update(status='COMPLETE' if complete else 'INCOMPLETE_NO_RETRY',raw_base_answer=output.decoded_text,raw_token_ids=ids)
            except GenerationBudgetExhausted:
                packet['status']='FORWARD_CAP_NO_RETRY'
            finally:budget.generating=False
            with (root/f'private/{alias}_GENERATION.json').open('x') as f:json.dump(packet,f)
            result['retention'].append(dict(alias=alias,status=packet['status'],generation_forwards=budget.request_forwards,generated_tokens=len(packet['raw_token_ids']) if 'raw_token_ids' in packet else None))
    result['source_generation_calls']=0
    result['new_optimization_steps']=0
    result['accepted_path_reconstruction_points']=27


def main():
    from .rank_reachability_entry import main as run
    run('vigor_rankRetention_v1_20261022',consume)


if __name__=='__main__':main()
