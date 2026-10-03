"""Reuse the Stage17 official-native FP16 image, prompt, mask and decoding path."""
import json
import os
from pathlib import Path
import torch


def load(config, root):
    for key in ('M3BENCH_LLAVA_SOURCE','M3BENCH_EXPECTED_LLAVA_SOURCE'):
        os.environ[key]=config['native_source']
    from m3bench_repro.editors.llava_runtime import LlavaMedEditorRuntime
    generation_path=Path(root)/'generation.private.json'
    generation_path.write_text(json.dumps(config['generation'])+'\n')
    runtime=LlavaMedEditorRuntime(device='cuda:0',run_root=Path(root),model_path=Path(config['model_path']),
        vision_path=Path(config['vision_path']),generation_config_path=generation_path,loader_mode='official_native')
    runtime.load_frozen_backbone(seed=20260912)
    runtime.model.config.use_cache=False
    if next(runtime.model.parameters()).dtype!=torch.float16:raise RuntimeError('expected native FP16 backbone')
    return runtime


def record(query):
    from m3bench_repro.editors.llava_runtime import EditorRecord
    return EditorRecord(query['query_id'],query['dataset'],query['question'],query['reference'],
                        query['question'],Path(query['image_path']),query.get('original_image_path',''),
                        0,'VERIFIED_SOURCE_ANSWER','NO_ROUTING_IN_ALPHAEDIT')


def prepared(runtime, query):
    batch=runtime.build_edit_batch(record(query))
    inputs=batch.forward_kwargs();inputs.pop('labels')
    return inputs,batch.labels


def generated(runtime, query):
    raw=runtime.adapter.prepare_inputs(query['image_path'],query['question'])
    if query.get('expected_prompt_ids') is not None and raw['input_ids'][0].tolist()!=query['expected_prompt_ids']:
        raise RuntimeError('native prompt differs from original Stage17 binding')
    if query.get('image_sha256') is not None and raw['image_sha256']!=query['image_sha256']:
        raise RuntimeError('native image differs from original Stage17 binding')
    output=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
    tokens=list(output.raw_token_ids)
    if not tokens or len(tokens)>1024:raise RuntimeError('invalid native continuation length')
    return dict(raw_answer=output.decoded_text,raw_token_ids=tokens,
                eos_reached=runtime.adapter.tokenizer.eos_token_id in tokens,
                hit_token_cap=len(tokens)==1024 and runtime.adapter.tokenizer.eos_token_id not in tokens)


def logits(runtime, query):
    inputs,labels=prepared(runtime,query)
    with torch.no_grad():
        output=runtime.model(**inputs).logits[0,:-1][labels[0,1:]!=-100].float().cpu()
    if not torch.isfinite(output).all():raise FloatingPointError('nonfinite native target logits')
    return output
