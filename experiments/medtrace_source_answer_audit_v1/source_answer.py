"""Generate unedited Base answers and audit the annotation target boundary."""
import os
import time
import traceback
from pathlib import Path
import torch
import probe as previous

p, c, RUN = previous.p, previous.c, previous.RUN
GPUS = previous.GPUS


def rows():
    basis, held = previous.split()
    result = []
    for role, values in [('BASIS', basis), ('HELDOUT', held)]:
        groups = list(dict.fromkeys(x['source_group'] for x in values))
        for row in values:
            result.append(dict(row, audit_role=role, audit_group=groups.index(row['source_group'])))
    return result


def output(row):return RUN/'private/outputs'/(c.digest(row['query_id'])+'.json')


def plan():
    values = rows()
    assert len(values) == 192 and len({r['query_id'] for r in values}) == 192
    groups = {r['source_group'] for r in values}
    excluded = {r['source_group'] for r in c.read(RUN/'private/EVAL_LEDGER.json')['queries'].values()}
    excluded.update(r['source_group'] for rs in c.read(Path(os.environ['BASE_ROOT'])/'private/U_ROLES.json').values() for r in rs)
    assert not groups & excluded and all(r['role'] == 'REPLAY_FIT' for r in values)
    assert {r['answer_kind'] for r in values} == {'yes','no','open'}
    lock = dict(generations=192, annotation_forwards=192, backward_calls=0, updates=0,
        max_new_Judge=192, persistent_checkpoints=0, groups=64, groups_per_role=32,
        roles=['BASIS','HELDOUT'], role_binding=c.digest(values), unedited_Base=True,
        source=c.read(RUN/'private/GPU_SOURCE_VERSION.json'))
    c.write(RUN/'private/ANSWER_LOCK.json', lock)
    c.write(RUN/'private/ANSWER_ROWS.json', values)
    c.write(RUN/'public/ADMISSION.json',dict(status='PASS',**{k:v for k,v in lock.items() if k not in ('role_binding','source')}))
    p.done('PLAN_COMPLETE')


def worker():
    part = int(os.environ['PARTITION'])
    with p.lease(GPUS[part]):
        runtime, bindings = c.load(GPUS[part])
        assert not runtime.get_module(c.LAYER)._forward_hooks
        assert runtime.generation_config['do_sample'] is False
        tokenizer = runtime.adapter.tokenizer
        for row in c.read(RUN/'private/ANSWER_ROWS.json')[part::6]:
            c.budget(); dest = output(row)
            assert not dest.exists(), 'No implicit generation retry'
            raw = runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
            began = time.time()
            with torch.inference_mode():g = runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
            answer = dict(raw_answer=g.decoded_text,raw_token_ids=list(g.raw_token_ids))
            batch = runtime.build_edit_batch(previous.scoped.replay.record(row,p.tasks()[0]))
            prefix = int(raw['input_ids'].shape[1])
            assert torch.equal(batch.raw_input_ids[:,:prefix],raw['input_ids']), 'Question prefix changed between annotation and generation'
            with torch.inference_mode():v = runtime.model(**batch.forward_kwargs())
            mask = batch.labels[:,1:] != -100
            logits = v.logits[:,:-1][mask].detach().double()
            target = batch.labels[:,1:][mask]
            assert len(target) >= 2 and int(target[-1]) == tokenizer.eos_token_id
            logp = logits.log_softmax(-1); pred = logits.argmax(-1)
            nll = -logp[torch.arange(len(target),device=target.device),target]
            full = runtime._tokenize_prompt(runtime.adapter._prompt(row['question'],g.decoded_text))
            prefix_roundtrip = torch.equal(full[:,:prefix],raw['input_ids'])
            reencoded = full[0,prefix:].tolist() if prefix_roundtrip else []
            leading = 0
            for token in batch.target_token_ids[:-1]:
                if tokenizer.decode([token],skip_special_tokens=False).strip():break
                leading += 1
            audit = dict(prefix_exact=True, tokens=len(target), NLL=float(nll.mean()),
                content_NLL=float(nll[:-1].mean()), EOS_NLL=float(nll[-1]),
                strict_all_correct=bool((pred==target).all()), content_all_correct=bool((pred[:-1]==target[:-1]).all()),
                EOS_correct=bool(pred[-1]==target[-1]), first_target_correct=bool(pred[0]==target[0]),
                first_teacher_equals_first_generated=bool(answer['raw_token_ids'] and int(pred[0])==answer['raw_token_ids'][0]),
                decoded_target_matches_annotation=tokenizer.decode(batch.target_token_ids,skip_special_tokens=True).strip()==row['reference'].strip(),
                target_leading_empty_tokens=leading, generated_roundtrip_prefix_exact=prefix_roundtrip,
                generated_roundtrip_tokens_exact=reencoded==answer['raw_token_ids'],
                generated_tokens=len(answer['raw_token_ids']), at_generation_cap=len(answer['raw_token_ids'])>=runtime.generation_config['max_new_tokens'])
            annotated = dict(row,image_sha256=raw['image_sha256'])
            bind = dict(question=row['question'],reference=row['reference'],image_sha256=raw['image_sha256'],
                image_path=row['image_path'],prompt_ids=raw['input_ids'][0].tolist(),
                attention_mask=raw['attention_mask'][0].tolist(),runtime=next(iter(bindings.values()))['runtime'],generation=runtime.generation_config)
            c.write(dest,dict(binding=dict(input=annotated,judge_input=bind,
                execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'),lock=c.digest(c.read(RUN/'private/ANSWER_LOCK.json'))),
                R0=answer,audit=audit,seconds=time.time()-began,unedited_Base=True,
                private_target_tokens=list(batch.target_token_ids),private_first_teacher_token=int(pred[0]),private_roundtrip_tokens=reencoded))
            print('ANSWER',part,row['audit_role'],row['audit_group'],row['answer_kind'],flush=True)
        assert not any(v.grad is not None for v in runtime.model.parameters())
    p.done('ANSWER_'+str(part))


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('source_answer.py','source_answer_worker',g,i) for i,g in enumerate(GPUS)])
    assert all(output(r).exists() for r in c.read(RUN/'private/ANSWER_ROWS.json'))
    p.done('GENERATION_COMPLETE')
    pipeline.wait([pipeline.launch('answer_queue.py','answer_ingest')])
    root = RUN/'private/judge_answer_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget()
        assert not list((root/'workers').glob('*/SCORER_FAILURE.json'))
        time.sleep(30)
    pipeline.wait([pipeline.launch('answer_report.py','answer_report')])


if __name__ == '__main__':
    try:
        {'source_answer_plan':plan,'source_answer_worker':worker,'source_answer_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
