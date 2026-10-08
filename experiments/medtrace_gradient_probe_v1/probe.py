"""Read-only W0 gradient probe. Source replay is a diagnostic, never a method component."""
import os
import random
import statistics
import time
import traceback
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
import replay

p, c, RUN = replay.p, replay.c, replay.RUN
PARENT = Path(os.environ['REPLAY_PARENT'])
GPUS = (4, 5, 6, 7)
INDICES = tuple(range(0, 32, 2))


def parts(logits, labels, eos):
    labels = labels[:, 1:]
    mask = labels != -100
    logits = logits[:, :-1].float()[mask]
    targets = labels[mask]
    losses = F.cross_entropy(logits, targets, reduction='none')
    tail = targets == eos
    assert tail.any() and (~tail).any()
    return losses.mean(), losses[tail].sum()/len(losses), dict(
        tokens=len(losses), content_tokens=int((~tail).sum()), eos_tokens=int(tail.sum()),
        content_nll=float(losses[~tail].mean().detach()), eos_nll=float(losses[tail].mean().detach()))


def cosine(a, b):
    denominator = a.norm()*b.norm()
    return float(torch.dot(a, b)/denominator) if denominator > 0 else None


def stats(a, b):
    return dict(norm=float(a.norm()), cosine_to_edit=cosine(a, b),
        dot_edit=float(torch.dot(a, b)), norm_over_edit=float(a.norm()/b.norm()) if b.norm() > 0 else None)


def selfcheck():
    torch.manual_seed(42)
    x = torch.randn(1, 5, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 3, 4, 2]])
    full, eos, detail = parts(x, labels, 2)
    a = torch.autograd.grad(full, x, retain_graph=True)[0]
    b = torch.autograd.grad(eos, x, retain_graph=True)[0]
    d = torch.autograd.grad(full-eos, x)[0]
    assert torch.allclose(a, b+d) and detail['tokens'] == 3 and detail['eos_tokens'] == 1
    assert torch.count_nonzero(a[:, 0]) == 0 and torch.count_nonzero(a[:, -1]) == 0
    assert cosine(torch.zeros(3), torch.ones(3)) is None
    assert cosine(torch.ones(3), -torch.ones(3)) < -.999
    return dict(status='PASS', shifted_target_mask=True, gradient_decomposition=True, zero_norm_handled=True)


def gradients(runtime, hook, expert, batch):
    labels = batch.labels
    target = labels[labels != -100]
    eos_id = runtime.adapter.tokenizer.eos_token_id
    assert tuple(target.tolist()) == batch.target_token_ids and int(target[-1]) == eos_id
    assert batch.image_token_count == 1 and torch.all(labels[:, :batch.target_start_expanded] == -100)
    if batch.attention_mask is not None:
        assert torch.all(labels[~batch.attention_mask.bool()] == -100)
    hook.set_teacher_routing(labels)
    output = runtime.model(**batch.forward_kwargs())
    full, eos, detail = parts(output.logits, labels, eos_id)
    assert torch.isfinite(output.loss) and torch.allclose(output.loss.float(), full, atol=1e-6, rtol=1e-5)
    params = tuple(expert.parameters())
    total = torch.autograd.grad(output.loss, params, retain_graph=True)
    tail = torch.autograd.grad(eos, params)
    flatten = lambda values: torch.cat([v.detach().float().reshape(-1).cpu() for v in values])
    a, b = flatten(total), flatten(tail)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert all(v.grad is None for v in runtime.model.parameters())
    detail.update(loss=float(full.detach()), loss_parity_error=float((output.loss-full).abs().detach()),
        gradient_norm=float(a.norm()), eos_contribution_norm=float(b.norm()), content_contribution_norm=float((a-b).norm()),
        content_eos_cosine=cosine(a-b, b), core_norms={k:float(v.norm()) for (k,_),v in zip(expert.named_parameters(), total)},
        supervision_checks_pass=True)
    return a, b, detail


def prepare():
    check = selfcheck()
    selected = [p.tasks()[i] for i in INDICES]
    old = c.read(Path(os.environ['DIAGNOSIS_PARENT'])/'private/TRAJECTORY_LOCK.json')
    assert old['selected_indices'] == list(INDICES) and old['selected_edits'] == [t['edit_id'] for t in selected]
    rows = c.read(PARENT/'private/REPLAY_FIT.json')
    assert len(rows) == 192 and all(x['role'] == 'REPLAY_FIT' for x in rows)
    lock = []
    for index, t in zip(INDICES, selected):
        order = list(range(4)); random.Random(t['seed']).shuffle(order)
        groups = list(range(64)); random.Random(t['seed']).shuffle(groups)
        replay_order = [3*g+k for g in groups for k in range(3)]
        historical = c.read(PARENT/'private/weights/SOURCE_REPLAY192'/t['anonymous_edit']/'TRAINING.json')['binding']
        assert historical['fit_order'] == order and historical['replay_order'] == replay_order
        assert [rows[i]['answer_kind'] for i in replay_order[:12]] == ['yes','no','open']*4
        lock.append(dict(index=index, edit_id=t['edit_id'], fit_order=order, replay_rows=replay_order[:12],
            W0=str(p.initial(t)), source_groups=[rows[i]['source_group'] for i in replay_order[:12]]))
    c.write(RUN/'private/LOCK.json', dict(items=lock, source_rows=rows))
    c.write(RUN/'public/ADMISSION.json', dict(status='PASS', diagnostic_only=True, replay_in_method=False,
        selected_indices=list(INDICES), experts=16, pairs=192, forwards=272, gradient_extractions=544,
        optimizer_updates=0, new_generations=0, new_Judge_attempts=0, new_weights=0, selfcheck=check))


def worker():
    part = int(os.environ['PARTITION']); gpu = int(os.environ['GPU'])
    assert gpu == GPUS[part]
    lock = c.read(RUN/'private/LOCK.json')
    records = []
    with p.lease(gpu):
        runtime, _ = c.load(gpu)
        for item in lock['items'][part::4]:
            c.budget(); t = p.tasks()[item['index']]
            saved = p.load_state(p.initial(t))
            expert = p.expert(saved['expert'], t['seed'], runtime.device)
            before = {k:v.detach().cpu().clone() for k,v in expert.state_dict().items()}
            hook = replay.MedTraceLayerHook(runtime.get_module(c.LAYER), expert); hook.attach()
            samples, vectors = [], []
            try:
                for kind, record in [('native',c.record(t))]+[('fit',replace(c.record(t),question=q)) for q in t['fit_questions']]:
                    a, b, d = gradients(runtime, hook, expert, runtime.build_edit_batch(record))
                    vectors.append(a); samples.append(dict(kind=kind, **d))
                assert len(vectors) == 5
                pairs = []
                for pos, index in enumerate(item['replay_rows']):
                    row = lock['source_rows'][index]
                    a, b, d = gradients(runtime, hook, expert, runtime.build_edit_batch(replay.record(row, t)))
                    edit = .5*(vectors[0]+vectors[1+item['fit_order'][pos%4]])
                    pairs.append(dict(position=pos+1, kind=row['answer_kind'], **d,
                        edit_norm=float(edit.norm()), replay=stats(a, edit), eos=stats(b, edit), content=stats(a-b, edit),
                        combined_norm=float((edit+a).norm()), combined_dot_edit=float(torch.dot(edit+a, edit))))
                    print('PROBE',item['index'],pos+1,flush=True)
                assert all(torch.equal(before[k], v.detach().cpu()) for k,v in expert.state_dict().items())
                assert all(v.grad is None for v in expert.parameters())
                records.append(dict(expert_index=item['index'], fixed_W0_unchanged=True, edit_samples=samples, pairs=pairs))
                c.write(RUN/'private/experts'/f"{item['index']:03d}.json", records[-1])
            finally:
                hook.detach()
            del expert, vectors
        peak = torch.cuda.max_memory_allocated()
    p.done('WORKER_'+str(part), dict(experts=len(records), peak_allocated_bytes=peak))


def report():
    records = [c.read(RUN/'private/experts'/f'{i:03d}.json') for i in INDICES]
    assert len(records) == 16 and all(x['fixed_W0_unchanged'] for x in records)
    pairs = [row for x in records for row in x['pairs']]
    assert len(pairs) == 192
    summary = {}
    for kind in ('yes','no','open'):
        rows = [x for x in pairs if x['kind'] == kind]
        summary[kind] = dict(pairs=len(rows), negative_full=sum(x['replay']['cosine_to_edit'] is not None and x['replay']['cosine_to_edit']<0 for x in rows),
            negative_content=sum(x['content']['cosine_to_edit'] is not None and x['content']['cosine_to_edit']<0 for x in rows),
            negative_eos=sum(x['eos']['cosine_to_edit'] is not None and x['eos']['cosine_to_edit']<0 for x in rows),
            median_full_cosine=statistics.median(x['replay']['cosine_to_edit'] for x in rows if x['replay']['cosine_to_edit'] is not None),
            median_norm_ratio=statistics.median(x['replay']['norm_over_edit'] for x in rows if x['replay']['norm_over_edit'] is not None),
            median_content_tokens=statistics.median(x['content_tokens'] for x in rows),
            median_eos_to_content_norm=statistics.median(x['eos_contribution_norm']/x['content_contribution_norm'] for x in rows if x['content_contribution_norm']>0),
            negative_combined_projection=sum(x['combined_dot_edit']<0 for x in rows),
            expert_mean_cosines=[dict(expert_index=e['expert_index'],value=statistics.mean(x['replay']['cosine_to_edit'] for x in e['pairs'] if x['kind']==kind and x['replay']['cosine_to_edit'] is not None)) for e in records])
    ledger=c.read(RUN/'RESOURCE_LEDGER.json'); inherited=c.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions']) == 4 and all(x.get('ended_epoch') for x in ledger['gpu_sessions'])
    assert not list((RUN/'private').rglob('*.pt'))
    result=dict(status='COMPLETE_DIAGNOSTIC_ONLY', replay_in_method=False, summary=summary, records=records,
        supervision_checks_pass=True, W0_unchanged=True, Base_gradient=False, optimizer_updates=0, new_generations=0,
        new_Judge_attempts=0, owned_weights=0, forwards=272, gradient_extractions=544,
        incremental_GPU_hours=(ledger['gpu_seconds_used']-inherited['gpu_seconds_used'])/3600,
        cumulative_GPU_hours=ledger['gpu_seconds_used']/3600, cumulative_Judge_attempts=ledger['Judge_attempts'])
    c.write(RUN/'public/RESULTS.json',result)
    lines=['# W0监督与梯度诊断','', '仅诊断对照；回放不属于用户方法。16个固定专家、192个来源/编辑配对，所有权重保持W0。', '',
        '| 来源答案 | 负夹角/64 | 夹角余弦中位数 | 来源/编辑范数中位数 | EOS/内容范数中位数 |', '|---|---:|---:|---:|---:|']
    for kind, x in summary.items():
        lines.append(f"| {kind} | {x['negative_full']}/64 | {x['median_full_cosine']:.4f} | {x['median_norm_ratio']:.3f} | {x['median_eos_to_content_norm']:.3f} |")
    lines += ['', '监督mask、target/EOS绑定、原模型损失与因果shift token平均CE一致性检查均通过。EOS与内容均保留原总token分母，不改变原损失权重。', '',
        '这些是teacher-forced训练来源在W0的原始梯度关系，不是Adam更新轨迹或实际生成性能。负夹角不能单独证明T2G损伤的因果机制；没有负夹角也不排除后续状态或路由机制。配对并非192个独立专家；逐专家均值及所有匿名数值见RESULTS.json。', '',
        f"完成272次前向、544次梯度提取；0次参数更新、0条新生成、0次新Judge、0个新checkpoint。新增GPU进程小时{result['incremental_GPU_hours']:.6f}，累计{result['cumulative_GPU_hours']:.6f}；累计Judge {result['cumulative_Judge_attempts']}。", '',
        '方法边界：不采用回放、不自动训练梯度手术或其他干预、不自动追加任何实验。']
    (RUN/'public/REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    p.done('REPORT_COMPLETE')


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('probe.py','probe_worker',gpu,part) for part,gpu in enumerate(GPUS)])
    report()


if __name__ == '__main__':
    try:
        {'prepare':prepare,'probe_worker':worker,'probe_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','all')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
