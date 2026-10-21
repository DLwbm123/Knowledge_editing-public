"""Terminal anonymous five-arm aggregates; no training feedback or raw answers."""
import contextlib
import io
import json
import os
from pathlib import Path
from experiments.fixed_input_guard8_v1 import report as shared
from experiments.fixed_input_guard8_v1.compare import long_outputs
from .run import ARMS
from .storage import owned_bytes


def main():
    root=Path(os.environ['RUN_ROOT'])
    read=lambda p:json.loads(p.read_text())
    assert read(root/'private/GENERATION_COMPLETE.json')['status']=='PASS'
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture):shared.main()
    results=json.loads(capture.getvalue())
    mechanism={}
    for arm in ARMS:
        folder=root/'private/fixed8'/arm
        assert not (folder/'active.pt').exists()
        assert not list((folder/'diagnostic_weights').glob('*.pt'))
        sources=[]
        for owner in range(1,9):
            data=read(folder/'sources'/f'{owner}.json')
            views=data.get('views',[])
            events=[e for v in views for e in v['sketch']['compressions']]
            sources.append(dict(owner=owner,qualified=data['qualified'],CE=data['CE'],
                summary_elapsed_seconds=data['summary_elapsed_seconds'],inputs_selected=data['inputs_selected'],
                history_rank=data.get('history_rank'),effective_new_history_rank=data.get('effective_new_history_rank'),
                compression_events=len(events),tail_loss=sum(e['tail_loss'] for e in events),
                shrink_loss=sum(e['shrink_loss'] for e in events),
                cleared_rows=sum(v['sketch']['cleared_rows'] for v in views),
                cleared_energy=sum(v['sketch']['cleared_energy'] for v in views),
                weighted_energy=sum(v['weights']['weighted_input_energy'] for v in views),
                post_clear_energy=sum(v['sketch']['post_clear_energy'] for v in views),
                VJPs=sum(v['numerics']['vjps'] for v in views),
                same_state=[v['same_state'] for v in views if 'same_state' in v]))
        diag=root/'private/evaluation/source6_diagnostic'/arm
        states=[]
        for path in sorted(diag.glob('*_*/METRICS.json')):
            d=read(path)
            states.append({k:d[k] for k in ('owner','slot','CE','target_margin','fixed_competitor_margin','generation','response')})
        mechanism[arm]=dict(sources=sources,first_change=read(diag/'FIRST_CHANGE.json'),diagnostic_states=states,
            summary_admission=read(folder/'ADMISSION.json')['summary_admission']['total_supervised_tokens'])
    historical=root.parent.parent/'fixed-input-context-pin8-v3-20261017/run/public/completion/RESULTS.json'
    value=dict(results=results,mechanism=mechanism,lengths=long_outputs(root,ARMS),
        historical_V3_P=read(historical)['arms']['P'],historical_reference_not_concurrent=True,
        owned_bytes_at_closure=owned_bytes(root),diagnostic_states_not_semantically_judged=True)
    destination=root/'public/completion';destination.mkdir(parents=True,exist_ok=True)
    (destination/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
    (destination/'COMPARISON.json').write_text(json.dumps(value,indent=2)+'\n')
    lines=['# 全上下文摘要五臂开发实验结果','',
        '这是既有前8源的开发比较；原V3 P为历史参照。CPU检查、原生准入通过不等于保护机制成功。','',
        '| 臂 | 接受/160 | 合格/8 | 最终T0 | 最终T1G | 最终T2G | 保护集已知错误/54 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        d=results['arms'][arm];p=d['prefixes']['8']
        rate=lambda g:str(p[g]['correct'])+'/'+str(p[g]['total'])
        lines.append(f"| {arm} | {d['formal_accepted']} | {d['qualified']} | {rate('T0')} | {rate('T1G')} | {rate('T2G')} | {p['protection']['known_wrong']} |")
    lines += ['', '完整逐源拟合、插入/最终保持、缺分、压缩损失、有效秩、逐接受更新诊断及费用见配套JSON。',
              '首次生成分叉不等于首次语义错误；低输入残差不保证自由生成正确。保留所有未合格源和历史缺分，不根据结果补训、重试或扩展。',
              'N与D的差异须结合额外VJP成本和新源学习共同解释；B/C同时改变插入及压缩频率，不能解释为纯位置数量效应。','']
    (destination/'REPORT_ZH.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=results['status'],report_files=[p.name for p in destination.iterdir()],owned_bytes=value['owned_bytes_at_closure'])))


if __name__=='__main__':main()
