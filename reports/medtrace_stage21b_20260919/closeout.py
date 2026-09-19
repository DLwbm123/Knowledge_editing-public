"""B closeout from frozen scored outputs; no new inference or judgments."""
import json, sys
from pathlib import Path


def run(root, dest):
    root, dest = Path(root), Path(dest)
    pub = root/'public'
    read = lambda name: json.loads((pub/name).read_text())
    write = lambda name, value: (dest/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    assert read('REPORT_STATUS.json')['stage_complete']
    assert read('EXIT_FACT_FIXED_L31_DOWN.json')['exit_code'] == 0
    assert read('ENVIRONMENT_ACCEPTANCE.json')['status'] == 'PASS'
    gpu, judge = read('GPU_BUDGET_LEDGER.json'), read('BUDGET_LEDGER.json')
    used = sum(s['seconds'] for s in gpu['sessions'])
    assert used <= 12600 and judge['new_judgment_items_dispatched'] <= 800
    profile = read('RESOURCE_PROFILE.json'); assert profile['edits'] == 45
    rows = read('COMPARISON_DERIVED.json')['results']
    def metric(arm, panel, name):
        return next(r for r in rows if (r['arm'],r['N'],r['panel'],r['metric']) == (arm,45,panel,name))
    table = '|方法|H面板|总体正确|Base正确子集保持|Base错误子集Fix|来源宏保持|来源数|\n|---|---|---|---|---|---|---|\n'
    for arm in ('Stage20_FACT_H','NO_H_HSIC','FACT_FIXED_L31_DOWN','Stage20_BE'):
        for panel in ('old_H_eval','new_H_eval'):
            a,r,f = [metric(arm,panel,m) for m in ('accuracy','Retention','Fix')]
            cell = lambda x: f"{x['numerator']:g}/{x['denominator']}" if x['value'] is not None else 'NA'
            macro = f"{r['source_macro']:.1%}" if r['source_macro'] is not None else 'NA'
            table += f"|{arm}|{panel}|{cell(a)}|{cell(r)}|{cell(f)}|{macro}|{r['source_count']}|\n"
    route = read('ROUTING_MECHANISM_ANALYSIS.json')
    pairs = [r for r in route['output_differences'] if r['N']==45 and 'FACT_FIXED_L31_DOWN' in (r['A'],r['B'])]
    write('B_ENDPOINT_ROUTING_SUMMARY.json',dict(pairs=pairs,main='Natural R0; no forced-expert replacement',scope='113 queries per final bank'))
    analysis = '# Stage21B 完成分析\n\nB 的45条编辑、当次写入与真实11/19/32/45前缀全部生成并评分，298项正式输出全部覆盖。两次机械回放另计，不当独立样本。\n\n'+table
    analysis += '\nL31 down_proj 保留了45/45 native纠错，但没有改善H保护：新H保持7/18，与L30 NO_H相同；严格新U保持0/8，低于L30 FACT和NO_H的7/8。旧U保持1/7，L30 FACT为3/7、NO_H为6/7。说明这一固定DEV流上存在明显位置相关退化；不能把它单独归因于H、模块或容量。\n\n'
    analysis += '与BE L31 up_proj相比，B新H保持7/18对3/18、旧H1/4对0/4；native同为45/45，新U均0/8。只支持局部描述性优势，不能宣称全面优越。B与BE只同Transformer block，并非同模块writer-only对照；B与历史L30结果还跨已登记的GPU设备lane。预定1/19/45 Base-OFF token一致及两次save/load/OFF检查通过，不代表全输入跨设备数值等价。\n\n'
    analysis += 'Q1–Q3：H未提高已饱和native；H/U仍有取舍，不能普遍归因保护。Q4–Q6：原L21前提已由append-only修订改为实际L31，现有结果不足以分离层位、模块、writer和训练差异。Q7：历史HSIC均L30，动态选层收益不支持。Q8：真实路由、输出差异和切换正误转移见ROUTING_MECHANISM_ANALYSIS；条件诊断不替代全输入结果。Q9：纯writer张量、含optimizer文件和GPU时间分别测量。Q10：45条已查看DEV、单顺序和有限来源，不支持长序列、患者独立或临床验证。\n\n'
    analysis += f"B GPU进程墙钟{gpu['sessions'][-1]['seconds']:.1f}秒；A+B累计{used:.1f}/12600秒。B新增26项判定，A+B累计{judge['new_judgment_items_dispatched']}/800；缓存命中{judge['exact_reuse_count']}是消费者累计，不是独立QA。峰值已分配显存{profile['peak_gpu_allocated_bytes']/1024**3:.2f}GiB；每writer纯张量{profile['mean_writer_tensor_bytes']/1024**2:.3f}MiB。实际货币费用提供商未暴露，NA而非0，tokens在账本。\n\n"
    analysis += 'Stage22 H主线L30在B结果前已锁定；全部新旧评价均DEV/regression。后续28,800秒/2,000项总授权不增加，Stage21余额不转入。保留现有bank/router/W0和原始评分绑定，当前空间足够，不作清理。\n'
    (dest/'STAGE21B_ANALYSIS_ZH.md').write_text(analysis)
    advisor=(pub/'ADVISOR_UPDATE_ZH.md').read_text().replace('回放须完全一致','回放已通过完全一致检查')
    advisor += '\n完成结论：L31未改善H保护，且严格新U保持降至0/8；后续按事先注册的L30 H/U权重与支持覆盖计划继续。完整H总体正确、保持/Fix和来源宏平均见STAGE21B_ANALYSIS_ZH.md，不能只凭PairCorrect宣称普遍优势。\n'
    (dest/'ADVISOR_UPDATE_ZH.md').write_text(advisor)
    names=['FACT_L31_RESULTS.json','FACT_L31_RESULTS.csv','COMPARISON_DERIVED.json','RISK_DIFFERENCES.json','ROUTING_MECHANISM_ANALYSIS.json','RESOURCE_PROFILE.json','GPU_BUDGET_LEDGER.json','BUDGET_LEDGER.json','ENVIRONMENT_ACCEPTANCE.json','REPORT_STATUS.json','STAGE20_IMMUTABILITY_CHECK.json','INTEGRATION_001.json','INTEGRATION_012.json']
    for name in names:
        data=(pub/name).read_bytes()
        assert not any(s in data for s in (b'"raw_answer"',b'"raw_token_ids"',b'"image_path"',b'"canonical_edit_id"',b'/Users/',b'/root/',b'/remote-home/')),name
        (dest/name).write_bytes(data)
    write('DELIVERY_STATUS.json',dict(Stage21B_generation='COMPLETE',Stage21B_scoring='COMPLETE',Stage21_A_and_B='COMPLETE',prefixes=[11,19,32,45],GPU_seconds=used,new_judgments=judge['new_judgment_items_dispatched'],publication='Separate verified private receipt; this file does not itself prove publication',Stage22_23='NOT_COMPLETE',Stage20_and_A_immutable=True))


if __name__ == '__main__':
    run(*sys.argv[1:])
