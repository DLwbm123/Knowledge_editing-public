"""Finite result collection/publication dependency; no new experiment scheduling."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def main(cfg):
    private=Path(cfg['private_root']);report=Path(cfg['report'])
    (private/'CLOSEOUT_ONCE').touch(exist_ok=False)
    signal.alarm(max(1,int(cfg['clock']['hard_end']-time.time()-5)))
    while True:
        assert time.time()<cfg['clock']['hard_end']-10
        code='from pathlib import Path\nimport json\nr=Path('+repr(cfg['remote_root'])+')\nprint(json.dumps({name:json.loads((r/name).read_text()) if (r/name).exists() else None for name in ["SUMMARY.json","COMPLETE.json","GUARD_FAILURE.json","EXIT.json","HEALTH.json","A/PROGRESS.json","B/PROGRESS.json","A/FAILURE.json","B/FAILURE.json"]}))'
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',cfg['python'],'-'],input=code,text=True,capture_output=True,timeout=45)
        assert result.returncode==0,result.stderr[-1000:]
        state=json.loads(result.stdout)
        (private/'LIVE_STATE.private.json').write_text(json.dumps(state,indent=2)+'\n')
        if state['COMPLETE.json']:
            summary=state['SUMMARY.json'];assert summary and summary['full_generations']==64
            (report/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
            a=summary['all']
            text=f'''# T5 同输入推理路径对照\n\n固定16任务、32输入、两个独立干净Base进程，共64次完整生成；0训练、0Judge、0名单或标签修改。\n\n本地A与官方入口B：原始tokens一致{a['tokens_equal']}/32，完整回答一致{a['answers_equal']}/32，prompt一致{a['prompts_equal']}/32。本地A复现历史tokens {a['A_historical_token_matches']}/32。\n\n回答前缀：A={a['A_prefix']}；B={a['B_prefix']}。这只是回答形式统计，不是重新计算CORE或临床正确率。\n\n输入像素精确一致{a['tensor_comparisons']['pixels']['exact']}/32，视觉特征精确一致{a['tensor_comparisons']['vision']['exact']}/32，projector输出精确一致{a['tensor_comparisons']['projector']['exact']}/32。逐层数值和状态分层见SUMMARY.json。\n\n两路在生成入口和精度上同时不同；不能仅凭输出变化归因于某一个因素，也不以较高准确率选择实现。官方入口指骨干参考推理，不是M3Bench作者实际T5执行器。工程面板含一条已标记的CT比较来源问题，不作临床准确率主张。历史UNKNOWN保持不变。\n\n参见PREREG.json及experiments/med_t5_inference_parity_v1源码。私有源绑定、实际输入张量、激活和完整回答未公开。\n'''
            (report/'REPORT_ZH.md').write_text(text)
            break
        if state['GUARD_FAILURE.json']:
            (report/'SUMMARY.json').write_text(json.dumps(dict(status='INCOMPLETE_WITH_PRESERVED_FAILURE',clock=cfg['clock'],generation_progress={arm:state[arm+'/PROGRESS.json'] for arm in ['A','B']},updates=0,Judge_calls=0),indent=2)+'\n')
            break
        time.sleep(20)
    sys.path.insert(0,str(Path(cfg['worktree'])/'experiments/med_t5_data_v1'))
    from job import publish
    publish(cfg)
    (private/'CLOSEOUT_COMPLETE.json').write_text(json.dumps(dict(time=time.time()))+'\n')


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['DIAG_LOCAL_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:
        (Path(cfg['private_root'])/'CLOSEOUT_FAILURE.private.json').write_text(json.dumps(dict(time=time.time(),error=traceback.format_exc()),indent=2));raise
