"""First20 report using unchanged single metrics, masks, contrasts and intervals."""
from pathlib import Path
import sqlite3
import time
import astra_queue as a
import interim20 as protocol
from selected_report146 import calculate


def main():
    q,RUN=a.q,a.RUN;queue=RUN/'private'/protocol.QUEUE
    assert (queue/'ALL_WORKERS_COMPLETE.json').exists()
    assert all((queue/'workers'/str(i)/'SCORER_DONE.json').exists() for i in range(4))
    db=sqlite3.connect('file:'+str(queue/'queue.sqlite')+'?mode=ro',uri=True)
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    records={};scores={}
    for path,binding,status,score in db.execute('SELECT c.path,c.output_binding,p.status,p.correct FROM consumer c JOIN payload p ON p.key=c.payload_key'):
        data=q.read(Path(path));assert q.digest(data)==binding
        key=tuple(data[k] for k in ('method','mode','prefix','owner'))+(data['binding']['input']['query_id'],)
        assert key not in records;records[key]=data;scores[key]=score if status=='FORMAT_VALID' else None
    result,owners=calculate(q.read(RUN/'private/ORIGINAL146_TASKS.json')[:20],q.read(RUN/'private/ORIGINAL_BASE_MASK146.json'),records,scores,interim20=True)
    result.update(status='INTERIM20_SCORING_RESOLVED',full146_complete=False,interim_descriptive_only=True,
        selection_or_stopping_authorized=False,bank146_evaluated=False,consumers=protocol.CONSUMERS)
    q.write(RUN/'private/INTERIM20_OWNER_AUDIT.json',owners)
    q.write(RUN/'public/INTERIM20_RESULTS.json',result)
    lines=['# 原序前20例中期观察','','仅原146固定single评测的子集；不代表完整146专家库，也不据此改变训练或门槛。','',
        '| 方法 | FIT宏平均% | T2G宏平均% | 新损伤 / fresh Base正确 |','|---|---:|---:|---:|']
    for method in ('M13','FISHER_MIN'):
        panel=result['panels'][method]['single_1'];damage=result['protection'][method+'_single_1']
        lines.append(f"| {method} | {panel['FIT']['macro']} | {panel['T2G']['macro']} | {damage['new_damage']} / {damage['fresh_Base_correct']} |")
    lines+=['','逐编辑保留、双区间、缺失界和支持分母见聚合JSON；owner明细私有。',
        '无同配方M1/M3的20例匹配对照，不能宣称达到M3保护水平；全部DEV、无独立确认或医学安全结论。']
    (RUN/'public/INTERIM20_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    q.write(RUN/'private/INTERIM20_REPORT_COMPLETE1.json',dict(status='COMPLETE',epoch=time.time()))


if __name__=='__main__':main()
