"""Bounded CPU checks for curriculum boundaries, source gradients and reporting."""
import copy
import os
import sqlite3
import time
from pathlib import Path
import torch
import curriculum_protocol as f
import curriculum_runtime as worker
import curriculum_report as report
import balanced_fisher

def main():
    r,c,run=worker.r,worker.c,worker.RUN
    selected=[f.selected(phase,i) for phase,n in f.PHASES for i in range(1,n+1)]
    assert len(selected)==540 and sum(map(len,selected))==940
    weights=[sum(w for row in selected for i,w in row if i==j) for j in range(5)]
    assert weights==[340.,50.,50.,50.,50.] and selected[139]==((0,1.),) and selected[140]==selected[220]==((0,.5),(1,.5))
    e=torch.nn.Linear(1,1,bias=False);e.weight.data.fill_(1.)
    class Runtime:
        def compute_loss(self,b):return (e(torch.tensor([[float(b)]]))**2).sum()
    class Hook:
        def set_teacher_routing(self,_):pass
    class Batch(int):
        labels=None
    batches=[Batch(i+1) for i in range(5)]
    old=copy.deepcopy(r.COUNTS)
    worker.gradients(Runtime(),Hook(),e,batches,'native',1);assert e.weight.grad.item()==2
    worker.gradients(Runtime(),Hook(),e,batches,'A2',4);assert e.weight.grad.item()==26
    assert r.COUNTS['backwards']-old['backwards']==3;r.COUNTS.clear();r.COUNTS.update(old)
    assert balanced_fisher.selfcheck()['status']=='PASS' and balanced_fisher.direction_selfcheck()['status']=='PASS'
    packet=worker.w.packet();assert worker.w.contract.source_admission(packet)['status']=='PASS'
    assert [len(worker.w.tasks()[i::2]) for i in (0,1)]==[73,73]
    for t in worker.w.tasks():
        for row in worker.w.trajectory.source_rows(packet,t['order']):assert Path(row['image_path']).is_file()
    parent=Path(f.read(run/'private/CURRICULUM_ADMISSION.json')['evaluation_parent']);records={};scores={}
    with sqlite3.connect('file:'+str(parent/'private/judge_direction146_astra_medium/queue.sqlite')+'?mode=ro',uri=True) as db:
        for path,status,score in db.execute('select c.path,p.status,p.correct from consumer c join payload p on p.key=c.payload_key'):
            d=f.read(path)
            if d['method'] not in ('BASE','M13P'):continue
            k=tuple(d[x] for x in ('method','mode','prefix','owner'))+(d['binding']['input']['query_id'],)
            records[k]=d;scores[k]=score if status=='FORMAT_VALID' else None
            if d['method']=='M13P':records[('P160',)+k[1:]]=d;scores[('P160',)+k[1:]]=scores[k]
    result,_=report.calculate(f.read(run/'private/ORIGINAL146_TASKS.json'),f.read(run/'private/ORIGINAL_BASE_MASK146.json'),records,scores,methods=('P160','M13P'),single_only=True)
    gate=f.gate(result);assert not gate['passed'] and not gate['T2G_improves'] and all(gate[k] for k in ('FIT','T1G','T1L','T2L','damage','complete_no_missing'))
    changed=copy.deepcopy(result);changed['paired_native_FIT160_contrasts']['single_1']['T2G']['M13P_minus_P160_macro_pp']=1
    assert f.gate(changed)['passed']
    changed['paired_native_FIT160_contrasts']['single_1']['T2L']['M13P_preserves_each_P160_owner']=False;assert not f.gate(changed)['passed']
    receipt=dict(status='PASS',curriculum_steps=540,source_backwards=940,weights=weights,gradient_selection=True,projection_edges=True,source_admitted=True,report_identity=True,strict_improvement=True,protection_failure_rejected=True,model_calls=0,epoch=time.time())
    c.write(run/'private/CURRICULUM_CPU_CHECK.json',receipt);print(receipt)
if __name__=='__main__':main()
