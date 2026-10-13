"""Actual eight-expert and 24-expert mixed-W0 banks, fixed R0."""
import os,sys
from pathlib import Path
RUN=Path(os.environ['RUN_ROOT']);BASE=Path(os.environ['BASE_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
import core
common=core.common;worker=core.worker;train=core.train

def main():
    import torch
    slot=int(os.environ['PARTITION']);gpu=int(os.environ['GPU']);ts=[t for t in common.read(BASE/'private/QUEUES.json')['tasks'] if t['cohort']=='P2'];roles=common.read(BASE/'private/U_ROLES.json')
    with common.lease(gpu):
        runtime,bindings=common.load(gpu);teachers=train.teachers_for(runtime,sum(roles.values(),[]))
        for prefix in [8,24]:
            bank=[]
            for t in ts[:prefix]:bank+=worker.router(runtime,t)
            for arm in ['FROZEN_W0','CE_U_MULTI','U_ONLY','HOLD_U']:
                points={}
                for i,t in enumerate(ts[:prefix]):
                    if i>=8 or arm=='FROZEN_W0':p=worker.w0_path(t,0)
                    elif arm=='CE_U_MULTI' and slot==0:p=Path(common.read(BASE/'private/edits'/t['anonymous_edit']/'s0/COMPLETE.json')['points']['CE_U_MULTI'])
                    else:p=RUN/'private/edits'/t['anonymous_edit']/('s'+str(slot))/arm/'step160.pt'
                    assert p.is_file();points[t['edit_id']]=p
                worker.evaluate(runtime,bindings,ts[prefix-1],slot,arm,0 if arm=='FROZEN_W0' else 160,points,bank,ts[:prefix],mode='bank',prefix=prefix,teacher_rows=teachers)
                if arm=='FROZEN_W0':
                    subset=[d['binding']['input']['query_id'] for p in (RUN/'private/outputs/bank'/ts[prefix-1]['anonymous_edit']/('s'+str(slot))/arm/'n0'/('p'+str(prefix))/'R0').glob('*.json') if (d:=common.read(p))['effective_expert'] is not None and d['binding']['input'].get('role') in ('CAL','CHECK')]
                    common.write(RUN/'private'/('FIXED_ON_'+str(prefix)+'_'+str(slot)+'.json'),dict(query_ids=sorted(subset),frozen_before_candidate_generation=True,source='W0_R0'))
                print('BANK_COMPLETE',prefix,slot,arm,flush=True)
        common.write(RUN/'private'/('EVAL_PART_'+str(slot)+'.json'),dict(status='COMPLETE',true_eight_bank=True,partial_update_24_bank=True,background16='SAME_W0_ALL_ARMS'))
if __name__=='__main__':main()
