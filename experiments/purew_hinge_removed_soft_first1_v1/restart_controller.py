"""One authorized pre-slot restart; reuse admission and retain every prior cost."""
import traceback,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import controller
c,R=controller.c,controller.R
original_stage=controller.stage

def stage(action,arm,owner,phase=None):
    if action.startswith(('eval_base_','admit_')):
        kind='BASE_IDENTITY' if action.startswith('eval_base_') else 'REAL_VLM_ADMISSION'
        assert c.read(R/'public'/(kind+'_'+arm+'.json'))['status']=='PASS'
        return
    if action=='learn_HR_1':action+='_restart1'
    original_stage(action,arm,owner,phase)

def main():
    receipt=c.read(R/'public/RESTART1_RECEIPT.json')
    assert receipt['restart_number']==1 and receipt['formal_slots_before']==0
    assert c.read(R/'RUN_MANIFEST.json')==receipt['original_manifest']
    assert not (R/'private/START_learn_HR_1_restart1.json').exists()
    c.budget();controller.stage=stage;controller.main()

if __name__=='__main__':
    try:main()
    except BaseException:
        c.write(R/'private/failures/controller.json',dict(epoch=time.time(),traceback=traceback.format_exc(),restart_number=1));raise
