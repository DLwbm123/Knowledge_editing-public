"""Explicit selected-lane entry; no legacy four-method controller or admission."""
import os
import traceback
from pathlib import Path
import sys
import runtime146 as worker
sys.path.insert(0,str(Path(__file__).parent))
import selected_training146 as training


def main():
    training.require_admission();worker.r.NATIVE_FORWARD_CAP=training.comparison146.CAPS['non_generation_forwards']
    action=os.environ['ACTION']
    if action in ('selected_train','selected_recovery_train','selected_prefill_recovery_train'):training.train()
    elif action in ('m13_146_single','m13_146_bank'):
        worker.configure()
        from bank146 import evaluate
        evaluate(292,2,training.require_admission,'SELECTED_CANDIDATES_FROZEN1.json','SELECTED_SCOPE1.json',training.source_file())
    else:raise ValueError(action)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        worker.r.save_counts()
        worker.c.write(worker.RUN/'private'/f'FAILURE_selected_runtime_{os.environ.get("ACTION")}_{os.environ.get("PARTITION")}.json',
            dict(error=repr(error),traceback=traceback.format_exc(),counts=worker.r.COUNTS,automatic_retry=False))
        raise
