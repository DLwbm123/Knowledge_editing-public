"""Use the validated queue with the fixed weaker-U phase identity."""
import os
import runpy
os.environ["JUDGE_PHASE"]="WEAK24"
if __name__=="__main__":
    runpy.run_path(os.environ["RUN_ROOT"]+"/private/tools/combo24_queue.py",run_name="__main__")
