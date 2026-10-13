import os,runpy
os.environ['SCORING_PHASE']='qualification'
runpy.run_path(os.environ['RUN_ROOT']+'/private/tools/score.py',run_name='__main__')
