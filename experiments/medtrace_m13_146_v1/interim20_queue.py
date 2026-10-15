"""Original sealed Astra queue with fixed first20 coverage and a new epoch."""
from types import SimpleNamespace
import contract
import json
import os
from pathlib import Path
import interim20 as protocol
import queue146 as base

base.contract=SimpleNamespace(original=contract.original,METHOD_CONSUMERS=protocol.METHOD_CONSUMERS)
base.MAX_JUDGE=protocol.CONSUMERS
base.EPOCH=protocol.EPOCH
base.QUEUE_FOLDER=protocol.QUEUE
base.BATCH_LEDGER='Astra_interim20_batches'
base.ADMISSION_FILE='INTERIM20_ADMISSION1.json'
base.SCOPE_FILE='SELECTED_SCOPE1.json'
base.GENERATION_FILE='INTERIM20_GENERATION_COMPLETE1.json'

if __name__=='__main__':
    failed=list((Path(os.environ['RUN_ROOT'])/'private').glob('FAILURE_interim20*.json'))
    operation=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"status"}'))
    if failed and operation['action'] in ('next','reserve'):
        assert operation['action']!='reserve','Own stage failed; no new attempt'
        print(json.dumps(dict(state=dict(status='STOPPED',reason='OWN_STAGE_FAILED'),rows=[])))
    else:base.main()
