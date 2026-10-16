"""Selected M13/Fmin queue; preserve the sealed scorer and one-attempt policy."""
import comparison146
import queue146 as base

base.contract=comparison146
base.MAX_JUDGE=comparison146.CONSUMERS
base.EPOCH='MEDTRACE_NATIVE146_TWO_METHODS_20261015_V1'
base.QUEUE_FOLDER='judge_native146_two_methods_astra_medium'
base.BATCH_LEDGER='Astra_native146_two_methods_batches'
base.ADMISSION_FILE='SELECTED_RUNTIME_ADMISSION1.json'
base.SCOPE_FILE='SELECTED_SCOPE1.json'
base.GENERATION_FILE='SELECTED_GENERATION_COMPLETE1.json'

EPOCH=base.EPOCH
contract=comparison146
ingest_generated=base.ingest_generated
known_scores=base.known_scores
coverage=base.coverage
request=base.request

if __name__=='__main__':base.main()
