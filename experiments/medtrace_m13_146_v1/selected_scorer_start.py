"""Explicit four-worker entry for the selected two-method isolated queue."""
import scorer_start

if __name__=='__main__':
    scorer_start.start(queue_file='selected_queue146.py',result_folder='judge_native146_two_methods_astra_medium',
        epoch='MEDTRACE_NATIVE146_TWO_METHODS_20261015_V1',admission_file='SELECTED_RUNTIME_ADMISSION1.json')
