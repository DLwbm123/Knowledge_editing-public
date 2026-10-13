"""Use the same finite diagnostic at the canonical zero-residual TT state."""
import os
import time
import traceback
import torch
import probe as q

c, p, RUN = q.c, q.p, q.RUN
original_compare = q.compare


def zero_state(t):
    expert = p.tr.TT4(t['seed'], 8, 8)
    state = expert.state_dict()
    assert torch.count_nonzero(state['G1']) == 0
    return state


def compare(reference, candidate, target):
    assert len(target) >= 2
    result = original_compare(reference, candidate, target)
    result['content'] = original_compare(reference[:-1], candidate[:-1], target[:-1])
    result['EOS'] = original_compare(reference[-1:], candidate[-1:], target[-1:])
    return result


def selfcheck():
    target = torch.tensor([0, 1])
    before = torch.tensor([[4., 0.], [4., 0.]], dtype=torch.float64)
    after = torch.tensor([[3., 0.], [0., 4.]], dtype=torch.float64)
    result = compare(before, after, target)
    assert result['content']['all_argmax_correct'] and result['EOS']['all_argmax_correct']
    assert result['content']['NLL_change'] > 0 and result['EOS']['NLL_change'] < 0
    assert abs(result['NLL_change'] - .5*(result['content']['NLL_change']+result['EOS']['NLL_change'])) < 1e-12


q.d.start_state = zero_state
q.compare = compare
q.VERIFY_ZERO_BASE = True


def plan():
    selfcheck()
    q.plan()
    lock = c.read(RUN/'private/PROBE_LOCK.json')
    lock.update(initial_state='ORIGINAL_SEEDED_ZERO_TT88', forward_calls=4832,
        zero_expert_Base_checks=808, full_and_content_EOS_metrics=True,
        extra_gate='Base_content_correct_at_least_8_sources_and_no_extra_content_token_losses')
    c.write(RUN/'private/PROBE_LOCK.json', lock)
    admission = c.read(RUN/'public/ADMISSION.json')
    admission.update({k:v for k,v in lock.items() if k not in ('split_binding','code')})
    admission['content_EOS_selfcheck'] = 'PASS'
    c.write(RUN/'public/ADMISSION.json', admission)


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('initial_probe.py','initial_response_worker',g,i) for i,g in enumerate(q.GPUS)])
    pipeline.wait([pipeline.launch('initial_report.py','initial_response_report')])


if __name__ == '__main__':
    try:
        {'initial_response_plan':plan,'initial_response_worker':q.worker,
         'initial_response_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
