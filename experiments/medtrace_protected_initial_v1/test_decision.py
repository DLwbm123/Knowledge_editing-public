"""A positive response-only result cannot bypass correctness support or loss gates."""
from initial_report import qualify


def result(groups, lost=0, original='LOCAL_MECHANISM_SIGNAL'):
    rows = []
    for group in range(groups):
        values = {a:{section:dict(KL=0., NLL_change=0., lost_correct_tokens=lost if a=='PROJECTED' else 0)
                     for section in ('content','EOS')} for a in ('RAW','PROJECTED','MATCHED_RAW')}
        rows.append(dict(role='HELDOUT_FIT',group=group,
            baseline={section:dict(all_argmax_correct=True) for section in ('content','EOS')},candidates=values))
    return qualify(dict(decision=original), [dict(zero_expert_Base_checks=808,diagnostics=rows)])['decision']


def main():
    assert result(8) == 'LOCAL_MECHANISM_SIGNAL'
    assert result(7) == 'INCONCLUSIVE_CORRECT_CONTENT_SUPPORT'
    assert result(8, lost=1) == 'NO_LOCAL_SUPPORT_CONTENT_LOSS'
    assert result(8, original='NO_LOCAL_SUPPORT') == 'NO_LOCAL_SUPPORT'
    print('PASS: support, additional loss, and prior joint-gate boundaries')


if __name__ == '__main__':main()
