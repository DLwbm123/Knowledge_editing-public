"""Separate content from termination while retaining the previous joint gate."""
import probe_report as report


def qualify(result, data):
    assert sum(x['zero_expert_Base_checks'] for x in data) == 808
    held = [r for x in data for r in x['diagnostics'] if r['role'] == 'HELDOUT_FIT']
    panels = {}
    for section in ('content', 'EOS'):
        correct = [r for r in held if r['baseline'][section]['all_argmax_correct']]
        panels[section] = dict(initial_correct_observations=len(correct),
            initial_correct_sources=len({r['group'] for r in correct}),
            all_arms={a:dict(KL=report.mean([r['candidates'][a][section]['KL'] for r in held]),
                NLL_change=report.mean([r['candidates'][a][section]['NLL_change'] for r in held]),
                lost_correct_tokens=sum(r['candidates'][a][section]['lost_correct_tokens'] for r in held)) for a in report.ARMS},
            initially_correct_arms={a:dict(KL=report.mean([r['candidates'][a][section]['KL'] for r in correct]),
                lost_correct_tokens=sum(r['candidates'][a][section]['lost_correct_tokens'] for r in correct)) for a in report.ARMS} if correct else None)
    content = panels['content']
    qualified = content['initial_correct_sources'] >= 8
    no_extra_loss = content['all_arms']['PROJECTED']['lost_correct_tokens'] <= content['all_arms']['MATCHED_RAW']['lost_correct_tokens']
    previous_gate = result['decision']
    if previous_gate == 'LOCAL_MECHANISM_SIGNAL':
        if not qualified: result['decision'] = 'INCONCLUSIVE_CORRECT_CONTENT_SUPPORT'
        elif not no_extra_loss: result['decision'] = 'NO_LOCAL_SUPPORT_CONTENT_LOSS'
    result.update(initial_state='ORIGINAL_SEEDED_ZERO_TT88', previous_joint_gate=previous_gate,
        content_EOS_panels=panels, correct_content_support_gate=qualified,
        no_extra_content_loss_gate=no_extra_loss, zero_expert_Base_checks=808)
    return result


if __name__ == '__main__': report.main(expected_forwards=4832, transform=qualify)
