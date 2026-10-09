"""Correct-answer holdout is fixed by previous Base judging, before the candidates."""
import probe_report as report


def qualify(result,data):
    assert sum(x['zero_expert_Base_checks'] for x in data)==808
    primary=[r for x in data for r in x['diagnostics'] if r['role']=='HELDOUT_FIT' and r['semantic_correct']]
    assert len(primary)==8*63 and len({r['group'] for r in primary})==29
    lost={a:sum(r['candidates'][a]['lost_correct_tokens'] for r in primary) for a in report.ARMS}
    prior=result['decision'];no_extra=lost['PROJECTED']<=lost['MATCHED_RAW']
    if prior=='LOCAL_MECHANISM_SIGNAL' and not no_extra:result['decision']='NO_LOCAL_SUPPORT_RESPONSE_TOKEN_LOSS'
    result.update(previous_joint_gate=prior,primary_definition='PREVIOUSLY_JUDGED_CORRECT_BASE_ANSWERS',
        primary_lost_correct_response_tokens=lost,no_extra_response_token_loss_gate=no_extra,
        basis_correct_questions=61,basis_correct_sources=31,zero_expert_Base_checks=808)
    return result


def selfcheck():
    rows=[dict(role='HELDOUT_FIT',semantic_correct=True,group=i%29,
        candidates={a:dict(lost_correct_tokens=0) for a in report.ARMS}) for i in range(8*63)]
    data=[dict(zero_expert_Base_checks=808,diagnostics=rows)]
    assert qualify(dict(decision='LOCAL_MECHANISM_SIGNAL'),data)['decision']=='LOCAL_MECHANISM_SIGNAL'
    rows[0]['candidates']['PROJECTED']['lost_correct_tokens']=1
    assert qualify(dict(decision='LOCAL_MECHANISM_SIGNAL'),data)['decision']=='NO_LOCAL_SUPPORT_RESPONSE_TOKEN_LOSS'
    assert qualify(dict(decision='NO_LOCAL_SUPPORT'),data)['decision']=='NO_LOCAL_SUPPORT'


if __name__=='__main__':
    selfcheck()
    report.main(expected_forwards=4552,expected_backwards=504,primary_questions=63,
                primary_filter=lambda row:row['semantic_correct'],transform=qualify)
