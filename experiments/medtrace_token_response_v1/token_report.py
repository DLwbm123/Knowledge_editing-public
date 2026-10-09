"""Apply the identical qualified-response joint gate to all-token constraints."""
import qualified_report as prior


def qualify(result,data):
    result=prior.qualify(result,data)
    assert all(d['basis_audit']['rows']==1514 and d['basis_audit']['mean_gradient_checks']==61 for d in data)
    result.update(basis_token_constraints=1514,mean_gradient_parity_checks=488,
        maximum_mean_gradient_relative_error=max(d['basis_audit']['maximum_mean_gradient_relative_error'] for d in data))
    return result


if __name__=='__main__':
    prior.selfcheck()
    prior.report.main(expected_forwards=4552,expected_backwards=12616,primary_questions=63,
        primary_filter=lambda row:row['semantic_correct'],transform=qualify)
