"""One frozen HE arm; shared budget translation for installation and CPU checks."""

VERSION = 'PUREW_HINGE_EVIDENCE_FIRST1_V1_20261016'
RUN_ROOT = '/data/bmw/Knowledge_editing/outputs/purew-hinge-evidence-first1-v1-20261016/run'
SCOPE_ROOT = '/data/bmw/Knowledge_editing/outputs/purew-objective-scope-v1-20261015/run'
PROJECTION_ROOT = '/data/bmw/Knowledge_editing/outputs/purew-projected-objective-first1-v1-20261015/run'
HINGE_ROOT = '/data/bmw/Knowledge_editing/outputs/purew-odds-hinge-first1-v1-20261016/run'


def effective_protocol(p):
    assert p['version']==VERSION and p['design']['new_arms']==p['design']['order']==['HE']
    assert p['design']['source_orders']==[1] and p['design']['slots_per_arm']==20
    objective=p['objective']
    assert objective['hinge_weight']==objective['evidence_weight']==1
    assert objective['removed_KL_weight']==0 and objective['evidence_target_ratio']==1.01
    assert objective['contrast_margin_loss'] and objective['recipe']=='ODDS_HINGE_ALPHA_095_EVIDENCE_LOG101_WEIGHT1_V1'
    assert objective['conditional_KL_weight']==objective['language_prefix_KL_weight']==0
    assert not any(objective[k] for k in ('hard_removed','probability_floor','coefficient_sweep','stop_gradient_full','stop_gradient_removed'))
    assert p['shared']['HARD_KL_NOT_ENFORCED'] and not p['shared']['fallback_active_set']
    caps=dict(GPU_process_hours=2,wall_hours=24,formal_slots=20,native_writes_including_admission=21,
              backwards=300,forwards_including_vision=15000,generations=92,new_Judge=87,
              candidate_checks=160,active_set_checks=0)
    assert all(p['budgets'][k]==v for k,v in caps.items())
    assert p['evaluation']['new_total']==87 and p['evaluation']['total_with_all_six_historical']==609
    assert p['gates']['complete_CE_strict_below']==.1 and p['gates']['allowed_individual_damage']==0
    assert p['gates']['all_ten_evidence_gaps_strict_positive_finite'] and p['gates']['unknown_fails']
    assert p['gates']['HE_T2G_per_edit_delta_lower_vs']==['HM','SH00']
    budget=dict(p['budgets'],minimum_free_bytes=8*1024**3,owned_matrix_limit_bytes=96*1024**3,
                maximum_formal_attempt_slots=20,maximum_committed_updates=21,
                maximum_forward_calls=15000,maximum_backwards=300,maximum_generations=92,
                maximum_new_Judge=87,largest_factor_FP64_bytes=38*58720256*8)
    return dict(p,budgets=budget,active_arms=['HE'],execution_revision='HINGE_EVIDENCE_FIRST1_V1',
                reference_run=SCOPE_ROOT,projection_reference_run=PROJECTION_ROOT,
                hinge_reference_run=HINGE_ROOT,
                K_reference_run='/data/bmw/Knowledge_editing/outputs/purew-protect100-v1-20261013/run')


def require_execution(p,authorization,hardware):
    effective_protocol(p)
    assert authorization.get('protocol_version')==VERSION
    assert authorization.get('active_arms')==['HE']
    assert all(authorization.get(k) is True for k in ('execution_authorized','install','model_calls','Judge')), 'HE is preparation-only; explicit execution authorization required'
    assert hardware.get('allocation_authorized') is True
    assert type(hardware.get('physical_GPU')) is int and 0<=hardware['physical_GPU']<=7
    assert isinstance(hardware.get('GPU_UUID'),str) and hardware['GPU_UUID'].startswith('GPU-')


def check_history(p):
    import json
    from pathlib import Path
    parent=Path(SCOPE_ROOT)
    hinge=Path(HINGE_ROOT);projection=Path(PROJECTION_ROOT)
    assert json.loads((hinge/'private/GPU_SOURCE_VERSION.json').read_text())['commit']==p['reference']['runtime_commit']
    hinge_protocol=json.loads((hinge/'public/PROTOCOL.json').read_text())
    assert hinge_protocol['shared']==p['shared'] and hinge_protocol['objective']['alpha']==.95
    assert hinge_protocol['reference']['runtime_commit']==json.loads((parent/'private/GPU_SOURCE_VERSION.json').read_text())['commit']
    assert hinge_protocol['historical_projection']['runtime_commit']==json.loads((projection/'private/GPU_SOURCE_VERSION.json').read_text())['commit']
    base_task=json.loads((parent/'private/BENCHMARK146_QUEUE.json').read_text())['tasks'][0]
    for history,arms in ((parent,('SH00','SH10')),(projection,('PM','PU')),(hinge,('HM','HU'))):
        assert json.loads((history/'private/BENCHMARK146_QUEUE.json').read_text())['tasks'][0]==base_task
        for name in ('source.json','review.json'):
            assert json.loads((history/'private/operator'/name).read_text())['rows'][0]==json.loads((parent/'private/operator'/name).read_text())['rows'][0]
        prior=json.loads((history/'public/RESULTS.json').read_text())
        for arm in arms:
            assert prior['coverage'][arm]=={'FORMAT_VALID':87}
            assert json.loads((history/'public'/('NATIVE_RELOAD_'+arm+'.json')).read_text())['status']=='PASS'
            assert (history/'private'/('FINAL_CONSUMED_'+arm+'.json')).exists()
        assert (history/'private/REPORT_COMPLETE.json').exists()
    return dict(status='PASS',source_seed_queue_exact=True,shared_optimizer_Fisher_protocol_exact=True,alpha_exact=True,old_arms=['SH00','SH10','PM','PU','HM','HU'],no_rerun=True)
