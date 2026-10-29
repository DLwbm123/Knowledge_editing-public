import role_admission

metadata = dict(evaluation=[dict(owner=(47, 117, 93, 87)[i % 4], group=f'e{i}') for i in range(69)],
                mapped_protection=[dict(group=f'm{i}') for i in range(31)],
                fallback_protection=[dict(group=f'f{i}') for i in range(30)])
deny = dict(V95=['protected95'], V111=['protected111'])
assert role_admission.admit(metadata, deny)['status'] == 'IDENTITY_CHECK_PASS'
for broken in ({'V95': ['e0'], 'V111': ['protected111']}, {'V95': [], 'V111': ['protected111']}):
    try:
        role_admission.admit(metadata, broken)
    except ValueError:
        pass
    else:
        raise AssertionError('incomplete or conflicting role data was accepted')
print('SYNTHETIC_ROLE_PASS')
