"""Strict role boundaries, using metadata only."""
def validate_roles(roles,banned):
    used=set()
    for role,rows in roles.items():
        assert role in ('FIT','CAL','CHECK')
        for row in rows:
            assert row['role']==role and row['source_role']=='train'
            group=(row['dataset'],row['image_id'])
            assert group not in banned and group not in used,'Source overlap or excluded source'
            assert row.get('protection_scope') and row.get('prior_role')=='NO_ROLE_FOUND_IN_AUDITED_INVENTORIES'
            used.add(group)
    return True
