"""Join only prebound fixed-panel decisions after candidate scoring ends.

The caller must independently validate prediction, runtime and score contracts.
Only aggregate results may be published; joined rows remain private.
"""
import metrics


def join(base_rows, roles, old_consumers, old_decisions, new_packet, new_decisions, unknown_keys, scoring_receipt):
    if scoring_receipt.get('status') not in ('COMPLETE','INCOMPLETE'):
        raise ValueError('Candidate scoring has not terminated')
    if new_packet.get('status') != 'SCORE_PACKET_FROZEN' or new_packet.get('source_events') != 508:
        raise ValueError('Candidate packet is not frozen')
    unknown_keys=set(unknown_keys)
    if set(new_decisions)&unknown_keys or set(new_decisions)|unknown_keys != set(new_packet['payloads']):
        raise ValueError('Candidate decision and unknown coverage mismatch')
    if scoring_receipt.get('unknown') != len(unknown_keys) or (scoring_receipt['status']=='COMPLETE') != (not unknown_keys):
        raise ValueError('Scoring completion status mismatch')
    base = {(r['owner'],r['query_id']):r for r in base_rows}
    if len(base_rows)!=508 or len(base)!=508 or len(roles)!=16:
        raise ValueError('Fixed panel coverage mismatch')
    role_map = {}
    for owner, groups in roles.items():
        if set(groups)!=set(metrics.ROLES):
            raise ValueError('Role keys mismatch')
        for role, qids in groups.items():
            for qid in qids:
                key=(int(owner),qid)
                if key in role_map or key not in base:
                    raise ValueError('Ambiguous or unbound eligible event')
                role_map[key]=role
    old = {}
    for row in old_consumers:
        if row['arm']!='CDEEP':
            continue
        key=(row['owner'],row['query_id'])
        if key in old or key not in base or row['key'] not in old_decisions:
            raise ValueError('Original deep score coverage mismatch')
        old[key]=old_decisions[row['key']]
    new = {}
    for row in new_packet['consumers']:
        key=(row['owner'],row['query_id'])
        if row['arm']!='SOURCE_TEACHER' or key in new or key not in base or row['key'] not in new_packet['payloads']:
            raise ValueError('Candidate score coverage mismatch')
        new[key]=new_decisions.get(row['key'])
    if len(new_packet['consumers'])!=508 or set(old)!=set(base) or set(new)!=set(base):
        raise ValueError('Four-arm event binding incomplete')
    def fields(x):
        return {k:(x.get(k) if x is not None else None) for k in ('core_correct','full_supported')}
    return [dict(owner=o,query_id=q,role=role_map.get((o,q)),scores={
        'BASE':fields(row['BASE']),'MED':fields(row['MED']),
        'CDEEP':fields(old[(o,q)]),'SOURCE_TEACHER':fields(new[(o,q)])})
        for (o,q),row in base.items()]
