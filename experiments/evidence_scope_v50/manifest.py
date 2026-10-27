"""Separation of writer supervision, routing scope and evaluation labels."""
REQUIRED={'case','source_cluster','edit_id','split','role','image','question','answer_if_authorized','edit_scope','scope_label','relation','evidence_type','evidence_provenance','reference_status','writer_support_role','routing_scope_role'}

def validate(rows):
    ids=set();splits={};train=set();held=set()
    for row in rows:
        if not REQUIRED<=row.keys():raise ValueError('Missing manifest fields')
        key=(row['edit_id'],row['query_id']);assert key not in ids;ids.add(key)
        splits.setdefault(row['source_cluster'],set()).add(row['split'])
        if row['writer_support_role']!='NONE':
            assert row['role']=='SOURCE' and row['answer_if_authorized'] is not None
            train.add((row['image'],row['question']))
        if row['role']=='HELDOUT':
            assert row['writer_support_role']=='NONE' and row['routing_scope_role']=='EVALUATION_ONLY'
            held.add((row['image'],row['question']))
        if row['evidence_type']=='BLACK_PROXY':
            assert row['scope_label']=='UNKNOWN' and row['answer_if_authorized'] is None
        if row['role']!='SOURCE':assert row['answer_if_authorized'] is None
    assert not train&held
    assert all(not ('CONFIRMATION' in v and len(v)>1) for v in splits.values())
    return dict(status='PASS',rows=len(rows),source_clusters=len(splits),training_leakage=False)


def query_input(row):
    # No label, role, answer length, or reference can enter the prediction boundary.
    return {'image':row['image'],'question':row['question']}
