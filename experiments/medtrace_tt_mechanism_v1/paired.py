"""Read-only matched-query audit of frozen forced and natural CHECK outputs."""
import os
import sqlite3
from collections import Counter
from pathlib import Path
import direction as d

c,p,RUN=d.c,d.p,d.RUN
PARENT=Path(os.environ['MECHANISM_PARENT'])


def main():
    db=sqlite3.connect('file:'+str(PARENT/'private/judge_direction24_astra_medium/queue.sqlite')+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    scores={x['key']:x['correct'] for x in db.execute('SELECT * FROM payload')}
    mapping={}
    for row in db.execute("SELECT * FROM consumer WHERE mode IN ('forced_source_CHECK','natural_source_CHECK')"):
        x=c.read(row['path']);assert c.digest(x)==row['output_binding']
        mapping[row['method'],row['mode'],row['query_id']]=x,scores[row['payload_key']]
    routes=c.read(RUN/'private/ROUTES.json');tasks=p.tasks();chosen={t['edit_id'] for t in tasks[:24]};new={t['edit_id'] for t in tasks[8:24]}
    records=[]
    for row in c.read(RUN/'private/SOURCE_CHECK_ROWS.json'):
        q=row['query_id'];forced=tasks[row['forced_expert_index']]['edit_id'];natural=routes[q]['effective_expert'];values={};outputs={}
        for mode,suffix in [('forced',''),('natural','_NAT')]:
            for label in ('BASE','W0','RAW_DIR_540'):
                out,score=mapping[label+suffix,mode+'_source_CHECK',q];assert score in (0,1)
                values[mode+'_'+label]=score;outputs[mode,label]=out
        assert values['forced_BASE']==values['natural_BASE']
        binding=[outputs[m,a]['binding']['judge_input'] for m in ('forced','natural') for a in ('BASE','W0','RAW_DIR_540')]
        assert all(x==binding[0] for x in binding)
        same=forced==natural
        equal={a:outputs['forced',a]['R0']['raw_token_ids']==outputs['natural',a]['R0']['raw_token_ids'] for a in ('BASE','W0','RAW_DIR_540')}
        assert equal['BASE']
        # Same input, selected expert and frozen algorithm must produce the same tokens.
        if same:assert equal['W0'] and equal['RAW_DIR_540']
        records.append(dict(query_id=q,source_group=row['source_group'],kind=row['answer_kind'],forced_expert=forced,natural_expert=natural,
            same_expert=same,forced_updated=forced in chosen,natural_updated=natural in chosen,natural_new16=natural in new,
            both_updated=forced in chosen and natural in chosen,Base_correct=values['forced_BASE']==1,values=values,tokens_equal=equal))
    panels=[]
    for cohort,predicate in [('all96',lambda x:True),('natural_updated24',lambda x:x['natural_updated']),('natural_new16',lambda x:x['natural_new16']),('both_updated',lambda x:x['both_updated'])]:
        for stratum,match in [('all',lambda x:True),('same_expert',lambda x:x['same_expert']),('different_expert',lambda x:not x['same_expert']),('Base_correct',lambda x:x['Base_correct'])]+[(k,lambda x,k=k:x['kind']==k) for k in ('yes','no','open')]:
            rows=[x for x in records if predicate(x) and match(x)];n=len(rows)
            panels.append(dict(cohort=cohort,stratum=stratum,n=n,source_groups=len({x['source_group'] for x in rows}),same_expert=sum(x['same_expert'] for x in rows),
                correct={m+'_'+a:sum(x['values'][m+'_'+a] for x in rows) for m in ('forced','natural') for a in ('BASE','W0','RAW_DIR_540')},
                paired_gains={m:dict(Counter(x['values'][m+'_RAW_DIR_540']-x['values'][m+'_W0'] for x in rows)) for m in ('forced','natural')},
                cross_mode_RAW_tokens_equal=sum(x['tokens_equal']['RAW_DIR_540'] for x in rows)))
    assert len(records)==96 and sum(x['natural_new16'] for x in records)==21
    c.write(RUN/'private/PAIRED_QUERY_AUDIT.json',records)
    matches=sum(x['same_expert'] for x in records)
    result=dict(status='COMPLETE',queries=96,panels=panels,same_expert_matches=matches,same_expert_consistency='PASS' if matches else 'NO_MATCHED_EXPERT_SUPPORT',new_GPU=0,new_Judge=0,
        interpretation='Matched-query descriptive audit; diagnostic forced expert is not routing ground truth. Cross-expert differences do not identify a causal routing error.')
    c.write(RUN/'public/PAIRED_QUERY_AUDIT.json',result);print(result)


if __name__=='__main__':main()
