"""Read frozen V61 outputs; write paired diagnostics to a separate destination."""
import argparse, collections, json
from pathlib import Path

def main(root, private, public):
    def read(name): return json.loads((root/name).read_text())
    protected = {n:(root/n).read_bytes() for n in ['BLIND_VERDICTS.json','VERDICTS_FROZEN.json','SUMMARY.json']}
    verdicts, packet = read('BLIND_VERDICTS.json'), {x['id']:x for x in read('BLIND_PACKET.json')}
    mapping = {(x['arm'],x['owner'],x['query']):x['blind_id'] for x in read('BLIND_MAPPING.json')}
    old = read('EXACT_LABEL_REUSE.json')
    arms = ['ALL_TOKENS','PREFILL_ONLY','VISUAL_ONLY']
    rows=[]
    for task in read('DATA_ADMISSION.json')['tasks']:
        for r in task['eval_rows']:
            item=dict(owner=task['order'],query=r['query_id'],roles=r['roles'],question=r['question'],reference=r['reference'],source_reference=task['source']['reference'],arms={})
            for i,a in enumerate(arms):
                key=mapping[(a,task['order'],r['query_id'])]
                out=json.loads((root/str(i)/task['key']/'outputs'/(r['query_id']+'.json')).read_text())
                assert out['raw_answer']==packet[key]['answer']
                item['arms'][a]=dict(blind_id=key,answer=out['raw_answer'],verdict=verdicts[key],label_origin='historical_exact_reuse' if key in old else 'V61_new_blind',EOS=out['EOS'],at_cap=out['at_cap'],generated_tokens=len(out['raw_token_ids']))
            rows.append(item)
    assert len(rows)==219
    def lab(r,a):return r['arms'][a]['verdict']['label']
    roles=sorted({role for r in rows for role in r['roles']})
    stats={}
    for role in roles:
        rs=[r for r in rows if role in r['roles']];stats[role]={}
        for a in arms[1:]:
            tr=collections.Counter(lab(r,arms[0])+'->'+lab(r,a) for r in rs)
            per={}
            for owner in sorted({r['owner'] for r in rs}):
                own=[r for r in rs if r['owner']==owner]
                per[owner]=dict(lost=sum(lab(r,arms[0])=='CORRECT' and lab(r,a)!='CORRECT' for r in own),gained=sum(lab(r,arms[0])!='CORRECT' and lab(r,a)=='CORRECT' for r in own))
            stats[role][a]=dict(n=len(rs),transitions=dict(tr),lost=sum(x['lost'] for x in per.values()),gained=sum(x['gained'] for x in per.values()),per_edit=per)
    t2=[r for r in rows if 'T2G' in r['roles']]
    a7=[r for r in t2 if lab(r,arms[0])!='CORRECT']
    changes=[r for r in rows if any(lab(r,a)!=lab(r,arms[0]) for a in arms[1:])]
    textchanges=[r for r in rows if any(r['arms'][a]['answer']!=r['arms'][arms[0]]['answer'] for a in arms[1:])]
    assert len(a7)==7 and collections.Counter(lab(r,arms[0]) for r in a7)=={'INCORRECT':2,'UNCERTAIN':4,'ABSTAIN':1}
    assert (stats['T2G'][arms[1]]['lost'],stats['T2G'][arms[1]]['gained'])==(9,1)
    assert (stats['T2G'][arms[2]]['lost'],stats['T2G'][arms[2]]['gained'])==(4,4)
    summary=dict(status='READ_ONLY_ANALYSIS_COMPLETE',frozen_scores_unchanged=True,new_model_calls=0,training_runs=0,consumers=len(rows),label_changed_consumers=len(changes),text_changed_consumers=len(textchanges),T2G_reference_equals_source=sum(r['reference']==r['source_reference'] for r in t2),A_T2G_noncorrect_by_edit=dict(collections.Counter(r['owner'] for r in a7)),paired=stats)
    private.mkdir(parents=True,exist_ok=True);public.mkdir(parents=True,exist_ok=True)
    for name,data in [('ALL_PAIRED_OUTPUTS.json',rows),('A_SEVEN.json',a7),('LABEL_CHANGED.json',changes)]:
        (private/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (public/'PAIRED_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# V61 私有逐题配对材料','仅派生分析；原评分不变。含完整原回答，限本地合法范围使用。','A=所有token，B=前缀，C=视觉。正文列出A七题及所有标签变化；ALL_PAIRED_OUTPUTS.json另保留全部219消费者，包括仅措辞变化。']
    seen=set()
    for r in a7+changes:
        k=(r['owner'],r['query'])
        if k in seen:continue
        seen.add(k);lines += [f"\n## E{r['owner']:03d} / {r['query']} / {','.join(r['roles'])}",f"问题：{r['question']}",f"参考：{r['reference']}"]
        for short,a in zip('ABC',arms):
            x=r['arms'][a];lines += [f"\n### {short}: {x['verdict']['label']} ({x['label_origin']}; EOS={x['EOS']}; cap={x['at_cap']})",x['answer'] or '[空回答]',f"原判理由：{x['verdict']['reason']}"]
    (private/'CASEBOOK.md').write_text('\n\n'.join(lines)+'\n')
    assert all((root/n).read_bytes()==b for n,b in protected.items()),'Frozen evidence changed'
    print(json.dumps({k:v for k,v in summary.items() if k!='paired'},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('private',type=Path);p.add_argument('public',type=Path);a=p.parse_args();main(a.root,a.private,a.public)
