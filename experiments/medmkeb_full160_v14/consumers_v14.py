"""Complete128 hidden review, paired accounting and exact dependency retirement."""
import json,random,collections,time
from pathlib import Path
from answer_v14 import check
ARMS=('SOURCE','RESERVE')
read=lambda p:json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def blind(private):
    old={};base={}
    for name in ('medmkeb_guard_v6_20261019','medmkeb_answer_v12_20261019','medmkeb_source31_v13_20261019'):
        previous=private.parent/name;packet={x['id']:x for x in read(previous/'BLIND_PACKET.json')};labels={x['id']:x['label'] for x in read(previous/'BLIND_VERDICTS.json')['rows']}
        for m in read(previous/'BLIND_MAPPING.json'):
            x=packet[m['id']];identity=(m['query_id'],x['question'],x['reference'],x['answer']);label=labels[m['id']];assert identity not in old or old[identity]==label;old[identity]=label
            if m.get('arm')=='BASE':assert m['query_id'] not in base or base[m['query_id']]==label;base[m['query_id']]=label
    items={};mapping=[];tasks=read(private/'DATA_ADMISSION.json')['tasks'];rows=read(private/'EVAL_ROWS.json')
    for i,t in enumerate(tasks):
        root=private/'remote'/str(i);assert read(root/'EXIT.json')['code']==0 and read(root/'STUDY_COMPLETE.json')['answers']==64
        bindings={x['query_id']:x for x in rows+[t['rephrase']]};assert len(bindings)==32
        for arm in ARMS:
            folder=root/arm/'outputs';assert check(folder,bindings)==32
            for q in sorted(bindings):
                v=read(folder/(q+'.json'));r=bindings[q];identity=(q,r['question'],r['reference'],v['raw_answer']);items.setdefault(identity,dict(question=r['question'],reference=r['reference'],answer=v['raw_answer']));mapping.append(dict(task=t['key'],arm=arm,query_id=q,identity=identity,EOS=v['EOS'],at_cap=v['at_cap']))
    order=list(items);random.Random(20261019).shuffle(order);ids={k:'R'+str(i+1).zfill(4) for i,k in enumerate(order)};packet=[dict(id=ids[k],**items[k]) for k in order];reused=[dict(id=ids[k],label=old[k],source='EXACT_FROZEN_IDENTITY') for k in order if k in old]
    for m in mapping:m['id']=ids[m.pop('identity')]
    assert len(mapping)==128
    for name,value in [('BLIND_PACKET.json',packet),('BLIND_MAPPING.json',mapping),('BASE_LABELS_FROZEN.json',base),('BLIND_VERDICTS_REUSED.json',dict(rows=reused)),('BLIND_PENDING.json',[x for x in packet if x['id'] not in {v['id'] for v in reused}])]:write(private/name,value)
    return dict(consumers=128,unique_reviews=len(packet),reused=len(reused),pending=len(packet)-len(reused))

def summarize(private):
    packet=read(private/'BLIND_PACKET.json');verdict=read(private/'BLIND_VERDICTS.json')['rows'];labels={x['id']:x['label'] for x in verdict};assert len(labels)==len(verdict)==len(packet) and set(labels)=={x['id'] for x in packet};assert set(labels.values())<={'CORRECT','INCORRECT','UNCERTAIN','CONTRADICTORY'}
    mapping=read(private/'BLIND_MAPPING.json');assert len(mapping)==128
    tasks=read(private/'DATA_ADMISSION.json')['tasks'];evals=[c['row']['query_id'] for c in read(private/'CANDIDATES_FROZEN.json')['candidates'] if c['split']=='eval'];assert len(evals)==27
    base=read(private/'BASE_LABELS_FROZEN.json');correct=[q for q in evals if base[q]=='CORRECT'];assert len(correct)==11
    results=[];exits=[];arms={}
    for i,t in enumerate(tasks):
        root=private/'remote'/str(i);e=read(root/'EXIT.json');assert e['code']==0 and read(root/'STUDY_COMPLETE.json')['answers']==64;exits.append(e)
        assert read(root/'RECONSTRUCTION_COMPLETE.json')['status']=='PASS' and read(root/'SOURCE_RESUME_ADMISSION.json')['status']=='PASS'
        for arm in ARMS:
            train=read(root/arm/'TRAIN_COMPLETE.json');curve=train['curve'];n=train['accepted_steps'];assert train['save_load_exact'] and len(curve)==n and [x['step'] for x in curve]==list(range(1,n+1)) and train['remaining_NA']==160-n
            assert n==160 if arm=='SOURCE' else 31<=n<=160
            if arm=='RESERVE':assert train['all_actual_outer_constraints_passed'] and all(x['max_support_regression']<=1e-4 for x in curve)
            values={x['query_id']:labels[x['id']] for x in mapping if x['task']==t['key'] and x['arm']==arm};expected={x['query_id'] for x in read(private/'EVAL_ROWS.json')}|{t['rephrase']['query_id']};assert set(values)==expected and len(values)==32
            arms[t['key'],arm]=values;source=t['source']['query_id'];cross=source.removesuffix('_native')+'_cross'
            results.append(dict(edit=i+1,arm=arm,accepted_steps=n,remaining_NA=160-n,source_gain=train['source_gain'],source_correct=values[source]=='CORRECT',cross_correct=values[cross]=='CORRECT',rephrase_correct=values[t['rephrase']['query_id']]=='CORRECT',base_correct_denominator=11,damage=sum(values[q]!='CORRECT' for q in correct),damage_to_uncertain=sum(values[q]=='UNCERTAIN' for q in correct),heldout_four_category=dict(collections.Counter(values[q] for q in evals))))
    counts={k:sum(x['counts'].get(k,0) for x in exits) for k in set().union(*(x['counts'] for x in exits))};assert counts['generations']==128 and counts['forwards']<=270000 and counts['backwards']<=18000 and counts['updates']<=578 and counts['candidate_checks']<=2400 and counts['reconstruction_updates']==62 and counts['reused_gradient_sequences']==110
    summary={a:dict(source_correct=sum(x['source_correct'] for x in results if x['arm']==a),cross_correct=sum(x['cross_correct'] for x in results if x['arm']==a),rephrase_correct=sum(x['rephrase_correct'] for x in results if x['arm']==a),damage=sum(x['damage'] for x in results if x['arm']==a),paired_checks=22) for a in ARMS}
    recovered=extra=0
    for t in tasks:
        control=arms[t['key'],'SOURCE'];sem=arms[t['key'],'RESERVE']
        recovered+=sum(control[q]!='CORRECT' and sem[q]=='CORRECT' for q in correct);extra+=sum(control[q]=='CORRECT' and sem[q]!='CORRECT' for q in correct)
    source_gate=all(x['source_correct'] for x in results);full=all(x['accepted_steps']==160 for x in results);control_damage=summary['SOURCE']['damage']>0
    return dict(status='COMPLETE_FULL160_PAIRED_CONSUMERS',tasks=results,arms=summary,base_correct_question_count=11,paired_checks_per_arm=22,recovered_by_RESERVE=recovered,extra_damage_by_RESERVE=extra,both_source_gates_passed=source_gate,all_160_complete=full,development_qualification=source_gate and full,control_has_damage=control_damage,protection_benefit_identified=source_gate and full and control_damage and recovered>extra and extra==0,independent_confirmation=False,EOS=sum(x['EOS'] for x in mapping),at_cap=sum(x['at_cap'] for x in mapping),counts=counts,gpu_process_hours=sum(x['seconds'] for x in exits)/3600)

def retire(root,v13):
    root=Path(root).resolve();v13=Path(v13).resolve();assert root.name==v13.name=='admission1' and 'medmkeb-full160-v14-' in str(root) and 'medmkeb-source31-v13-' in str(v13)
    assert read(root/'CONSUMERS_COMPLETE.json')['status']=='COMPLETE_FULL160_PAIRED_CONSUMERS';paths=[]
    for i in (0,1):
        folder=root/str(i);launch=read(folder/'LAUNCH.json');proc=Path('/proc')/str(launch['pid'])/'stat';assert not proc.exists() or proc.read_text().split()[21]!=str(launch['start_ticks']);assert read(folder/'EXIT.json')['code']==0
        for arm in ARMS:
            assert read(folder/arm/'TRAIN_COMPLETE.json')['save_load_exact'];assert len(list((folder/arm/'outputs').glob('*.json')))==32;paths.extend(folder/arm/n for n in ('active.pt','final.pt'))
        paths.append(v13/str(i)/'active.pt')
    assert len(paths)==10 and all(p.is_file() and not p.is_symlink() and (p.resolve().is_relative_to(root) or p.resolve().is_relative_to(v13)) for p in paths)
    receipt=dict(time=time.time(),files=[dict(path=str(p),bytes=p.stat().st_size) for p in paths],historical_assets_touched=False,reason='all registered V14 and V13 continuation consumers complete',weights_have_no_retained_copy=True,reconstruction_requires_training=True);receipt['total_bytes']=sum(x['bytes'] for x in receipt['files']);write(root/'RETIREMENT_PLAN.json',receipt)
    for p in paths:p.unlink()
    write(root/'RETIREMENT_COMPLETE.json',receipt);return dict(files=10,bytes=receipt['total_bytes'])

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['blind','summarize','retire']);p.add_argument('--private',type=Path);p.add_argument('--root',type=Path);p.add_argument('--v13',type=Path);a=p.parse_args()
    if a.mode=='blind':print(blind(a.private))
    elif a.mode=='summarize':result=summarize(a.private);write(a.private/'CONSUMERS_COMPLETE.json',result);print(result)
    else:print(retire(a.root,a.v13))
