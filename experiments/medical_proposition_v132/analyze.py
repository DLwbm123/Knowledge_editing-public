"""Anonymous paired summaries; no questions, answers, image identities or private paths."""
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
import urllib.request


def summarize(local, metadata):
    frozen=json.loads((local/'CONSUMERS_FROZEN.json').read_text())
    verdicts=json.loads((local/'DECISIONS.json').read_text())['decisions']
    rows=[dict(v,**{k:x for k,x in verdicts[v['payload_key']].items() if k!='opaque_query_id'})
          for v in frozen['consumers']]
    assert len(rows)==345 and len(verdicts)==frozen['unique']
    arms=('BASE','LEGACY','FULL_CTX','MED','GEN')
    owners=metadata['owners']
    panels=('T0','FIT','T1G','T2G','T1L','T2L')
    metrics={}
    for arm in arms:
        metrics[arm]={}
        for panel in panels:
            values=[r for r in rows if r['arm']==arm and panel in r['panels'] and
                    (panel not in ('T1L','T2L') or r['original_Base_qualified'])]
            counts=Counter(r['owner'] for r in values)
            metrics[arm][panel]=dict(n=len(values),support_edits=len(counts),
                **{field:dict(correct=sum(v[field] for v in values),
                    edit_macro_percent=100*statistics.mean([
                        sum(v[field] for v in values if v['owner']==o)/counts[o]
                        for o in counts]) if counts else None)
                    for field in ('core_correct','full_supported')})
    bykey={(r['arm'],r['owner'],r['query_id']):r for r in rows}
    localities=[r for r in rows if r['arm']=='BASE' and any(p in ('T1L','T2L') for p in r['panels'])]
    damage={}
    for arm in arms[1:]:
        damage[arm]={}
        for field in ('core_correct','full_supported'):
            base=[r for r in localities if r[field]]
            kept=[r for r in base if bykey[('LEGACY',r['owner'],r['query_id'])][field]]
            old_lost=[r for r in base if not bykey[('LEGACY',r['owner'],r['query_id'])][field]]
            damage[arm][field]=dict(Base_correct=len(base),Base_and_legacy_correct=len(kept),
                total_Base_damage=sum(not bykey[(arm,r['owner'],r['query_id'])][field] for r in base),
                newly_damaged=sum(not bykey[(arm,r['owner'],r['query_id'])][field] for r in kept),
                old_damage_repaired=sum(bykey[(arm,r['owner'],r['query_id'])][field] for r in old_lost))
    per_edit={}
    for index,o in enumerate(owners):
        per_edit[f'E{index+1:02}']={}
        for arm in arms:
            v=[r for r in rows if r['owner']==o and r['arm']==arm and 'T2G' in r['panels']]
            assert v
            per_edit[f'E{index+1:02}'][arm]=dict(n=len(v),core_correct=sum(r['core_correct'] for r in v),
                full_supported=sum(r['full_supported'] for r in v))
    contrasts={}
    for ref in ('LEGACY','FULL_CTX','GEN'):
        delta=[100*(v['MED']['core_correct']/v['MED']['n']-v[ref]['core_correct']/v[ref]['n'])
               for v in per_edit.values()]
        rng=random.Random(20261026)
        sampled=sorted(statistics.mean(rng.choices(delta,k=4)) for _ in range(10000))
        contrasts['MED_minus_'+ref]=dict(T2G_macro_pp=statistics.mean(delta),
            up=sum(v>0 for v in delta),same=sum(v==0 for v in delta),down=sum(v<0 for v in delta),
            descriptive_edit_bootstrap95=[sampled[249],sampled[9749]],
            source_groups_same_four_as_edits=True,independent_confirmation=False)
    quality={arm:dict(n=sum(r['arm']==arm for r in rows),EOS=sum(r['arm']==arm and r['EOS'] for r in rows),
        at_cap=sum(r['arm']==arm and r['at_cap'] for r in rows),
        unsupported_expansion=sum(r['arm']==arm and r['unsupported_expansion'] for r in rows),
        extra_contradiction=sum(r['arm']==arm and r['extra_contradiction'] for r in rows)) for arm in arms}
    signal=all(contrasts['MED_minus_'+x]['T2G_macro_pp']>0 for x in ('LEGACY','FULL_CTX','GEN'))
    signal=signal and all(damage['MED'][field]['newly_damaged']==0 for field in ('core_correct','full_supported'))
    result=dict(status='SMALL_DEV_DIRECTIONAL_SIGNAL' if signal else 'NO_JOINT_SMALL_DEV_IMPROVEMENT',
        scope='Four exposed development edits; anatomy-presence mapping; not clinical or independent confirmation',
        metrics=metrics,paired_T2G=contrasts,per_edit=per_edit,locality_damage=damage,
        quality=quality,cost=metadata['cost'],curvature=metadata['curvature'],
        grading=dict(protocol='MEDPROP_SMALL4_CORE_FULL_SOURCE_V1',model='gpt-6-astra',effort='medium',
                     consumers=345,unique_payloads=frozen['unique'],missing=0,new_common_epoch=True),
        no_automatic_expansion=True,time=time.time())
    lines=['# 4编辑医学命题保护实验结果','',
        '**本轮为四编辑开发试验；不构成独立确认，不自动扩展到146。**','',
        '比较同一共同评分epoch的Base、原M13P（LEGACY）、FULL_CTX、MED和GEN；历史accepted标签未更改。',
        'FULL按verified reference的保守来源支持判定，参考外的医学解释即使可能真实，也不能当作已验证。','',
        '|方法|T0 CORE|FIT CORE|T1G CORE|T2G CORE|T2G FULL|固定T2L CORE|',
        '|---|---:|---:|---:|---:|---:|---:|']
    def cell(arm,panel,field='core_correct'):
        v=metrics[arm][panel];x=v[field]['edit_macro_percent']
        return 'NA' if x is None else f"{x:.2f}% ({v[field]['correct']}/{v['n']})"
    for arm in arms:
        lines.append('|'+arm+'|'+'|'.join([cell(arm,'T0'),cell(arm,'FIT'),cell(arm,'T1G'),cell(arm,'T2G'),
            cell(arm,'T2G','full_supported'),cell(arm,'T2L')])+'|')
    lines+=['','百分比为编辑宏平均，括号为微计数；固定locality只使用原资格掩码，空支持为NA。','']
    for ref,v in contrasts.items():
        lines.append(f"- {ref}：T2G {v['T2G_macro_pp']:+.2f}pp；{v['up']}升/{v['same']}同/{v['down']}降；"
                     f"四编辑描述性bootstrap95%区间[{v['descriptive_edit_bootstrap95'][0]:.2f}, {v['descriptive_edit_bootstrap95'][1]:.2f}]pp。")
    lines+=['','## 保护与完整回答','',
        '以下新增损伤只在本共同epoch的Base正确且LEGACY仍正确身份上计数；旧伤修复单列，不能抵消新伤。',
        '这是小样本已有locality输入的审计，不能保证未测病例或全域安全。','',
        '|方法|CORE新增损伤/支持|FULL新增损伤/支持|CORE旧伤修复|FULL旧伤修复|',
        '|---|---:|---:|---:|---:|']
    for arm in arms[1:]:
        a,b=damage[arm]['core_correct'],damage[arm]['full_supported']
        lines.append(f"|{arm}|{a['newly_damaged']}/{a['Base_and_legacy_correct']}|{b['newly_damaged']}/{b['Base_and_legacy_correct']}|{a['old_damage_repaired']}|{b['old_damage_repaired']}|")
    lines+=['','## 机制与费用','',
        '31/61原BASIS项构造解剖结构出现命题，覆盖24来源组；其余项三臂共同保留原曲率。逐锚点匹配trace；MED/GEN逐项秩1谱匹配，不声称全局谱相同。',
        '实际mask、有限差分、legacy一步、dense二次型及梯度、换基和save/load检查的回执与冻结配置保存在私有运行记录。',
        'GPU进程驻留时间包含加载、等待和共享资源竞争，不是纯GPU计算时间；逻辑forward包括PrefixCache复用的调用，不能直接当作完整模型FLOPs。','',
        '```json',json.dumps(dict(cost=result['cost'],curvature=result['curvature'],quality=quality),ensure_ascii=False,indent=2),'```','',
        '## 本轮判断','',
        ('出现有限开发正信号，仍需更充分验证。' if signal else '未观察到本轮预定联合方向性改善。'),
        '四编辑和单个GEN方向仅约束本轮；完整逐编辑计数、全部对照、损伤、生成完整性与费用见RESULTS.json。没有用本轮结果改变候选、种子、步数、权重或来源，也没有自动启动扩大实验。','']
    return result,'\n'.join(lines)


def publish(config, result, report):
    work=Path(config['workspace']);folder=work/'reports/medical_proposition_small4_v132_20261026'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    (folder/'REPORT_ZH.md').write_text(report)
    env=dict(__import__('os').environ)
    env.update(HTTP_PROXY='http://127.0.0.1:7897',HTTPS_PROXY='http://127.0.0.1:7897',
               ALL_PROXY='http://127.0.0.1:7897',NO_PROXY='',no_proxy='')
    def git(*args):
        return subprocess.check_output(['git','-C',str(work),*args],env=env,text=True).strip()
    assert git('branch','--show-current')==config['branch']
    assert git('rev-parse','HEAD')==config['publication_parent'], 'Branch advanced; preserve report for review'
    assert git('remote','get-url','origin')==config['remote_url']
    assert not git('diff','--cached','--name-only'), 'Unrelated staged changes preserved'
    files=['reports/medical_proposition_small4_v132_20261026/'+x for x in ('RESULTS.json','REPORT_ZH.md')]
    subprocess.run(['git','-C',str(work),'diff','--check'],check=True,env=env)
    git('add','--',*files)
    assert set(git('diff','--cached','--name-only').splitlines())==set(files)
    git('commit','-m','Report fixed four-edit medical proposition results')
    commit=git('rev-parse','HEAD')
    git('-c','http.proxy=http://127.0.0.1:7897','push','origin','HEAD:refs/heads/'+config['branch'])
    remote_sha=git('-c','http.proxy=http://127.0.0.1:7897','ls-remote','origin','refs/heads/'+config['branch']).split()[0]
    assert remote_sha==commit
    url='https://raw.githubusercontent.com/'+config['repository']+'/'+commit+'/'+files[1]
    opener=urllib.request.build_opener(urllib.request.ProxyHandler(dict(http='http://127.0.0.1:7897',https='http://127.0.0.1:7897')))
    with opener.open(url,timeout=30) as response:
        assert response.status==200
        response.read(100)
    return dict(status='PUBLIC_PUSH_AND_ANONYMOUS_ACCESS_VERIFIED',commit=commit,
        url='https://github.com/'+config['repository']+'/blob/'+commit+'/'+files[1],time=time.time())
