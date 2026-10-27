"""Compare frozen source predicates to existing object annotations conservatively."""
import collections
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.private/med150_source_provenance_v1_20261028'

def main():
    data=json.loads((OUT/'ANNOTATIONS_PRIVATE.json').read_text());assert len(data)==8
    frozen=json.loads((ROOT/'.private/med150_source_visual_v1_20261028/INPUTS_FROZEN.private.json').read_text())['groups']
    qa=json.loads((ROOT/'.private/med150_source_review_v1_20261028/SOURCE_PACKET_PRIVATE.json').read_text())['records']
    assert len(frozen)==8 and sum(x['target_qid'] is not None for x in frozen)==6
    per=[];nempty=0;total=0
    def source_status(row,labels):
        answer=str(row['answer']).strip().casefold()
        # Positive categories support a label at source-annotation level only.
        if answer and answer!='none' and answer in labels:return 'EXPLICIT_POSITIVE_CATEGORY_PRESENT'
        if row['content_type']=='Quantity' and answer=='2' and {'left kidney','right kidney'}<=labels:return 'BILATERAL_CATEGORY_PRESENT'
        return 'UNKNOWN_NO_MATCHING_POSITIVE_CATEGORY'
    for i,(row,v) in enumerate(zip(frozen,data),1):
        assert row['group_index']==i and v['mask_exists'];annotation=v['detection'];assert isinstance(annotation,list)
        labels=[]
        for item in annotation:
            assert isinstance(item,dict) and len(item)==1
            k,box=next(iter(item.items()));assert isinstance(k,str) and isinstance(box,list) and len(box)==4
            assert all(isinstance(z,(int,float)) for z in box);labels.append(k.strip().casefold())
        assert len(labels)==len(set(labels));nempty+=not labels;total+=len(labels)
        if row['target_qid'] is None:
            per.append(dict(group_index=i,status='NO_PRIOR_BASE_PAIR',object_annotations=len(labels)));continue
        target=source_status(qa[str(row['target_qid'])],set(labels))
        protect=[dict(qid=q,source_status=source_status(qa[str(q)],set(labels))) for q in row['protect_qids']]
        per.append(dict(group_index=i,target_source_status=target,protections=protect,object_annotations=len(labels),clinical='UNKNOWN',patient_independence='UNKNOWN'))
    t=collections.Counter(v['target_source_status'] for v in per if 'target_source_status' in v)
    p=collections.Counter(x['source_status'] for v in per for x in v.get('protections',[]))
    assert sum(t.values())==6 and sum(p.values())==9
    private=dict(status='SOURCE_ANNOTATION_AUDIT_COMPLETE_NOT_CLINICALLY_ADMITTED',time=time.time(),per_group=per,clinical='UNKNOWN',patient_independence='UNKNOWN')
    target=OUT/'REVIEW_PRIVATE.json';assert not target.exists();target.write_text(json.dumps(private,indent=2)+'\n')
    public=dict(status=private['status'],groups=8,detection_files=8,detection_nonempty_groups=8-nempty,detection_empty_groups=nempty,
        positive_object_annotations=total,prior_base_potential_groups=6,prior_base_protection_questions=9,
        target_positive_category_present=t['EXPLICIT_POSITIVE_CATEGORY_PRESENT'],target_unknown_no_positive_category=t['UNKNOWN_NO_MATCHING_POSITIVE_CATEGORY'],
        protection_positive_category_present=p['EXPLICIT_POSITIVE_CATEGORY_PRESENT']+p['BILATERAL_CATEGORY_PRESENT'],
        protection_unknown_no_positive_category=p['UNKNOWN_NO_MATCHING_POSITIVE_CATEGORY'],
        empty_detection_is_negative=False,per_record_physician_attestation=False,patient_identifier_in_detection=False,
        patient_independence='UNKNOWN',clinical_ground_truth='UNKNOWN',image_level_split_paper_vs_local_group_overlap=272,
        comparisons='Source annotation concordance only; no independent clinical verification',model_calls=0,pixel_reads=0,qa_split_files_opened=0)
    dst=ROOT/'reports/med150_source_provenance_v1_20261028';(dst/'RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    text='''# MED固定来源组：原始标注与来源证据

[SLAKE原论文](https://arxiv.org/html/2102.09542)称有经验的医师参与视觉语义标注、医生提出问题；这提供数据集级来源信息，并不是本次对每个病例重新进行的医师核验，也不能保证本地数据版本与论文原版完全一致。

八个固定TRAIN_DEV组的`detection.json`均存在且可解析；6组有正向目标框标注，2组为空。依据此前固定的6个Base潜在目标与9个潜在保护问题，来源类别标注与**1/6个目标**、**3/9个保护问题**存在直接的正向对应。其余5个目标和6个保护问题缺少可对应的正向类别标注，仍为UNKNOWN；尤其空检测结果或未标注类别不能解释为无病、无器官或问题错误。该对照只说明数据源内部的类别一致性，不代表框、问答、图像判断或临床诊断已被独立验证。

这8个检测文件的数据结构只包含类别及框坐标，未提供患者身份或研究级关联字段；患者独立性仍为UNKNOWN。全部6个目标和9个保护问题的独立临床真实性仍为UNKNOWN，不能当作正式合格训练样本或临床保护证明。2个上轮没有Base潜在组合的组仍保留，不因标注筛掉或补组。

[原论文](https://arxiv.org/html/2102.09542)描述按图像划分训练/验证/测试，但本机既有元数据审计发现272个train图像组也出现在validation或test，已在固定候选前整组排除。数据集作者的[页面](https://www.med-vqa.com/slake/)说明后续清洗版本可能与论文略有不同；尚不能确定本地版本身份或该差异的原因，更不能据此宣称患者级隔离。

本轮仅读取固定8组的检测标注和mask文件存在性；没有读取mask像素、split问答、其他组或保留组内容，没有模型调用、训练或评分。逐例标注、图像、问答及路径保持私有；原MED150、M13P与既有负结果不变。
'''
    (dst/'REPORT_ZH.md').write_text(text);print(json.dumps(public))

if __name__=='__main__':main()
