"""Record nonclinical image/predicate observations and retire only staged copies."""
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.private/med150_source_visual_v1_20261028'
TEMP=Path('/private/tmp/j.20261028.9')


def main():
    frozen=json.loads((OUT/'INPUTS_FROZEN.private.json').read_text());stage=json.loads((OUT/'STAGING_RECEIPT.json').read_text());rows=frozen['groups']
    assert len(rows)==8 and stage['groups']==8 and stage['decoded']
    pairs=[]
    for row in rows:
        if row['target_qid'] is None:continue
        assert row['group_index'] in (2,4,5,6,7,8)
        for q in row['protect_qids']:
            pairs.append(dict(group_index=row['group_index'],target_qid=row['target_qid'],protect_qid=q,text_attributes_different=True,
                medical_grounding='UNKNOWN',patient_independence='UNKNOWN',source_image_byte_identity='NOT_INDEPENDENTLY_VERIFIED'))
    assert len(pairs)==9
    private=dict(status='NONCLINICAL_REVIEW_COMPLETE_NO_CLINICAL_ADMISSION',time=time.time(),images_seen=8,nonblank_images=8,
        pixel_dimensions=[[320,320],[256,256],[256,256],[256,256],[256,256],[240,240],[240,240],[240,240]],
        target_groups=6,pairs=pairs,unpaired_groups=2,clinical_confirmation=False,physician_review=False,
        image_observation='Original pictures visually nonblank and coarse grayscale layouts visible; no sequence, organ, count or pathology truth established',
        restrictions='Text-level distinct attribute does not establish ground-truth labels or clinically independent protection')
    p=OUT/'REVIEW.private.json';assert not p.exists();p.write_text(json.dumps(private,indent=2)+'\n')
    public=dict(status=private['status'],images_seen=8,nonblank_images=8,image_dimensions=dict(sizes=[[320,320],[256,256],[240,240]],counts=[1,4,3]),
        original_groups=8,potential_base_groups=6,groups_without_pair=2,potential_protection_pairs=9,text_distinct_attribute_pairs=9,
        medically_verified_targets=0,medical_target_status='UNKNOWN',medical_protection_status='UNKNOWN',patient_independence='UNKNOWN',
        clinical_confirmation=False,physician_review=False,model_calls=0,source_sizes_match=True,byte_identity_verified=False,
        limitation='Nonclinical appearance and textual distinctions do not establish image truth, independence, or method effectiveness')
    dst=ROOT/'reports/med150_source_visual_v1_20261028';(dst/'RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    report='''# MED固定来源组：非临床图像与谓词审查

八组原图均可解码、非空白，灰度图的粗略布局可见；图像分别为320×320（1张）、256×256（4张）和240×240（3张）。非临床观察不能确认MRI序列、器官数量或异常诊断，也不能证明任一参考答案是临床真值。原始图像按既定路径只读复制，工具完成且目标大小/可解码性检查通过；没有逐字节验证复制身份。

按上阶段预先固定的6个潜在目标、9条Base FULL支持的保护问题，对原问题文本进行对照：9对所问属性在文字层面不同（例如成像属性与数量，或异常与成像/器官属性）；这不证明两项均由图像独立支持，也不证明患者层面的独立性。剩余2组本来没有符合Base可行性规则的组合，保留在总分母，不补位。

因此，6个目标的医学图像支持、9个保护关系的医学图像支持及全部患者/研究独立性仍为UNKNOWN。数据集标注和Astra来源一致性不替代医学专业审查；不能据此称6组合格、临床安全或独立确认。原图及逐条问答仅在受限本地临时目录查看，不公开；原MED R0及既有负结果不改变。

全部8个固定原图均纳入查看；无新增模型调用、训练或评分。若要推进MED开发，可仅在明确标注“来源假设、医学未确认”的前提下另行冻结方法及对照；正式临床或患者独立结论仍须可追溯的专业来源。结果不能通过替换困难组或读取保留集来补齐。
'''
    (dst/'REPORT_ZH.md').write_text(report)
    files=[TEMP/f'{i:02d}.png' for i in range(1,9)];assert all(p.is_file() for p in files)
    assert {p.name for p in TEMP.iterdir()}=={p.name for p in files}
    total=sum(p.stat().st_size for p in files)
    for p in files:p.unlink()
    TEMP.rmdir();(OUT/'TEMP_CLEANUP_RECEIPT.json').write_text(json.dumps(dict(time=time.time(),files=8,bytes=total,originals_unchanged=True),indent=2)+'\n')
    print(json.dumps(dict(status=public['status'],groups=8,pairs=9,clinical='UNKNOWN',temporary_copies_retired=8)))

if __name__=='__main__':main()
