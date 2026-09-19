"""Read frozen train-only contracts; no new medical judgment or source scan."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage18_support import validate_task
from scripts.medtrace.prepare_stage2_sources import normalized, reviewed_attribute


def run(directory):
    d=Path(directory);stream=json.loads((d/'private/run/private/STREAM.json').read_text())
    tasks=stream['tasks'];assert len(tasks)==45
    key=lambda r:(r['image_sha256'],normalized(r['question']))
    rows=[]
    for t in tasks:
        validate_task(t,fasttrack_branch='C_FACT')
        h={key(r) for r in t['H_fit']};u={key(r) for r in t['U_fit']}
        a=reviewed_attribute(t['native']['question'])
        overlap=len(h&u)
        conflicts=sum(key(x)==key(y) and normalized(x['reference'])!=normalized(y['reference']) for x in t['H_fit'] for y in t['U_fit'])
        rows.append(dict(position=t['order'],H_U_same_QA=overlap,H_U_conflicting_reference=conflicts,
                         U_same_native_attribute=sum(a is not None and a==reviewed_attribute(r['question']) for r in t['U_fit']),
                         H_sources=len({r['source_group'] for r in t['H_fit']})))
    assert all(r['H_U_conflicting_reference']==0 for r in rows)
    result=dict(status='EXISTING_TRAIN_CONTRACT_CPU_AUDIT',validated_tasks=45,development_tasks=19,rows=rows,
        reference='Frozen historical qualification reused; no new medical-reference approval. Expanded H and sealed candidates require separate qualification.',
        answer_mask='Exact question/image prefix and padding ignored (-100); one image, nonempty target, expanded target tokens equal unexpanded completion. build_target_only_labels and build_edit_batch.',
        EOS='Existing completion tokenization preserved; trainer rejects missing EOS. No answer rewrite or tokenization repair.',
        CE='Existing runtime model loss on target-only labels, original shifted causal mean; native .5 + fit .5 + weighted H.',
        U_KL='Full vocabulary FP32 Base||student, sum vocabulary / number of predictor positions, temperature1. Frozen Base OFF, exact generated teacher tokens, no truncation; >128 rejected.',
        teacher_binding='36 retained caches CPU validated; live loader also checks source/input/image/labels/predictor positions/runtime/generation/code ancestry/distribution shape.',
        H_exposure='Historical FACT: 320 one-H slots per edit, source-balanced schedule already existed; actual new weighted gradient cosines only steps1/80/160/320.',
        route_coverage='Historical H/evaluation routing is available in frozen outputs. Expanded train-H candidate key/radius coverage remains pending Stage22C; no coverage claim yet.',
        clinical_signature=False,patient_study='UNKNOWN',new_GPU_seconds=0,new_judgments=0)
    (d/'TRAINING_CONTRACT_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Validated train-only contracts:',len(rows),'H/U same QA:',sum(r['H_U_same_QA'] for r in rows),'same-native U attribute:',sum(r['U_same_native_attribute'] for r in rows))


if __name__=='__main__':run(sys.argv[1])
