"""Bounded metadata-only SLAKE admission; source answers are not selection inputs."""
import argparse, collections, json, re, time
from pathlib import Path


def audit(checkout, data, out):
    private = checkout / '.private'
    prior = json.loads((private / 'medmkeb_independent_admission_v15_20261019/CANDIDATES_FROZEN.json').read_text())
    manifests = {checkout / p for p in prior['history_sources']}
    for pattern in ['*/DATA_ADMISSION.json', '*/stage/DATA_ADMISSION.json', '*/completed/DATA_ADMISSION.json', '*/CANDIDATES_FROZEN.json', '*/SOURCE_ROWS.json', '*/ORIGINAL146_EXPOSURE_PRIVATE.json']:
        manifests.update(private.glob(pattern))
    manifests.discard(out / 'CANDIDATES_FROZEN.json')
    used, receipts = set(), []
    for p in sorted(manifests):
        if not p.exists():
            raise FileNotFoundError(p)
        assert p.stat().st_size < 50_000_000
        obj = json.loads(p.read_text())
        if isinstance(obj, dict) and 'files_with_image_references' in obj:
            obj = [x['image_keys'] for x in obj['files_with_image_references'] if 'EXCLUSION_IDS' not in x['path']]
        groups = set(x.lower() for x in re.findall(r'xmlab\d+', json.dumps(obj), re.I))
        used.update(groups)
        receipts.append(dict(path=str(p),source_groups=len(groups),bytes=p.stat().st_size))
    splits, groups_by_split = {}, {}
    for split in ['train', 'validation', 'test']:
        rows = json.loads((data / (split + '.json')).read_text())
        splits[split] = rows
        groups_by_split[split] = {Path(x['img_name']).parts[0].lower() for x in rows}
    cross = set()
    for a in splits:
        for b in splits:
            if a != b: cross |= groups_by_split[a] & groups_by_split[b]
    candidates, counts = [], {}
    for split, rows in splits.items():
        by_group = collections.defaultdict(list)
        for row in rows:
            if row['q_lang'] != 'en' or row['answer_type'] != 'OPEN': continue
            group = Path(row['img_name']).parts[0].lower()
            if group in used or group in cross: continue
            by_group[group].append(row)
        selected = sorted(by_group, key=lambda g: int(g[5:]))[:8]
        counts[split] = dict(all_image_groups=len(groups_by_split[split]),unmatched_open_groups=len(by_group),selected=len(selected))
        for group in selected:
            rows = sorted(by_group[group], key=lambda x: int(x['qid']))
            candidates.append(dict(split=split,source_group=group,metadata_image=rows[0]['img_name'],qid_pool=[x['qid'] for x in rows],role={'train':'TRAIN_CANDIDATE','validation':'DEV_CANDIDATE','test':'CONFIRM_CANDIDATE'}[split],patient_independence='UNKNOWN',source_admission='PENDING',exposure='NO_MATCH_IN_ENUMERATED_MANIFESTS_NOT_PROOF_OF_NO_EXPOSURE'))
    summary = dict(status='METADATA_ONLY_NO_GPU_ADMISSION',counts=counts,historical_slake_groups=len(used),cross_split_groups=len(cross),history_manifest_count=len(receipts),model_calls=0,pixel_reads=0,answers_used_for_selection=False,patient_independence='UNKNOWN',next='Check actual historical native-image aliases and source provenance before reading selected training answers or launching model qualification.')
    out.mkdir(parents=True,exist_ok=True)
    target=out/'CANDIDATES_FROZEN.json';assert not target.exists()
    target.write_text(json.dumps(dict(summary=summary,candidates=candidates,history_receipts=receipts,time=time.time()),indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(audit(a.checkout,a.data,a.out),indent=2))
