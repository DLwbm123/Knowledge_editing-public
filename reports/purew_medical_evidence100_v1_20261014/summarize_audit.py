"""Summarize private source-only triage; no evaluation/model/server entry point.

Inputs remain private. Only aggregate counts and limitations are written publicly.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

SOURCE_KEYS = {
    'order', 'dataset', 'question', 'answer', 'original_fits', 'source_metadata',
    'source_split', 'source_answer_matches', 'image_file', 'image_size',
    'image_format', 'boxes', 'mask',
}
META_KEYS = {'content_type', 'modality', 'location', 'image_organ',
             'question_type', 'answer_type', 'q_lang'}
KINDS = {'LOCAL_IMAGE', 'GLOBAL_IMAGE', 'KG'}


def valid_box(box, size):
    if len(box) != 4 or any(type(v) not in (int, float) or
                            not math.isfinite(v) for v in box):
        return False
    x, y, width, height = box
    return (x >= 0 and y >= 0 and width > 0 and height > 0 and
            x + width <= size[0] and y + height <= size[1])


def fits_preserve_source(row):
    q = row['question']
    expected = [f'Please answer the following question: {q}', f'Question: {q}',
                f'{q} Please provide an answer.', f'Please respond to this question: {q}']
    return row['original_fits'] == expected


def summarize(packet, review):
    assert set(packet) == {'rows', 'evaluation_material_read', 'model_calls', 'gpu_calls'}
    assert packet['evaluation_material_read'] is False
    assert packet['model_calls'] == packet['gpu_calls'] == 0
    rows = packet['rows']
    assert len(rows) == 100 and [r['order'] for r in rows] == list(range(1, 101))
    assert len(review['rows']) == 100
    decisions = {r['order']: r for r in review['rows']}
    assert len(decisions) == 100 and set(decisions) == set(range(1, 101))
    counts, local_status, blockers, all_reasons = Counter(), Counter(), Counter(), Counter()
    box_questions = pixel_masks = fits = matched = visual_reviews = 0
    unique_images, annotation_images = set(), {}
    by_dataset = {}
    eligible = []
    for row in rows:
        assert set(row) == SOURCE_KEYS, 'Reject unexpected input fields'
        assert set(row['source_metadata']) <= META_KEYS
        assert row['source_answer_matches'] is True
        assert fits_preserve_source(row), 'FIT scope differs from original wrapper contract'
        matched += 1
        fits += len(row['original_fits'])
        d = decisions[row['order']]
        assert d['primary_kind'] in KINDS and isinstance(d['reason'], str) and d['reason']
        assert d['pixel_mask_admitted'] is False and d['neutral_view_admitted'] is False
        assert all(valid_box(b['xywh'], row['image_size']) and b['valid'] is True
                   for b in row['boxes'])
        assert set(d['selected_box_indices']) <= {b['index'] for b in row['boxes']}
        status = d['region_status']
        if status == 'ELIGIBLE_COARSE_BOX':
            assert d['primary_kind'] == 'LOCAL_IMAGE'
            assert d['selected_box_indices'] and d['reason_codes'] == []
            assert d['source_semantic_binding'] is True and d['source_visual_triage'] is True
            eligible.append(row['order'])
        else:
            assert not d['selected_box_indices']
            assert (d['primary_kind'] == 'LOCAL_IMAGE') == status.startswith('BLOCKED_')
        if d['primary_kind'] == 'LOCAL_IMAGE':
            local_status[status] += 1
            blockers.update(d['reason_codes'])
        all_reasons.update(d['reason_codes'])
        counts[d['primary_kind']] += 1
        by_dataset.setdefault(row['dataset'], Counter())[d['primary_kind']] += 1
        box_questions += bool(row['boxes'])
        visual_reviews += d['source_visual_triage']
        unique_images.add(row['image_file'])
        if row['mask'] is not None:
            mask = row['mask']
            assert mask['present'] is True and mask['aligned_size'] is True
            assert mask['size'] == row['image_size']
            annotation_images[row['image_file']] = row
            pixel_masks += 1
    assert counts.total() == 100 and local_status.total() == counts['LOCAL_IMAGE']
    assert len(eligible) == review['eligible_coarse_box_count']
    mask_buckets = Counter('few_colors_le8' if r['mask']['unique_colors'] <= 8 else
                          'many_colors_gt256' if r['mask']['unique_colors'] > 256 else
                          'intermediate' for r in annotation_images.values())
    return {
        'status': 'SOURCE_TRIAGE_COMPLETE_COARSE_BOX_SUPPORT_ONLY',
        'source_native_questions': 100, 'source_native_answers_matched': matched,
        'source_answer_match_normalization': 'NFKC, casefold and whitespace; not a byte-identity claim',
        'unique_source_images': len(unique_images),
        'original_FIT_wrappers_verified': fits,
        'reviewer': review['reviewer'], 'independent_clinician_review': False,
        'source_visual_triage_questions': visual_reviews,
        'primary_kind_counts': dict(counts),
        'primary_kind_by_dataset': {k: dict(v) for k, v in by_dataset.items()},
        'original_metadata_KG_questions': sum(r['source_metadata'].get('content_type') == 'KG'
                                              for r in rows),
        'local_region_status': dict(local_status),
        'local_blocking_reason_counts': dict(blockers),
        'coarse_region_admitted_questions': len(eligible),
        'coarse_region_admitted_scope': 'existing brain MRI lesion box unions only',
        'pixel_mask_admitted_questions': 0, 'neutral_view_admitted_questions': 0,
        'annotation_availability': {
            'questions_with_detection_file': box_questions,
            'unique_images_with_detection_and_mask_files': len(annotation_images),
            'valid_xywh_boxes_on_unique_images': sum(len(r['boxes']) for r in annotation_images.values()),
            'questions_with_mask_file': pixel_masks,
            'unique_mask_mode_counts': dict(Counter(r['mask']['mode'] for r in annotation_images.values())),
            'unique_mask_color_buckets': dict(mask_buckets),
            'mask_label_map_found': any(r['mask']['semantic_mapping_present'] for r in annotation_images.values()),
            'image_mask_dimension_mismatches': 0,
            'VQA_RAD_named_regions_or_masks_found': 0,
        },
        'source_split_counts': dict(Counter(r['dataset'] + '/' + r['source_split'] for r in rows)),
        'source_role': 'frozen native editing inputs, not an official train-only dataset split',
        'evaluation_panel_fields_in_review_input': 0,
        'native_target_exception': 'existing native edit QAs may also be T0 under the frozen active-target rule; no independent evaluation material loaded',
        'additional_model_or_Judge_requests': 0, 'GPU_calls': 0, 'new_training_updates': 0,
        'clinical_causal_evidence_qualification': 'NOT_ESTABLISHED',
        'patient_independence': 'UNKNOWN', 'full_cohort_region_coverage': False,
        'original146': 'keep paused', 'OL': 'keep original predecessor barrier',
    }


def selfcheck():
    assert valid_box([1, 2, 3, 4], [8, 8])
    assert not valid_box([1, 2, 30, 4], [8, 8])
    assert not valid_box([1, 2, float('nan'), 4], [8, 8])
    assert not valid_box([1, 2, 0, 4], [8, 8])
    assert not valid_box([True, 2, 3, 4], [8, 8])
    q = 'private source'
    r = {'question': q, 'original_fits': [f'Please answer the following question: {q}',
         f'Question: {q}', f'{q} Please provide an answer.', f'Please respond to this question: {q}']}
    assert fits_preserve_source(r)
    r['original_fits'][0] += ' new fact'
    assert not fits_preserve_source(r)
    assert 'evaluation_answer' not in SOURCE_KEYS and 'base_correct' not in SOURCE_KEYS
    return {'status': 'PASS', 'box_bounds_and_nonfinite_rejected': True,
            'FIT_content_change_rejected': True, 'source_field_allowlist': True,
            'clinical_image_truth_checked_by_this_test': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    if args.selfcheck:
        print(json.dumps(selfcheck()))
    else:
        assert args.packet and args.review and args.output
        result = summarize(json.loads(args.packet.read_text()), json.loads(args.review.read_text()))
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'status': result['status'], 'questions': 100,
                          'eligible_coarse_boxes': result['coarse_region_admitted_questions']}))
