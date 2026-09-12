"""Runnable CPU checks; no raw data, models, GPUs, or Judge calls."""
from copy import deepcopy
import unittest

from scripts.medtrace.stage17_contract import correctness, require_binding, support_masks
from scripts.medtrace.stage17_inventory import summarize


class Contract(unittest.TestCase):
    def test_inventory_is_not_a_newly_authorized_cohort(self):
        data = {'cohort': {'final_t0_n': 179, 'tasks': {'T0': dict(candidate_edit_count=189,
            eligible_edit_count=179, eligible_probe_count=179, unique_image_count=159)}}}
        data.update({k: {'event_ids': [k]} for k in ('dev', 'qual8', 'qual16')})
        value = summarize({k: {'content': v} for k, v in data.items()})
        self.assertEqual(value['historical']['eligible_T0'], 179)
        self.assertIsNone(value['actual_stage17_N'])
        self.assertTrue(all(v['count'] is None for v in value['support_masks'].values()))
        self.assertEqual(value['raw_formal_inputs_read_by_this_inventory'], 0)

    def test_reused_judge_schema_and_validator(self):
        from scripts.medtrace.astra_judge_bundle import schema, validate
        batch = {'batch_id': 'stage17_test', 'records': [{'opaque_query_id': '0'*64}]}
        response = {'batch_id': 'stage17_test', 'decisions': [dict(opaque_query_id='0'*64, is_correct=False)]}
        self.assertFalse(validate(batch, response)[0]['is_correct'])
        self.assertFalse(schema(batch)['additionalProperties'])
        response['decisions'][0]['is_correct'] = 0
        with self.assertRaises(ValueError):
            validate(batch, response)

    def test_masks_and_common_support(self):
        rows = [dict(edit_id='a', roles=dict(U_fit=True, H_fit=True, G=False, H_eval=None)),
                dict(edit_id='b', roles=dict(U_fit=None, H_fit=True, G=True, H_eval=True))]
        m = support_masks(rows)
        self.assertTrue(m['E_H']['a'])
        self.assertFalse(m['E_HG']['a'])
        self.assertIsNone(m['E_U']['b'])
        with self.assertRaises(ValueError):
            support_masks(rows + rows[:1])

    def test_frozen_denominators_and_native_failure(self):
        rows = [dict(edit_id='a', base_correct=True, post_correct=False, native_correct=False),
                dict(edit_id='a', base_correct=False, post_correct=True, native_correct=False),
                dict(edit_id='b', base_correct=True, post_correct=True, native_correct=True)]
        self.assertEqual(correctness(rows, 'retention')['probe_micro'], .5)
        self.assertEqual(correctness(rows, 'fix')['probe_micro'], 1.)
        self.assertEqual(correctness(rows, 'postacc')['edit_macro'], .75)
        self.assertEqual(correctness(rows, 'paircorrect')['edit_macro'], .5)
        self.assertIsNone(correctness([], 'fix')['probe_micro'])
        with self.assertRaises(ValueError):
            correctness([dict(rows[0], post_correct=None)], 'postacc')

    def test_cache_binds_state_not_answer_string(self):
        value = dict(input={'ids': [1], 'image': 'x'}, runtime='r', generation={'cap': 1024},
                     writer='w', prefix=1, training={'support': ['u'], 'seed': 1})
        require_binding(value, deepcopy(value))
        for field in value:
            bad = deepcopy(value); bad[field] = None
            with self.assertRaises(ValueError):
                require_binding(bad, value)
        with self.assertRaises(ValueError):
            require_binding({}, {})


if __name__ == '__main__':
    unittest.main()
