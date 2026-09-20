import unittest
from r2_contract_patch import (ContractError, OutputRegistry, consumer_id,
                               group_slots, require_native_verdicts,
                               resolve_teacher)


class ContractPatchTests(unittest.TestCase):
    def test_list_slots_keeps_double_prefix(self):
        grouped = group_slots([{"t": 7, "expert": 7}, {"t": 7, "expert": 6}])
        self.assertEqual([x["expert"] for x in grouped[7]], [7, 6])

    def test_missing_teacher_cannot_fallback(self):
        with self.assertRaisesRegex(ContractError, "MISSING_EXACT_TEACHER"):
            resolve_teacher("active", input_fp="a", teacher_version="v", prefix=2, store={})

    def test_masked_role_does_not_lookup_teacher(self):
        self.assertIsNone(resolve_teacher("masked", prefix=4, available_from=3,
                                          masked_reference=True, store={}))

    def test_consumer_contains_event_and_version(self):
        a = consumer_id("run", "B1", 7, "slot-1", "input", "w0")
        b = consumer_id("run", "B1", 7, "slot-2", "input", "w1")
        self.assertNotEqual(a, b)

    def test_identity_and_barrier(self):
        r = OutputRegistry(); cid = consumer_id("run", "B1", 7, "slot-1", "input", "w0")
        r.observe(cid, [1, 2], "yes")
        with self.assertRaisesRegex(ContractError, "EXECUTION_IDENTITY_VIOLATION"):
            r.observe(cid, [1, 3], "yes")
        with self.assertRaisesRegex(ContractError, "NATIVE_RELEASE_BARRIER"):
            require_native_verdicts({cid}, {})
        require_native_verdicts({cid}, {cid: {"status": "COMPLETE", "correct": True}})


if __name__ == "__main__":
    unittest.main()
