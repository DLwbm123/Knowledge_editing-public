from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn

from m3bench_repro.editors.llava_runtime import build_target_only_labels
from m3bench_repro.editors.routed_layers import (
    GraceValueLinear,
    RoutedFullLinear,
    RoutedLoRALinear,
)
from m3bench_repro.editors.routing import (
    GraceCodebook,
    MemoryRouter,
    balanced_radius,
    cosine_distances,
    euclidean_distances,
)


class RoutingTests(unittest.TestCase):
    def test_cosine_numerical(self):
        keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        query = torch.tensor([1.0, 0.0])
        self.assertTrue(torch.allclose(cosine_distances(keys, query), torch.tensor([0.0, 1.0])))

    def test_euclidean_numerical(self):
        keys = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        query = torch.tensor([0.0, 0.0])
        self.assertTrue(torch.allclose(euclidean_distances(keys, query), torch.tensor([0.0, 5.0])))

    def test_radius_below_equal_above(self):
        router = MemoryRouter("euclidean")
        router.add("edit", torch.tensor([0.0]), 1.0)
        self.assertTrue(router.route(torch.tensor([0.999])).activated)
        self.assertTrue(router.route(torch.tensor([1.0])).activated)
        self.assertFalse(router.route(torch.tensor([1.001])).activated)

    def test_balance_radius_regression(self):
        radius = balanced_radius(
            torch.tensor([0.0]),
            torch.tensor([0.2]),
            torch.tensor([1.0]),
            alpha=0.2,
            distance="euclidean",
        )
        self.assertAlmostEqual(float(radius), 0.84, places=6)

    def test_router_roundtrip(self):
        router = MemoryRouter("cosine")
        router.add("a", torch.tensor([1.0, 0.0]), 0.2, [7, 8])
        restored = MemoryRouter.from_state(router.export_state())
        self.assertEqual(restored.logical_ids, ["a"])
        self.assertTrue(restored.route(torch.tensor([1.0, 0.0])).activated)


class GraceCodebookTests(unittest.TestCase):
    def test_insert_reuse_and_collision(self):
        codebook = GraceCodebook(distance="cosine", eps_init=1.0)
        first = codebook.insert_with_source_semantics("a", torch.tensor([1.0, 0.0]), [10])
        self.assertEqual(first.action, "insert")
        reused = codebook.insert_with_source_semantics("a2", torch.tensor([0.99, 0.01]), [10])
        self.assertEqual(reused.action, "same_label_reuse")
        self.assertEqual(reused.effective_logical_edit_id, "a")
        collision = codebook.insert_with_source_semantics("b", torch.tensor([0.0, 1.0]), [20])
        self.assertEqual(collision.action, "collision_split")
        self.assertEqual(len(codebook), 2)

    def test_euclidean_far_insert_diagnostic(self):
        codebook = GraceCodebook(distance="euclidean", eps_init=1.0)
        codebook.insert_with_source_semantics("a", torch.tensor([0.0]), [1])
        result = codebook.insert_with_source_semantics("b", torch.tensor([3.0]), [2])
        self.assertEqual(result.action, "insert_far")

    def test_source_label_match(self):
        self.assertTrue(GraceCodebook.source_label_match([1, 3], [2, 2]))
        self.assertFalse(GraceCodebook.source_label_match([1, 3], [2, 3]))


class TargetMaskTests(unittest.TestCase):
    def test_prompt_image_target_padding_contract(self):
        prefix = torch.tensor([[1, -200, 2, 3]])
        full = torch.tensor([[1, -200, 2, 3, 7, 8, 0]])
        attention = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
        labels, targets = build_target_only_labels(
            full, prefix, attention, image_token_index=-200
        )
        self.assertEqual(targets, (7, 8))
        self.assertTrue(torch.equal(labels, torch.tensor([[-100, -100, -100, -100, 7, 8, -100]])))
        labels2, targets2 = build_target_only_labels(
            full.clone(), prefix.clone(), attention.clone(), image_token_index=-200
        )
        self.assertTrue(torch.equal(labels, labels2))
        self.assertEqual(targets, targets2)

    def test_prefix_mismatch_fails_closed(self):
        with self.assertRaises(RuntimeError):
            build_target_only_labels(
                torch.tensor([[1, 2, 4]]),
                torch.tensor([[1, 3]]),
                torch.ones((1, 3), dtype=torch.long),
                image_token_index=-200,
            )


class RoutedLayerTests(unittest.TestCase):
    def test_grace_value_hit_miss_and_roundtrip(self):
        base = nn.Linear(2, 3, bias=False)
        nn.init.zeros_(base.weight)
        wrapper = GraceValueLinear(base, replacement="replace_prompt")
        value = wrapper.add_cold_value("edit.one", seed=9)
        value.data.copy_(torch.tensor([1.0, 2.0, 3.0]))
        inputs = torch.ones((1, 4, 2))
        miss = wrapper(inputs)
        self.assertTrue(torch.equal(miss, torch.zeros_like(miss)))
        wrapper.set_active("edit.one", token_index=3)
        hit = wrapper(inputs)
        self.assertTrue(torch.equal(hit[0, :3], torch.tensor([[1.0, 2.0, 3.0]]).expand(3, -1)))
        self.assertTrue(torch.equal(hit[0, 3], torch.zeros(3)))
        state = wrapper.export_state()
        restored = GraceValueLinear(copy.deepcopy(base), replacement="replace_prompt")
        restored.load_exported_state(state)
        restored.set_active("edit.one", token_index=3)
        self.assertTrue(torch.equal(hit, restored(inputs)))

    def test_balancedit_full_copy_hit_miss_isolation(self):
        base = nn.Linear(2, 2, bias=False)
        nn.init.eye_(base.weight)
        wrapper = RoutedFullLinear(base)
        original = wrapper(torch.tensor([[1.0, 2.0]]))
        edit = wrapper.add_edit("a")
        nn.init.zeros_(edit.weight)
        wrapper.set_active("a")
        self.assertTrue(torch.equal(wrapper(torch.tensor([[1.0, 2.0]])), torch.zeros((1, 2))))
        wrapper.set_active(None)
        self.assertTrue(torch.equal(wrapper(torch.tensor([[1.0, 2.0]])), original))
        self.assertTrue(torch.equal(base.weight, torch.eye(2)))

    def test_belora_logical_id_coordinates_full_adapter_set(self):
        base1 = nn.Linear(2, 2, bias=False)
        base2 = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(base1.weight)
        nn.init.zeros_(base2.weight)
        wrappers = [
            RoutedLoRALinear(base1, rank=1, alpha=1, dropout=0.0),
            RoutedLoRALinear(base2, rank=1, alpha=1, dropout=0.0),
        ]
        for index, wrapper in enumerate(wrappers):
            a, b = wrapper.add_adapter("edit-a", seed=10 + index)
            a.data.fill_(1.0)
            b.data.fill_(1.0 + index)
            wrapper.set_active("edit-a")
        x = torch.tensor([[1.0, 2.0]])
        outputs = [wrapper(x) for wrapper in wrappers]
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
        for wrapper in wrappers:
            wrapper.set_active(None)
            self.assertTrue(torch.equal(wrapper(x), torch.zeros_like(x)))

    def test_belora_two_edit_parameter_isolation_and_roundtrip(self):
        base = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(base.weight)
        wrapper = RoutedLoRALinear(base, rank=1, alpha=1, dropout=0.0)
        a1, b1 = wrapper.add_adapter("one", seed=1)
        a2, b2 = wrapper.add_adapter("two", seed=2)
        a1.data.fill_(1.0)
        b1.data.fill_(1.0)
        a2.data.fill_(2.0)
        b2.data.fill_(2.0)
        x = torch.tensor([[1.0, 1.0]])
        wrapper.set_active("one")
        out1 = wrapper(x)
        wrapper.set_active("two")
        out2 = wrapper(x)
        self.assertFalse(torch.equal(out1, out2))
        state = wrapper.export_state()
        restored = RoutedLoRALinear(copy.deepcopy(base), rank=1, alpha=1, dropout=0.0)
        restored.load_exported_state(state)
        restored.set_active("one")
        self.assertTrue(torch.equal(out1, restored(x)))
        self.assertTrue(torch.equal(base.weight, torch.zeros_like(base.weight)))


if __name__ == "__main__":
    unittest.main()
