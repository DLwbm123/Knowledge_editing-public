"""Non-destructive routed layer wrappers for paper-spec memory editors."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_slot(logical_edit_id: str) -> str:
    return "edit_" + hashlib.sha256(logical_edit_id.encode("utf-8")).hexdigest()[:24]


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


class GraceValueLinear(nn.Module):
    """Frozen base linear plus GRACE's routed value replacement."""

    def __init__(self, base: nn.Linear, *, replacement: str = "replace_prompt"):
        super().__init__()
        if replacement not in {"replace_prompt", "replace_last", "replace_all"}:
            raise ValueError(f"unsupported GRACE replacement: {replacement}")
        self.base = base
        freeze_module(self.base)
        self.replacement = replacement
        self.values = nn.ParameterDict()
        self.logical_to_slot: dict[str, str] = {}
        self.slot_to_logical: dict[str, str] = {}
        self.active_logical_id: str | None = None
        self.token_index = -1
        self.enabled = True

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def add_cold_value(self, logical_edit_id: str, *, seed: int) -> nn.Parameter:
        if logical_edit_id in self.logical_to_slot:
            return self.values[self.logical_to_slot[logical_edit_id]]
        slot = safe_slot(logical_edit_id)
        if slot in self.slot_to_logical and self.slot_to_logical[slot] != logical_edit_id:
            raise RuntimeError("logical-edit slot collision")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        value = torch.rand(self.out_features, generator=generator, dtype=torch.float32)
        parameter = nn.Parameter(value.to(self.base.weight.device), requires_grad=True)
        self.values[slot] = parameter
        self.logical_to_slot[logical_edit_id] = slot
        self.slot_to_logical[slot] = logical_edit_id
        return parameter

    def get_value(self, logical_edit_id: str) -> nn.Parameter:
        return self.values[self.logical_to_slot[logical_edit_id]]

    def set_active(self, logical_edit_id: str | None, *, token_index: int = -1) -> None:
        if logical_edit_id is not None and logical_edit_id not in self.logical_to_slot:
            raise KeyError(logical_edit_id)
        self.active_logical_id = logical_edit_id
        self.token_index = int(token_index)

    def disable(self) -> None:
        self.active_logical_id = None

    def train_only(self, logical_edit_id: str | None) -> list[nn.Parameter]:
        trainable = []
        for slot, parameter in self.values.items():
            active = logical_edit_id is not None and self.slot_to_logical[slot] == logical_edit_id
            parameter.requires_grad_(active)
            if active:
                trainable.append(parameter)
        return trainable

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        logical_id = self.active_logical_id
        if not self.enabled or logical_id is None:
            return base_output
        value = self.get_value(logical_id).to(dtype=base_output.dtype)
        if base_output.ndim == 2:
            expanded = base_output.unsqueeze(0)
            squeeze = True
        elif base_output.ndim == 3:
            expanded = base_output
            squeeze = False
        else:
            raise ValueError(f"unexpected linear output shape: {tuple(base_output.shape)}")
        sequence_length = expanded.shape[1]
        token_index = min(self.token_index, sequence_length - 1)
        replacement = value.view(1, 1, -1).expand(expanded.shape[0], sequence_length, -1)
        mask = torch.zeros(
            (expanded.shape[0], sequence_length, 1), dtype=torch.bool, device=expanded.device
        )
        if self.replacement == "replace_all":
            mask[:] = True
        elif self.replacement == "replace_last":
            mask[:, token_index, :] = True
        elif self.replacement == "replace_prompt":
            mask[:, :token_index, :] = True
        output = torch.where(mask, replacement, expanded)
        return output.squeeze(0) if squeeze else output

    def export_state(self) -> dict:
        return {
            "replacement": self.replacement,
            "values": {
                logical_id: self.values[slot].detach().cpu()
                for logical_id, slot in self.logical_to_slot.items()
            },
        }

    def load_exported_state(self, state: dict) -> None:
        if state["replacement"] != self.replacement:
            raise ValueError("GRACE replacement contract mismatch")
        self.values = nn.ParameterDict()
        self.logical_to_slot.clear()
        self.slot_to_logical.clear()
        for logical_id, value in state["values"].items():
            slot = safe_slot(logical_id)
            parameter = nn.Parameter(
                torch.as_tensor(value, dtype=torch.float32, device=self.base.weight.device),
                requires_grad=False,
            )
            self.values[slot] = parameter
            self.logical_to_slot[logical_id] = slot
            self.slot_to_logical[slot] = logical_id


class RoutedFullLinear(nn.Module):
    """BalanceEdit wrapper: base linear plus per-edit full transformation copies."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        freeze_module(self.base)
        self.edited = nn.ModuleDict()
        self.logical_to_slot: dict[str, str] = {}
        self.slot_to_logical: dict[str, str] = {}
        self.active_logical_id: str | None = None
        self.enabled = True

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def add_edit(self, logical_edit_id: str) -> nn.Linear:
        if logical_edit_id in self.logical_to_slot:
            raise ValueError(f"duplicate BalanceEdit transformation: {logical_edit_id}")
        slot = safe_slot(logical_edit_id)
        edited = copy.deepcopy(self.base).to(device=self.base.weight.device, dtype=torch.float32)
        for parameter in edited.parameters():
            parameter.requires_grad_(True)
        self.edited[slot] = edited
        self.logical_to_slot[logical_edit_id] = slot
        self.slot_to_logical[slot] = logical_edit_id
        return edited

    def get_edit(self, logical_edit_id: str) -> nn.Linear:
        return self.edited[self.logical_to_slot[logical_edit_id]]

    def set_active(self, logical_edit_id: str | None) -> None:
        if logical_edit_id is not None and logical_edit_id not in self.logical_to_slot:
            raise KeyError(logical_edit_id)
        self.active_logical_id = logical_edit_id

    def train_only(self, logical_edit_id: str | None) -> list[nn.Parameter]:
        trainable = []
        for slot, module in self.edited.items():
            active = logical_edit_id is not None and self.slot_to_logical[slot] == logical_edit_id
            for parameter in module.parameters():
                parameter.requires_grad_(active)
                if active:
                    trainable.append(parameter)
        return trainable

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logical_id = self.active_logical_id
        if not self.enabled or logical_id is None:
            return self.base(inputs)
        edited = self.get_edit(logical_id)
        return edited(inputs.to(dtype=torch.float32)).to(dtype=inputs.dtype)

    def export_state(self) -> dict:
        return {
            "edits": {
                logical_id: {
                    key: value.detach().cpu()
                    for key, value in self.edited[slot].state_dict().items()
                }
                for logical_id, slot in self.logical_to_slot.items()
            }
        }

    def load_exported_state(self, state: dict) -> None:
        self.edited = nn.ModuleDict()
        self.logical_to_slot.clear()
        self.slot_to_logical.clear()
        for logical_id, weights in state["edits"].items():
            edited = self.add_edit(logical_id)
            edited.load_state_dict(
                {
                    name: torch.as_tensor(value, dtype=torch.float32, device=self.base.weight.device)
                    for name, value in weights.items()
                }
            )
            for parameter in edited.parameters():
                parameter.requires_grad_(False)


class RoutedLoRALinear(nn.Module):
    """BELoRA linear wrapper with independent adapters keyed by logical edit ID."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        if rank <= 0 or alpha <= 0 or dropout != 0.0:
            raise ValueError("BELoRA lock requires positive rank/alpha and dropout=0")
        self.base = base
        freeze_module(self.base)
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = float(dropout)
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.logical_to_slot: dict[str, str] = {}
        self.slot_to_logical: dict[str, str] = {}
        self.active_logical_id: str | None = None
        self.enabled = True

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def add_adapter(self, logical_edit_id: str, *, seed: int) -> tuple[nn.Parameter, nn.Parameter]:
        if logical_edit_id in self.logical_to_slot:
            raise ValueError(f"duplicate BELoRA adapter: {logical_edit_id}")
        slot = safe_slot(logical_edit_id)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        a = torch.empty((self.rank, self.in_features), dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        b = torch.zeros((self.out_features, self.rank), dtype=torch.float32)
        self.lora_A[slot] = nn.Parameter(a.to(self.base.weight.device), requires_grad=True)
        self.lora_B[slot] = nn.Parameter(b.to(self.base.weight.device), requires_grad=True)
        self.logical_to_slot[logical_edit_id] = slot
        self.slot_to_logical[slot] = logical_edit_id
        return self.lora_A[slot], self.lora_B[slot]

    def set_active(self, logical_edit_id: str | None) -> None:
        if logical_edit_id is not None and logical_edit_id not in self.logical_to_slot:
            raise KeyError(logical_edit_id)
        self.active_logical_id = logical_edit_id

    def train_only(self, logical_edit_id: str | None) -> list[nn.Parameter]:
        trainable = []
        for slot in self.lora_A:
            active = logical_edit_id is not None and self.slot_to_logical[slot] == logical_edit_id
            self.lora_A[slot].requires_grad_(active)
            self.lora_B[slot].requires_grad_(active)
            if active:
                trainable.extend((self.lora_A[slot], self.lora_B[slot]))
        return trainable

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        logical_id = self.active_logical_id
        if not self.enabled or logical_id is None:
            return base_output
        slot = self.logical_to_slot[logical_id]
        value = inputs.to(dtype=torch.float32)
        delta = F.linear(F.linear(value, self.lora_A[slot]), self.lora_B[slot]) * self.scaling
        return base_output + delta.to(dtype=base_output.dtype)

    def export_state(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "adapters": {
                logical_id: {
                    "A": self.lora_A[slot].detach().cpu(),
                    "B": self.lora_B[slot].detach().cpu(),
                }
                for logical_id, slot in self.logical_to_slot.items()
            },
        }

    def load_exported_state(self, state: dict) -> None:
        if (state["rank"], state["alpha"], state["dropout"]) != (
            self.rank,
            self.alpha,
            self.dropout,
        ):
            raise ValueError("BELoRA adapter config mismatch")
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.logical_to_slot.clear()
        self.slot_to_logical.clear()
        for logical_id, weights in state["adapters"].items():
            slot = safe_slot(logical_id)
            self.lora_A[slot] = nn.Parameter(
                torch.as_tensor(weights["A"], dtype=torch.float32, device=self.base.weight.device),
                requires_grad=False,
            )
            self.lora_B[slot] = nn.Parameter(
                torch.as_tensor(weights["B"], dtype=torch.float32, device=self.base.weight.device),
                requires_grad=False,
            )
            self.logical_to_slot[logical_id] = slot
            self.slot_to_logical[slot] = logical_id


def trainable_parameters(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    return [parameter for module in modules for parameter in module.parameters() if parameter.requires_grad]
