"""Reconstruction is an ordinary original-weight load, never an inference adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import os
import torch
from torch import nn
from .editor import MatrixRuntime, tensor_digest
from .contracts import enforce_storage


def export_native(runtime: MatrixRuntime, path: Path, *, base_binding: str,
                  storage_limit_bytes: int = 20 * 1024**3) -> dict[str, Any]:
    if len(base_binding) != 64:
        raise ValueError("required clean Base scientific digest")
    snapshot = runtime.weight.detach().cpu().clone()
    manifest = dict(format="base_plus_original_matrix", base_binding=base_binding,
                    path=runtime.path, shape=list(snapshot.shape), dtype=str(snapshot.dtype),
                    original_parameter_count=sum(p.numel() for p in runtime.model.parameters()),
                    editable_parameter_count=snapshot.numel(), new_deployment_parameters=0,
                    weight_digest=tensor_digest(snapshot), version=runtime.version,
                    base_tensor_digests={k:tensor_digest(v) for k,v in runtime.base_state.items()},
                    hooks_required=False, kv_cache_reusable=False)
    path.mkdir(parents=True, exist_ok=False)
    enforce_storage(path.parent, storage_limit_bytes, 2 * snapshot.numel() * snapshot.element_size() + 16384)
    temporary = path / "matrix.tmp"
    torch.save(snapshot, temporary)
    os.replace(temporary, path / "matrix.pt")
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_original_matrix(model: nn.Module, path: Path, *, base_binding: str) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["base_binding"] != base_binding:
        raise ValueError("wrong Base reconstruction")
    state = model.state_dict()
    if set(state) != set(manifest["base_tensor_digests"]) or any(
        tensor_digest(v) != manifest["base_tensor_digests"][k] for k,v in state.items()
    ):
        raise ValueError("loaded model is not the bound clean Base")
    weight = torch.load(path / "matrix.pt", map_location="cpu", weights_only=True)
    parameter = dict(model.named_parameters())[manifest["path"]]
    if list(parameter.shape) != manifest["shape"] or str(parameter.dtype) != manifest["dtype"] or tensor_digest(weight) != manifest["weight_digest"]:
        raise ValueError("matrix binding mismatch")
    with torch.no_grad():
        parameter.copy_(weight.to(parameter))
    return manifest


def verify_clean_reload(reference: nn.Module, reloaded: nn.Module, inputs: dict[str, Any],
                        *, atol: float = 1e-8, rtol: float = 1e-6) -> dict[str, Any]:
    a, b = reference.state_dict(), reloaded.state_dict()
    if set(a) != set(b) or any(a[k].shape != b[k].shape for k in a):
        raise ValueError("clean native keys/shape mismatch")
    if any(m._forward_hooks or m._forward_pre_hooks for m in reloaded.modules()):
        raise ValueError("clean inference cannot require hooks")
    with torch.no_grad():
        x, y = reference(**inputs), reloaded(**inputs)
        x = x.logits if hasattr(x, "logits") else x
        y = y.logits if hasattr(y, "logits") else y
    error = float((x-y).abs().max())
    return dict(status="PASS" if torch.allclose(x,y,atol=atol,rtol=rtol) else "FAIL",
                max_logit_error=error, argmax_equal=bool(torch.equal(x.argmax(-1),y.argmax(-1))),
                native_generation="PENDING_NATIVE_CHECK")
