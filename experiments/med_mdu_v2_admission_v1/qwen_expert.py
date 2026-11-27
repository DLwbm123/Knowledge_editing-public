"""TT expert and mechanical checks for the Qwen2.5-VL-7B replication."""
import os

import torch
from torch import nn


class QwenExpert(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.epsilon = 1e-6
        self.G1 = nn.Parameter(torch.zeros(1, 56, 8))
        rng = torch.Generator().manual_seed(seed)
        for name, shape in (('G2', (8, 64, 4)), ('G3', (4, 148, 8)), ('G4', (8, 128, 1))):
            x = torch.randn(shape[1] * shape[2], shape[0], generator=rng)
            q, r = torch.linalg.qr(x, mode='reduced')
            q *= torch.where(torch.diag(r) < 0, -1., 1.)
            setattr(self, name, nn.Parameter(q.T.reshape(shape).contiguous()))

    def factors(self):
        output = torch.einsum('ia,ajb->ijb', self.G1[0], self.G2).reshape(3584, 4)
        inputs = torch.einsum('aib,bj->aij', self.G3, self.G4[:, :, 0]).reshape(4, 18944)
        return output, inputs

    def residual(self, activation):
        b, a = self.factors()
        x = activation.to(a.dtype)
        x = x / (x.square().mean(-1, keepdim=True).sqrt() + self.epsilon)
        return ((x @ a.T) @ b.T)

    def parameter_groups(self):
        return [self.G3, self.G4], [self.G1, self.G2]


def verify():
    import json
    from pathlib import Path
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

    model = Path(os.environ['QWEN_MODEL_CHECK'])
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    shards = set(index.values())
    assert all((model / filename).is_file() for filename in shards)
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    with torch.device('meta'):
        network = Qwen2_5_VLForConditionalGeneration(config)
    modules = [(n, m) for n, m in network.named_modules() if n.endswith('.layers.18.mlp.down_proj')]
    assert len(modules) == 1
    name, layer = modules[0]
    assert f'{name}.weight' in index
    assert (layer.in_features, layer.out_features) == (18944, 3584)

    torch.manual_seed(20261127)
    expert = QwenExpert(20261127)
    assert sum(p.numel() for p in expert.parameters()) == 8256
    x = torch.randn(1, 5, layer.in_features)
    assert torch.count_nonzero(expert.residual(x)) == 0
    with torch.no_grad():
        expert.G1.normal_(std=0.01)
    b, a = expert.factors()
    x = x / (x.square().mean(-1, keepdim=True).sqrt() + expert.epsilon)
    expected = (x @ a.T) @ b[:16].T
    dense = b[:16] @ a
    assert torch.allclose(x @ dense.T, expected, rtol=3e-5, atol=3e-6)

    with torch.no_grad():
        from methods.medtrace.core import MedTraceLayerHook
        layer.to_empty(device='cpu')
        layer.weight.zero_()
        baseline = layer(x)
        hook = MedTraceLayerHook(layer, expert)
        hook.attach()
        try:
            labels = torch.tensor([[-100, -100, -100, 1, 2]])
            hook.set_teacher_routing(labels)
            output = layer(x)
            expected_mask = torch.tensor([[False, False, True, True, False]])
            assert torch.equal((output != baseline).any(-1), expected_mask)
            assert torch.allclose(output[expected_mask], expert.residual(x)[expected_mask])
        finally:
            hook.detach()
    print(json.dumps(dict(model_type=config.model_type, module=name, input=layer.in_features,
                          output=layer.out_features, num_layers=config.num_hidden_layers, shards=len(shards),
                          parameters=8256, zero_residual=True, dense_equivalence=True, teacher_mask=True,
                          checkpoint_weight_loaded=False)))


if __name__ == '__main__':
    verify()
