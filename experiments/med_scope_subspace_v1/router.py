"""Projection and matched prototype routing with negative-only calibration."""
import math

import torch

from .scope_solver import row_basis


class ScopeRouter:
    def __init__(self, bank, fits, prototypes=None):
        self.owners = sorted(bank)
        if not self.owners or set(fits) != set(bank):
            raise ValueError('Expert/FIT bindings differ')
        if prototypes is not None and set(prototypes) != set(bank):
            raise ValueError('Expert/prototype bindings differ')
        self.threshold = None
        self.dimension = bank[self.owners[0]].size(-1)
        if prototypes is None:
            bases = [row_basis(bank[o]) for o in self.owners]
            self.directions = torch.stack(bases)
            self.fit_scale = torch.stack([
                torch.quantile((fits[o].to(a) @ a.T).norm(dim=-1), .5)
                for o, a in zip(self.owners, bases, strict=True)])
        else:
            means = torch.stack([prototypes[o].mean(0) for o in self.owners])
            norms = means.norm(dim=-1)
            if (norms <= 0).any():
                raise ValueError('Zero prototype')
            self.directions = means / norms[:, None]
            self.fit_scale = None
        if self.fit_scale is not None and (not torch.isfinite(self.fit_scale).all()
                                           or (self.fit_scale <= 0).any()):
            raise ValueError('Invalid FIT denominator')

    def scores(self, feature):
        x = feature.reshape(-1, self.dimension).to(self.directions)
        if self.fit_scale is None:
            return (x @ self.directions.T) / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scores = x @ self.directions.flatten(0, 1).T
        return scores.view(len(x), len(self.owners), -1).norm(dim=-1) / self.fit_scale

    def calibrate(self, feature_groups, alpha):
        if not 0 < alpha < 1:
            raise ValueError('alpha must be strictly between zero and one')
        # One exchangeable calibration score per image, using its worst query.
        scores = [float(self.scores(x).max()) for x in feature_groups]
        if not all(math.isfinite(x) for x in scores):
            raise ValueError('Nonfinite calibration score')
        k = math.ceil((len(scores) + 1) * (1 - alpha))
        self.threshold = sorted(scores)[k - 1] if k <= len(scores) else math.inf
        return self.threshold

    def route(self, feature):
        if self.threshold is None:
            raise ValueError('Calibrate before routing')
        values, indexes = self.scores(feature).max(dim=1)
        return [(self.owners[ix] if score > self.threshold else None, score)
                for score, ix in zip(values.tolist(), indexes.tolist(), strict=True)]
