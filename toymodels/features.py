"""A data source of "features"."""
from __future__ import annotations

from typing import Optional

import torch


class FeatureSet:
    def __init__(
        self,
        n_features: int,
        n_groups: Optional[int] = None,
        minval: float = 0.0,
        maxval: float = 1.0,
        sparsity: float = 0.0,
    ):
        n_groups = n_features if n_groups is None else n_groups

        assert (
            n_features % n_groups == 0
        ), f"n_groups ({n_groups}) needs to evenly divide n_features ({n_features})"
        assert minval < maxval

        self.n_features = n_features
        self.n_groups = n_groups
        self.minval = minval
        self.maxval = maxval
        self.sparsity = sparsity

    def sample(self) -> torch.Tensor:
        """Generates a sample from the feature set specification."""
        vals = torch.rand(self.n_features) * (self.maxval - self.minval) + self.minval
        mask = torch.repeat_interleave(
            torch.rand(self.n_groups) < self.sparsity,
            self.n_features // self.n_groups,
        )

        vals[mask] = 0

        return vals
