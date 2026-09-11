#!/usr/bin/env python3
"""
Hierarchical Mixture-of-Experts model: one family "router" BNN + one variant "expert"
BNN per family. Both reuse BNN_model.py's existing BinaryMLP/StandardMLP classes
unchanged - this module only orchestrates them and handles the stage-wise freeze.
"""

import torch
import torch.nn as nn

from BNN_model import BinaryMLP, StandardMLP


def _model_class(precision):
    if precision == "float32":
        return StandardMLP
    assert precision == "binary", f"unknown precision {precision!r}"
    return BinaryMLP


def _build_single(model_type, num_classes, in_features, hidden_sizes, precision):
    if model_type == "deep":
        hidden_sizes = (4096, 4096, 128)
    cls = _model_class(precision)
    return cls(in_features=in_features, hidden_sizes=tuple(hidden_sizes), num_classes=num_classes)


def build_moe(num_families, variants_per_family, in_features, model_type="shallow",
              router_hidden_sizes=(1024, 128), expert_hidden_sizes=(1024, 128), precision="binary"):
    """Returns (router, experts).

    variants_per_family: either a single int (same variant count for every family,
    kept for convenience/backward compat) or a list of length num_families giving
    each family's own variant count - required once families have non-uniform
    variant counts (e.g. after dropping rare classes from some families but not
    others), since a single global count would silently build the wrong-sized
    expert for any family that differs from it.

    router_hidden_sizes / expert_hidden_sizes: independent hidden-layer widths for
    the router vs. every expert (both ignored for model_type='deep', which keeps
    the fixed 4096->4096->128 architecture).

    precision: 'binary' (BinaryMLP, the hardware-deployable default) or 'float32'
    (StandardMLP, for the precision-ablation baseline).
    """
    if isinstance(variants_per_family, int):
        variants_per_family = [variants_per_family] * num_families
    assert len(variants_per_family) == num_families

    router = _build_single(model_type, num_families, in_features, router_hidden_sizes, precision)
    experts = [_build_single(model_type, variants_per_family[i], in_features, expert_hidden_sizes, precision)
               for i in range(num_families)]
    return router, experts


class HierarchicalMoEClassifier(nn.Module):
    """Wraps a router + per-family experts for checkpointing/freezing convenience."""

    def __init__(self, router, experts):
        super().__init__()
        self.router = router
        self.experts = nn.ModuleList(experts)

    def freeze_router(self):
        """Stage-wise handoff: stop the router's params from training and its
        BatchNorm running stats from drifting once expert training begins."""
        self.router.requires_grad_(False)
        self.router.eval()

    def predict_family(self, x):
        return self.router(x).argmax(dim=1)

    def predict_variant(self, x, family_idx):
        return self.experts[family_idx](x).argmax(dim=1)
