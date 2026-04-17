"""
fedgraphr1/server/lora_sparsifier.py
======================================
Top-k sparsification for LoRA parameter communication (§P4.6).

Reduces uplink communication cost by transmitting only the top-k% of
LoRA parameter values (by absolute magnitude) per tensor.  Zero-valued
positions are excluded from the server-side aggregation using a
mask-weighted FedAvg (see FederatedLoRAServer._sparse_fedavg).

Design (paper ablation):
  - top_k_ratio=1.0  → no sparsification  (FedAvg baseline)
  - top_k_ratio=0.1  → top 10% per tensor (~10× compression)
  - top_k_ratio=0.01 → top 1% per tensor  (~100× compression)

Per-tensor sparsification (not global) is used because LoRA A/B matrices
have different magnitudes across layers — a global threshold would
disproportionately zero out low-variance layers.

Plan reference: Section P4.6 (LoRA sparsification)
"""

from __future__ import annotations

import logging
from typing import Dict

import torch

logger = logging.getLogger("fedgraphr1")


class TopKSparsifier:
    """Top-k magnitude sparsification for LoRA state dicts.

    All methods are static — no instance state needed.
    """

    @staticmethod
    def sparsify(
        state_dict: Dict[str, torch.Tensor],
        top_k_ratio: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Return a new state dict with only the top-k% values kept per tensor.

        Elements below the k-th percentile threshold (by |value|) are set to
        zero.  The returned tensors are detached CPU float32 clones.

        Args:
            state_dict: LoRA state dict (param_name → tensor).
            top_k_ratio: Fraction of elements to keep, in (0, 1].
                         1.0 = keep all (no-op).  0.1 = keep top 10%.

        Returns:
            New state dict with zeros in low-magnitude positions.
        """
        if top_k_ratio >= 1.0:
            return {k: v.clone().cpu().float() for k, v in state_dict.items()}

        top_k_ratio = max(1e-6, min(1.0, top_k_ratio))
        sparse = {}
        for name, tensor in state_dict.items():
            t = tensor.detach().cpu().float()
            n_total = t.numel()
            k = max(1, int(round(n_total * top_k_ratio)))

            flat = t.reshape(-1)
            abs_flat = flat.abs()
            # kth-largest threshold
            if k < n_total:
                threshold = torch.topk(abs_flat, k, largest=True, sorted=False).values.min()
                mask = abs_flat >= threshold
                sparse_flat = flat * mask.float()
            else:
                sparse_flat = flat.clone()

            sparse[name] = sparse_flat.reshape(t.shape)

        return sparse

    @staticmethod
    def density(state_dict: Dict[str, torch.Tensor]) -> float:
        """Fraction of non-zero elements across all tensors.

        Useful as a paper metric: density=0.1 means 90% communication saving.
        """
        total = 0
        nonzero = 0
        for t in state_dict.values():
            total += t.numel()
            nonzero += t.bool().sum().item()
        return nonzero / total if total > 0 else 0.0

    @staticmethod
    def estimate_bytes(
        state_dict: Dict[str, torch.Tensor],
        sparse: bool = False,
    ) -> int:
        """Estimate uplink bytes for *state_dict*.

        Args:
            state_dict: LoRA state dict.
            sparse: If True, count only non-zero elements (coordinate format:
                    value + index, each float32/int32 = 8 bytes per element).
                    If False, count all elements as dense float32 (4 bytes).

        Returns:
            Estimated byte count.
        """
        total = 0
        for t in state_dict.values():
            if sparse:
                nnz = int(t.bool().sum().item())
                total += nnz * 8   # float32 value + int32 index
            else:
                total += t.numel() * 4  # dense float32
        return total
