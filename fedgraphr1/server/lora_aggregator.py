"""
fedgraphr1/server/lora_aggregator.py
=======================================
Server-side LoRA weight aggregation for Federated GRPO.

Implements FedAvg for LoRA parameters (§4.3.1):
  θ_global = Σ_k (n_k / N) * θ_k
where n_k = local training samples, N = total samples.

Design notes (§4.3.1):
  - Server-side aggregation is IDENTICAL for FedAvg and FedProx.
    FedProx differs only in the client-side loss function (proximal term).
  - Only LoRA parameter tensors are aggregated (lora_A, lora_B matrices).
    Base model weights are frozen and never transmitted.
  - Communication overhead: ~10-20 MB per client per round (rank=16).

Plan reference: Section 4.3.1 (FederatedLoRAServer)
               Phase 3 deliverable: server/lora_aggregator.py
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("fedgraphr1")


class FederatedLoRAServer:
    """Aggregate LoRA weights from multiple clients via FedAvg.

    Args:
        aggregation_strategy: "fedavg" (default) — more strategies can be
            added here without touching client code.

    Plan §4.3.1
    """

    def __init__(self, aggregation_strategy: str = "fedavg"):
        self.strategy = aggregation_strategy
        # The current global LoRA state dict (str → tensor)
        self.global_lora_state: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        client_updates: List[Tuple[str, Dict[str, torch.Tensor], int]],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client LoRA updates into a new global state.

        Args:
            client_updates: List of (client_id, lora_state_dict, num_samples).
                num_samples is used for sample-weighted averaging.

        Returns:
            Aggregated global LoRA state dict.

        Plan §4.3.1  "θ_global = Σ_k (n_k / N) * θ_k"
        """
        if not client_updates:
            logger.warning("[LoRAServer] No client updates received.")
            return self.global_lora_state or {}

        if self.strategy == "fedavg":
            aggregated = self._fedavg(client_updates)
        elif self.strategy == "masked_fedavg":
            aggregated = self._masked_fedavg(client_updates)
        else:
            raise ValueError(
                f"Unknown aggregation strategy '{self.strategy}'. "
                "Supported: 'fedavg', 'masked_fedavg'."
            )

        self.global_lora_state = aggregated
        logger.info(
            f"[LoRAServer] FedAvg aggregated {len(client_updates)} clients, "
            f"{len(aggregated)} parameter tensors."
        )
        return aggregated

    def _fedavg(
        self,
        client_updates: List[Tuple[str, Dict[str, torch.Tensor], int]],
    ) -> Dict[str, torch.Tensor]:
        """Sample-weighted averaging of LoRA tensors.

        Standard FedAvg: θ_global = Σ_k (n_k / N) * θ_k

        When clients use top-k sparsification (P4.6), zero positions in
        sparse updates naturally reduce the weighted average toward zero —
        this is the standard sparse-communication approach used in FL papers
        (zero-filling bias is acceptable and avoids the ambiguity between
        "true zero weight" and "sparsified zero").

        Plan §4.3.1: "θ_global = Σ_k (n_k / N) * θ_k"
        """
        total_samples = sum(n for _, _, n in client_updates)
        if total_samples == 0:
            # Fallback to uniform averaging if sample counts are unavailable
            total_samples = len(client_updates)
            client_updates = [
                (cid, sd, 1) for cid, sd, _ in client_updates
            ]

        aggregated: Dict[str, torch.Tensor] = {}
        for _, state_dict, n_samples in client_updates:
            weight = n_samples / total_samples
            for param_name, param_value in state_dict.items():
                pv = param_value.float()  # accumulate in float32
                if param_name not in aggregated:
                    aggregated[param_name] = torch.zeros_like(pv)
                aggregated[param_name] = aggregated[param_name] + weight * pv

        return aggregated

    def _masked_fedavg(
        self,
        client_updates: List[Tuple[str, Dict[str, torch.Tensor], int]],
    ) -> Dict[str, torch.Tensor]:
        """Mask-weighted FedAvg for sparse updates (P4.6 alternative).

        Excludes zero positions from each client's contribution so that
        sparsified zeros do not pull the global model toward zero:

            θ_global[i] = Σ_k w_k * θ_k[i] * m_k[i]
                         / max(Σ_k w_k * m_k[i], ε)

        where m_k[i] = 1 if θ_k[i] ≠ 0, else 0.

        Use this strategy when ALL clients apply the same top_k_ratio and
        zero positions are guaranteed to mean "not transmitted" rather than
        "true zero weight".  Pass aggregation_strategy="masked_fedavg" to
        FederatedLoRAServer to activate this path.
        """
        total_samples = sum(n for _, _, n in client_updates)
        if total_samples == 0:
            total_samples = len(client_updates)
            client_updates = [(cid, sd, 1) for cid, sd, _ in client_updates]

        aggregated: Dict[str, torch.Tensor] = {}
        weight_sum: Dict[str, torch.Tensor] = {}

        for _, state_dict, n_samples in client_updates:
            weight = n_samples / total_samples
            for param_name, param_value in state_dict.items():
                pv = param_value.float()
                mask = pv.bool().float()
                if param_name not in aggregated:
                    aggregated[param_name] = torch.zeros_like(pv)
                    weight_sum[param_name] = torch.zeros_like(pv)
                aggregated[param_name] += weight * pv * mask
                weight_sum[param_name] += weight * mask

        for param_name in aggregated:
            denom = weight_sum[param_name].clamp(min=1e-8)
            aggregated[param_name] = aggregated[param_name] / denom
            aggregated[param_name] *= (weight_sum[param_name] > 0).float()

        return aggregated

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save the global LoRA state dict to *path*.

        Plan §4.3.5 (Checkpoint & Resume)
        """
        if self.global_lora_state is None:
            logger.warning("[LoRAServer] Nothing to save — global state is empty.")
            return
        torch.save(self.global_lora_state, path)
        logger.info(f"[LoRAServer] Saved global LoRA state → {path}")

    def load(self, path: str):
        """Load a previously saved global LoRA state from *path*.

        Plan §4.3.5 (Checkpoint & Resume)
        """
        self.global_lora_state = torch.load(path, map_location="cpu")
        logger.info(
            f"[LoRAServer] Loaded global LoRA state from {path} "
            f"({len(self.global_lora_state)} tensors)"
        )

    def broadcast(
        self, client_ids: List[str]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Build per-client downlink payloads (all identical for FedAvg).

        Returns:
            Dict mapping client_id → global LoRA state dict copy.
        """
        if self.global_lora_state is None:
            return {cid: {} for cid in client_ids}
        return {
            cid: {k: v.clone() for k, v in self.global_lora_state.items()}
            for cid in client_ids
        }
