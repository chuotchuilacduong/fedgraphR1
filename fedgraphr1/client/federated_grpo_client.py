"""
fedgraphr1/client/federated_grpo_client.py
============================================
Client-side GRPO training engine with LoRA for federated learning.

Wraps a PEFT-enabled causal LM and exposes:
  - set_lora_state_dict()     : load global LoRA weights from server
  - get_lora_state_dict()     : extract local LoRA delta for upload
  - compute_proximal_loss()   : FedProx regularisation term (§4.3.1)
  - train_local_epochs()      : run E epochs of local GRPO training

Design decisions (§4.3.1):
  - Only LoRA parameters are trained / communicated (~0.1% of total params).
  - Advantage computation is PURE LOCAL (per-prompt, no global baseline).
    This preserves GRPO's correct semantics in the FL setting (§4.2.2).
  - FedProx proximal term (μ/2)‖θ_k − θ_global‖² can be added to the
    GRPO loss when args.fedprox_mu > 0 (default 0 = FedAvg behaviour).
  - The GRPO loss itself is computed outside this class by the existing
    verl/trainer/ppo/core_algos.py — this class handles only the
    LoRA lifecycle and FL-specific modifications.

Plan reference: Section 4.3 (Quy trình Đồng bộ Trọng số Mô hình)
               Phase 3 deliverable: client/federated_grpo_client.py
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import torch

logger = logging.getLogger("fedgraphr1")


class FederatedGRPOClient:
    """Manages the LoRA-based GRPO training state for one federated client.

    This class is intentionally thin — it handles LoRA weight management
    and the FedProx proximal term.  The actual GRPO training loop
    (rollout generation, reward computation, PPO-clip update) is delegated
    to FederatedRayPPOTrainer.train_local_epochs().

    Args:
        base_model_path: HuggingFace model ID or local path for the base LLM.
        lora_rank: LoRA rank r (default 16, from §4.3.2 config).
        lora_alpha: LoRA scaling alpha (default 32).
        lora_target_modules: List of module names to apply LoRA to.
        fedprox_mu: Proximal term coefficient μ.  0.0 = FedAvg, >0 = FedProx.
        device: torch.device or string.

    Plan §4.3.1
    """

    def __init__(
        self,
        base_model_path: str,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: Optional[List[str]] = None,
        fedprox_mu: float = 0.0,
        device: str = "auto",
    ):
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        self.base_model_path = base_model_path
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.fedprox_mu = fedprox_mu
        self.device = _resolve_device(device)

        # Cached global LoRA params for FedProx proximal term
        self._global_lora_params: Optional[Dict[str, torch.Tensor]] = None

        # The PEFT model — initialised lazily on first use
        self._model = None

    # ------------------------------------------------------------------
    # Lazy model initialisation
    # ------------------------------------------------------------------

    def _ensure_model(self):
        """Lazily load and wrap the base model with LoRA."""
        if self._model is not None:
            return

        logger.info(
            f"[FederatedGRPOClient] Loading base model: {self.base_model_path}"
        )
        try:
            from transformers import AutoModelForCausalLM
            from peft import get_peft_model, LoraConfig

            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
            ).to(self.device)

            lora_cfg = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=0.05,
                bias="none",
            )
            self._model = get_peft_model(base, lora_cfg)
            # Freeze base model — only LoRA parameters are trainable
            for name, param in self._model.named_parameters():
                if "lora_" not in name:
                    param.requires_grad = False

            logger.info(
                f"[FederatedGRPOClient] LoRA model ready — "
                f"rank={self.lora_rank}, "
                f"alpha={self.lora_alpha}, "
                f"targets={self.lora_target_modules}"
            )
        except ImportError as e:
            logger.warning(
                f"[FederatedGRPOClient] Could not load model ({e}). "
                "LoRA operations will be no-ops."
            )

    # ------------------------------------------------------------------
    # LoRA weight management
    # ------------------------------------------------------------------

    def get_lora_state_dict(
        self, top_k_ratio: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """Extract LoRA weights for upload to the server.

        Returns only the lora_A / lora_B matrices — ignoring base model
        weights to minimise communication overhead.

        Args:
            top_k_ratio: If given, apply top-k sparsification (P4.6).
                         E.g. 0.1 keeps the top 10% of values per tensor.
                         None (default) = no sparsification.

        Plan §4.3.1, P4.6
        """
        self._ensure_model()
        if self._model is None:
            return {}
        state_dict = {
            name: param.data.clone().cpu()
            for name, param in self._model.named_parameters()
            if "lora_" in name
        }
        if top_k_ratio is not None and top_k_ratio < 1.0:
            from fedgraphr1.server.lora_sparsifier import TopKSparsifier
            state_dict = TopKSparsifier.sparsify(state_dict, top_k_ratio)
            density = TopKSparsifier.density(state_dict)
            logger.debug(
                f"[FederatedGRPOClient] LoRA sparsified at "
                f"top_k={top_k_ratio:.2%} → density={density:.4f}"
            )
        return state_dict

    def set_lora_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ):
        """Load aggregated LoRA weights from the server.

        Also caches a copy of the global params for the FedProx proximal
        term (§4.3.1 set_lora_state_dict caches global params).

        Plan §4.3.1
        """
        self._ensure_model()
        if self._model is None or not state_dict:
            return

        # Cache global params for FedProx before overwriting
        self._global_lora_params = {
            k: v.clone().detach().to(self.device) for k, v in state_dict.items()
        }

        model_state = self._model.state_dict()
        for name, param in state_dict.items():
            if name in model_state:
                model_state[name].copy_(param.to(self.device))

        logger.debug(
            f"[FederatedGRPOClient] Loaded {len(state_dict)} LoRA parameter tensors."
        )

    # ------------------------------------------------------------------
    # FedProx proximal term
    # ------------------------------------------------------------------

    def compute_proximal_loss(self) -> torch.Tensor:
        """Compute FedProx proximal term: (μ/2) ‖θ_k − θ_global‖².

        Returns torch.tensor(0.0) when fedprox_mu == 0 (FedAvg mode)
        or when no global weights have been received yet.

        Plan §4.3.1  "FedProx proximal term (CLIENT-SIDE)"
        """
        if self.fedprox_mu == 0.0 or self._global_lora_params is None:
            return torch.tensor(0.0, device=self.device)

        if self._model is None:
            return torch.tensor(0.0, device=self.device)

        prox_loss = torch.tensor(0.0, device=self.device)
        for name, param in self._model.named_parameters():
            if "lora_" in name and name in self._global_lora_params:
                diff = param - self._global_lora_params[name]
                prox_loss = prox_loss + diff.pow(2).sum()

        return (self.fedprox_mu / 2.0) * prox_loss

    # ------------------------------------------------------------------
    # Training delegation (thin wrapper — actual loop is in trainer)
    # ------------------------------------------------------------------

    def get_model(self):
        """Return the PEFT-wrapped model for use by FederatedRayPPOTrainer."""
        self._ensure_model()
        return self._model

    @property
    def num_lora_params(self) -> int:
        """Total number of trainable LoRA parameters."""
        self._ensure_model()
        if self._model is None:
            return 0
        return sum(
            p.numel()
            for n, p in self._model.named_parameters()
            if "lora_" in n and p.requires_grad
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> torch.device:
    """Resolve 'auto' → best available device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
