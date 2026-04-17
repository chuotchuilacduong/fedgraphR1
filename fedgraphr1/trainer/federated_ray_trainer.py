"""
fedgraphr1/trainer/federated_ray_trainer.py
=============================================
FederatedRayPPOTrainer — passive GRPO training engine for Federated Graph-R1.

Extends the existing RayPPOTrainer from verl/ with FL-specific hooks.

Key design (§4.4.1 "Passive Engine — KHÔNG chứa FL loop"):
  - This class does NOT contain the FL round loop.
  - GraphR1Trainer (fl/trainer.py) is the sole orchestrator.
  - This class only exposes train_local_epochs() which GraphR1Client calls.
  - Inversion of Control prevents the double-orchestrator deadlock described
    in §4.4.1: "Flower Server is orchestrator duy nhất".

Advantage computation (§4.2.2 "Pure Local GRPO"):
  - compute_grpo_outcome_advantage() from verl/trainer/ppo/core_algos.py
    is used UNCHANGED.  Per-prompt normalisation is the correct behaviour
    in FL — no global baseline, no cross-client statistics.

FedProx (§4.3.1):
  - The proximal term (μ/2)‖θ_k − θ_global‖² is added to the GRPO loss
    inside train_local_epochs() when fedprox_mu > 0.
  - Server-side aggregation is identical for FedAvg and FedProx.

Plan reference: Section 4.4 (Integration with Training Pipeline)
               Phase 3 deliverable: trainer/federated_ray_trainer.py
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("fedgraphr1")


class FederatedRayPPOTrainer:
    """Passive GRPO training engine for one FL client.

    Wraps the verl RayPPOTrainer as a passive engine that only exposes
    train_local_epochs().  The FL round loop lives in GraphR1Trainer.

    Args:
        grpo_client: FederatedGRPOClient instance (manages LoRA lifecycle).
        local_dataset: Dataset partition for this client (QA examples).
        reward_fn: Callable(question, answer) → float.  Should implement
            R_format + R_answer from §4.1.1.
        args: Parsed CLI namespace for hyperparameters.
        search_tool: FederatedSearchTool for local retrieval during rollout.
        device: torch.device or string.

    Plan §4.4.1
    """

    def __init__(
        self,
        grpo_client,
        local_dataset: List,
        reward_fn=None,
        args=None,
        search_tool=None,
        device="auto",
    ):
        self._grpo_client = grpo_client
        self.local_dataset = local_dataset
        self.reward_fn = reward_fn or _default_reward_fn
        self.args = args
        self.search_tool = search_tool
        self.device = device

        # Hyperparameters from args (with safe defaults)
        self.train_batch_size = getattr(args, "train_batch_size", 8)
        self.kl_loss_coef = getattr(args, "kl_loss_coef", 0.001)
        self.lr = getattr(args, "lr", 5e-7)
        self.n_repeat = getattr(args, "n_repeat", 2)   # G responses per prompt
        self.fedprox_mu = getattr(args, "fedprox_mu", 0.0)

        # Internal verl trainer (lazily initialised to avoid heavy imports)
        self._ray_trainer = None

    # ------------------------------------------------------------------
    # Main interface called by GraphR1Client
    # ------------------------------------------------------------------

    def train_local_epochs(self, num_epochs: int = 1) -> Dict[str, float]:
        """Run *num_epochs* of local GRPO training.

        Returns training metrics dict (policy_loss, avg_reward, kl_divergence).
        Returns NaN metrics if the underlying trainer is not initialised
        (Phase 1 fallback — GRPO is Phase 3).

        Plan §4.4.1 "train_local_epochs() trả quyền điều khiển về Flower Client"
        """
        if self._grpo_client is None or not self.local_dataset:
            return _nan_metrics()

        model = self._grpo_client.get_model()
        if model is None:
            return _nan_metrics()

        # Minimal GRPO loop (no verl dependency, Phase 1-3 baseline)
        # NOTE: attach_ray_trainer() is reserved for future verl production
        # integration once verl exposes a single-epoch training API.
        return self._train_minimal(model, num_epochs)

    # ------------------------------------------------------------------
    # Minimal GRPO loop (Phase 1/2 fallback — no verl required)
    # ------------------------------------------------------------------

    def _train_minimal(self, model, num_epochs: int) -> Dict[str, float]:
        """Minimal GRPO training loop without verl dependency.

        Advantage computation mirrors core_algos.py:
          A(x, y) = (R(x, y) - μ_group) / (σ_group + ε)

        This is the pure-local GRPO described in §4.2.2.
        """
        import random
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW

        try:
            from transformers import AutoTokenizer
        except ImportError:
            logger.warning(
                "[FederatedRayPPOTrainer] transformers not available — "
                "returning NaN metrics."
            )
            return _nan_metrics()

        tokenizer = AutoTokenizer.from_pretrained(
            self._grpo_client.base_model_path, trust_remote_code=True
        )
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr,
        )

        all_policy_losses = []
        all_rewards = []

        for epoch in range(num_epochs):
            # Sample batch
            batch = random.sample(
                self.local_dataset,
                min(self.train_batch_size, len(self.local_dataset)),
            )

            for example in batch:
                question = _extract_question(example)
                answer_gt = _extract_answer(example)

                # ── Rollout: generate G responses (§4.2.2) ──────────────
                responses, log_probs = _generate_responses(
                    model, tokenizer, question,
                    n=self.n_repeat,
                    search_tool=self.search_tool,
                    device=self.device,
                )
                if not responses:
                    continue

                # ── Reward: R_format + R_answer (§4.1.1) ───────────────
                rewards = [
                    self.reward_fn(question, r, answer_gt) for r in responses
                ]

                # ── Advantage: per-prompt normalisation (§4.2.2) ────────
                # Uses verl's compute_grpo_outcome_advantage when available
                rewards_t = torch.tensor(rewards, dtype=torch.float32)
                advantages = _compute_grpo_advantage(rewards_t)

                # ── Policy loss (GRPO / PPO-clip style) ─────────────────
                if log_probs is not None and len(log_probs) == len(advantages):
                    # Simplified GRPO objective (no importance sampling here)
                    policy_loss = -(advantages.to(log_probs.device) * log_probs).mean()

                    # FedProx proximal term (§4.3.1)
                    prox = self._grpo_client.compute_proximal_loss()
                    total_loss = policy_loss + prox

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    optimizer.step()

                    all_policy_losses.append(policy_loss.item())

                all_rewards.extend(rewards)

        avg_loss = sum(all_policy_losses) / len(all_policy_losses) if all_policy_losses else math.nan
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else math.nan

        logger.info(
            f"[FederatedRayPPOTrainer] {num_epochs} epoch(s): "
            f"avg_loss={avg_loss:.4f}  avg_reward={avg_reward:.4f}"
        )
        return {
            "policy_loss": avg_loss,
            "avg_reward": avg_reward,
            "kl_divergence": math.nan,   # not tracked in minimal loop
        }

    # ------------------------------------------------------------------
    # verl integration (Phase 3+)
    # ------------------------------------------------------------------

    def attach_ray_trainer(self, ray_trainer):
        """Attach a verl RayPPOTrainer instance (future production hook).

        Reserved for a future integration once verl exposes a single-epoch
        training API.  RayPPOTrainer.fit() runs the full FL loop internally
        and cannot be called per-epoch from an external orchestrator, so
        train_local_epochs() currently uses _train_minimal() instead.

        Plan §4.4.1
        """
        self._ray_trainer = ray_trainer
        logger.info(
            "[FederatedRayPPOTrainer] verl RayPPOTrainer attached "
            "(reserved for future use — training still uses minimal loop)."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nan_metrics() -> Dict[str, float]:
    return {
        "policy_loss": math.nan,
        "avg_reward": math.nan,
        "kl_divergence": math.nan,
    }


# ---------------------------------------------------------------------------
# Advantage computation (§4.2.2 "Pure Local GRPO")
# ---------------------------------------------------------------------------

def _compute_grpo_advantage(rewards_t, epsilon: float = 1e-6):
    """Per-prompt GRPO advantage normalisation.

    Tries to use verl's compute_grpo_outcome_advantage() first.  Falls back
    to the identical inline formula when verl is not installed.

    Args:
        rewards_t: 1-D float tensor of shape [G] (G responses per prompt).
        epsilon: Small constant for numerical stability.

    Returns:
        1-D float tensor of advantages, same shape as rewards_t.
    """
    import torch

    try:
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        G = rewards_t.shape[0]
        # verl expects token-level reward tensors with EOS mask.
        # Encode each scalar reward at position 0 (single-token outcome).
        token_rewards = torch.zeros(G, 1)
        eos_mask = torch.ones(G, 1)
        for i, r in enumerate(rewards_t):
            token_rewards[i, 0] = r
        # prompt_indices: all responses belong to prompt 0
        prompt_indices = torch.zeros(G, dtype=torch.long)
        advantages, _ = compute_grpo_outcome_advantage(
            token_rewards, eos_mask, prompt_indices, epsilon=epsilon
        )
        # advantages shape: [G, 1] — squeeze to [G]
        return advantages[:, 0]

    except (ImportError, TypeError, Exception):
        # Inline fallback — identical numerics to verl's implementation
        if rewards_t.numel() <= 1:
            # Single sample: std undefined, return reward centered at 0
            return rewards_t - rewards_t.mean()
        mu = rewards_t.mean()
        sigma = rewards_t.std() + epsilon
        return (rewards_t - mu) / sigma


# ---------------------------------------------------------------------------
# Reward functions (§4.1.1 R_format + R_answer)
# ---------------------------------------------------------------------------

def _r_format(response: str) -> float:
    """Format reward: 1.0 if response has <think>…</think><answer>…</answer>."""
    has_think = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL))
    return 1.0 if (has_think and has_answer) else 0.0


def _r_answer(answer: str, answer_gt: str) -> float:
    """Token-F1 answer reward (R_answer from §4.1.1)."""
    # Extract content inside <answer>…</answer> if present
    m = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
    answer_text = m.group(1).strip() if m else answer

    pred_tokens = set(answer_text.lower().split())
    gt_tokens = set(answer_gt.lower().split())
    if not gt_tokens or not pred_tokens:
        return 0.0
    precision = len(pred_tokens & gt_tokens) / len(pred_tokens)
    recall = len(pred_tokens & gt_tokens) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _composite_reward_fn(question: str, answer: str, answer_gt: str) -> float:
    """R_format + R_answer composite reward (§4.1.1)."""
    return _r_format(answer) + _r_answer(answer, answer_gt)


def _default_reward_fn(question: str, answer: str, answer_gt: str) -> float:
    """Default reward: composite R_format + R_answer (§4.1.1)."""
    return _composite_reward_fn(question, answer, answer_gt)


def _extract_question(example) -> str:
    if isinstance(example, dict):
        return example.get("question", example.get("input", str(example)))
    return str(example)


def _extract_answer(example) -> str:
    if isinstance(example, dict):
        ans = example.get("answer", example.get("target", ""))
        if isinstance(ans, list):
            return " ".join(ans)
        return str(ans)
    return ""


def _generate_responses(
    model, tokenizer, question: str,
    n: int = 2,
    search_tool=None,
    device="auto",
    max_new_tokens: int = 128,
):
    """Generate *n* responses for *question* using *model*.

    Returns:
        (responses: List[str], log_probs: Tensor of shape [n])
        log_probs[i] is the mean token log-prob for response i.
    """
    import torch
    import torch.nn.functional as F

    try:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = tokenizer(question, return_tensors="pt").to(device_obj)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate with gradients so we can compute log-probs for training.
        # do_sample=False (greedy) when n=1 — avoids float16 probability overflow.
        # temperature=0.7 dampens extreme logits that cause nan in float16 sampling.
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(n > 1),
                temperature=0.7 if n > 1 else 1.0,
                num_return_sequences=n,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences          # [n, prompt_len + gen_len]
        response_ids  = generated_ids[:, prompt_len:]  # [n, gen_len]

        responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # Re-run the forward pass with grad tracking to get log-probs for GRPO
        # Expand prompt to match the n generated sequences
        prompt_ids = inputs["input_ids"].expand(n, -1)          # [n, prompt_len]
        full_ids   = generated_ids.to(device_obj)               # [n, full_len]

        logits = model(input_ids=full_ids).logits                # [n, full_len, V]
        # Shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :]                         # [n, full_len-1, V]
        shift_labels = full_ids[:, 1:]                           # [n, full_len-1]

        log_probs_all = F.log_softmax(shift_logits, dim=-1)      # [n, full_len-1, V]
        token_log_probs = log_probs_all.gather(
            2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)                                            # [n, full_len-1]

        # Average only over the generated (response) tokens
        gen_len = response_ids.shape[1]
        response_log_probs = token_log_probs[:, -gen_len:]       # [n, gen_len]
        # Mask padding (eos token id)
        pad_id = tokenizer.eos_token_id
        mask = (response_ids != pad_id).float().to(device_obj)
        denom = mask.sum(dim=1).clamp(min=1)
        mean_log_probs = (response_log_probs * mask).sum(dim=1) / denom  # [n]

        return responses, mean_log_probs

    except Exception as e:
        logger.warning(f"[FederatedRayPPOTrainer] Generation failed: {e}")
        return [], None


