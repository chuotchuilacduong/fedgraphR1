"""
fedgraphr1/fl/trainer.py
==========================
GraphR1Trainer — the FL round-loop orchestrator for Federated Graph-R1.

Workflow (the paper's key contribution: KG improves each FL round):

  Round 0  ── KG Initialisation
    • ALL clients run entity extraction (no GRPO — model not yet improved)
    • Server aggregates → G_H^global built for the first time
    • Server broadcasts full KG fragments to all clients

  Round r ≥ 1  ── GRPO + KG Refinement  (iterated)
    ┌─ Downlink ──────────────────────────────────────────────────────
    │  KG changed last round → broadcast updated fragments + LoRA
    │  KG unchanged          → broadcast LoRA only  (bandwidth saving)
    ├─ Client local work ────────────────────────────────────────────
    │  a. Load updated KG fragment (skip if kg_unchanged)
    │  b. GRPO training — π_θ trains on QA examples using growing KG
    │  c. Extract next document batch (π_ext fixed, not affected by GRPO)
    │  d. Upload (LoRA delta + extraction result)
    ├─ Server aggregation ───────────────────────────────────────────
    │  a. Merge extraction results → G_H^global grows/refines
    │  b. FedAvg LoRA → global model improves
    │  c. Compute KGDelta; rebuild partitioner only if KG changed
    │  d. Flush KV stores to disk
    └─ Validation ───────────────────────────────────────────────────
       Retrieval Recall@K + EM/F1 against dev set using current KG
       (no GPT judge — fast, runs every eval_freq rounds)

What actually improves per round:
  KG coverage grows because more documents are processed each round
  (pre-determined batching schedule, not model quality feedback).
  π_θ QA performance improves as GRPO trains on increasingly complete KG context.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Dict, List, Optional

from fedgraphr1.fl.base import BaseClient, BaseServer
from fedgraphr1.utils.debug_logger import step, timed_step

logger = logging.getLogger("fedgraphr1")


class GraphR1Trainer:
    """Orchestrates the federated learning round loop for Graph-R1.

    Args:
        args:       Parsed CLI namespace.
        clients:    List of GraphR1Client instances (pre-instantiated).
        server:     GraphR1Server instance.
        dev_data:   Optional list of dev QA dicts for per-round evaluation.
                    If None, validation is skipped.
        eval_freq:  Run validation every this many rounds (default: 1).
    """

    def __init__(
        self,
        args,
        clients: List[BaseClient],
        server: BaseServer,
        dev_data: Optional[List[dict]] = None,
        eval_freq: int = 1,
    ):
        self.args = args
        self.clients = clients
        self.server = server
        self.message_pool = server.message_pool
        self.dev_data = dev_data or []
        self.eval_freq = eval_freq

        self._best_metrics: Dict = {"best_round": 0}
        self._round_history: List[Dict] = []
        self._wandb_enabled = self._check_wandb()

        # Per-round evaluator — lazy init on first validation call
        self._evaluator = None

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """Run the full FL loop for args.num_rounds rounds.

        Round 0 is a KG-initialisation-only round (no GRPO).
        Rounds 1+ run the full GRPO + KG refinement cycle.
        """
        num_rounds = getattr(self.args, "num_rounds", 40)
        num_clients = len(self.clients)
        client_frac = getattr(self.args, "client_frac", 1.0)

        logger.info(
            f"[GraphR1Trainer] Starting FL: "
            f"{num_rounds} rounds, {num_clients} clients, frac={client_frac}"
        )
        self.server.initialize()

        for round_id in range(num_rounds):
            round_start = time.time()

            # ── Sample clients ──────────────────────────────────────────
            k = max(1, int(num_clients * client_frac))
            sampled = sorted(random.sample(range(num_clients), k))
            self.message_pool["round"] = round_id
            self.message_pool["sampled_clients"] = sampled
            # Round 0 flag — clients read this to skip GRPO on init round.
            # When using a pre-built KG (--pretrained_kg_dir), the KG is already
            # loaded on all clients before training starts, so all rounds including
            # round 0 run GRPO training (no extraction-only init round needed).
            using_pretrained_kg = bool(getattr(self.args, "pretrained_kg_dir", None))
            self.message_pool["kg_init_round"] = (round_id == 0) and not using_pretrained_kg

            print(f"\n{'═'*60}")
            is_init = self.message_pool.get("kg_init_round", False)
            print(
                f"  Round {round_id}  │  clients: {sampled}"
                + ("  [KG INIT — extraction only]" if is_init else "")
            )
            print(f"{'═'*60}")

            # ── [1] Downlink: server → clients ──────────────────────────
            step(f"Round {round_id} │ [1] downlink")
            with timed_step(f"Round {round_id} │ server.send_message"):
                self.server.send_message()

            # ── [2] Client local work ───────────────────────────────────
            step(f"Round {round_id} │ [2] client execution")
            client_metrics_list: List[Dict] = []
            lora_densities: List[float] = []

            for cid in sampled:
                with timed_step(f"Round {round_id} │ client_{cid}.execute"):
                    self.clients[cid].execute()
                with timed_step(f"Round {round_id} │ client_{cid}.send_message"):
                    self.clients[cid].send_message()

                msg = self.message_pool.get(f"client_{cid}", {})
                if "metrics" in msg:
                    client_metrics_list.append(msg["metrics"])
                if msg.get("lora_density") is not None:
                    lora_densities.append(msg["lora_density"])

            # ── [3] Server aggregation ──────────────────────────────────
            step(f"Round {round_id} │ [3] server aggregation")
            with timed_step(f"Round {round_id} │ server.execute"):
                self.server.execute()

            kg_delta = self.server.get_last_kg_delta()

            # ── [4] Validation (standard metrics, no GPT judge) ─────────
            val_metrics: Dict = {}
            if self.dev_data and (round_id % self.eval_freq == 0):
                step(f"Round {round_id} │ [4] validation")
                val_metrics = self._run_validation(round_id)

            # ── Aggregate, log, record ──────────────────────────────────
            avg_metrics = _average_metrics(client_metrics_list)
            kg_stats    = self.server.get_kg_stats()
            round_time  = time.time() - round_start

            self._log_round(
                round_id, avg_metrics, kg_stats, kg_delta,
                val_metrics, round_time,
            )

            self._round_history.append({
                "round": round_id,
                **avg_metrics,
                **{f"kg_{k}": v for k, v in kg_stats.items()},
                **val_metrics,
                "round_time_s": round_time,
            })

            if self._wandb_enabled:
                self._wandb_log(
                    round_id, avg_metrics, kg_stats, kg_delta,
                    val_metrics, round_time, lora_densities,
                )

        logger.info("[GraphR1Trainer] Training complete.")
        return self._round_history

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validation(self, round_id: int) -> Dict:
        """Evaluate KG quality and QA performance.

        Uses RoundEvaluator which computes:
          - Retrieval Recall@K  (KG quality proxy, no model needed)
          - EM / F1             (context-level, no model inference needed)
        """
        if self._evaluator is None:
            self._evaluator = self._init_evaluator()
        if self._evaluator is None:
            return {}

        # Reload FAISS indices from the server's freshly flushed KV stores
        self._evaluator.reload()

        try:
            return self._evaluator.evaluate(round_id)
        except Exception as e:
            logger.warning(
                f"[GraphR1Trainer] Validation failed (round {round_id}): {e}"
            )
            return {}

    def _init_evaluator(self):
        """Create RoundEvaluator pointing at the KG directory.

        Uses --pretrained_kg_dir when set (pre-built KG); otherwise falls back
        to the server's working_dir (built during the FL loop).
        """
        if not self.dev_data:
            return None
        try:
            from fedgraphr1.eval.round_evaluator import RoundEvaluator
            kg_dir = (
                getattr(self.args, "pretrained_kg_dir", None)
                or getattr(self.server, "_working_dir", None)
                or getattr(self.args, "working_dir", "expr/fedgraphr1")
            )
            return RoundEvaluator(
                dev_data=self.dev_data,
                kg_dir=kg_dir,
                top_k=getattr(self.args, "eval_top_k", 10),
                max_eval_samples=getattr(self.args, "max_eval_samples", 200),
            )
        except Exception as e:
            logger.warning(f"[GraphR1Trainer] Could not init RoundEvaluator: {e}")
            return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_round(
        self,
        round_id: int,
        avg_metrics: Dict,
        kg_stats: Dict,
        kg_delta,
        val_metrics: Dict,
        elapsed: float,
    ):
        is_init = (round_id == 0)
        # Client stats
        train_line = (
            f"  CLIENTS (avg):  "
            f"entities={avg_metrics.get('num_entities','?')}  "
            f"hyperedges={avg_metrics.get('num_hyperedges','?')}"
        )
        if not is_init:
            train_line += (
                f"  loss={avg_metrics.get('policy_loss', float('nan')):.4f}"
                f"  reward={avg_metrics.get('avg_reward', float('nan')):.4f}"
            )
        else:
            train_line += "  [extraction only — no GRPO]"
        print(train_line)

        # KG delta
        kg_delta_str = ""
        if kg_delta is not None:
            if kg_delta.changed:
                kg_delta_str = (
                    f"+{kg_delta.new_entities} ent, "
                    f"+{kg_delta.new_hyperedges} he, "
                    f"+{kg_delta.new_edges} edges  → broadcast"
                )
            else:
                kg_delta_str = "no change  → broadcast skipped"
        print(
            f"  SERVER KG:      "
            f"entities={kg_stats.get('num_entity_nodes','?')}  "
            f"hyperedges={kg_stats.get('num_hyperedge_nodes','?')}  "
            f"| {kg_delta_str}"
        )

        # Validation
        if val_metrics:
            print(
                f"  VALIDATION:     "
                f"recall@K={val_metrics.get('val/recall_at_k', float('nan')):.4f}  "
                f"EM={val_metrics.get('val/em', float('nan')):.4f}  "
                f"F1={val_metrics.get('val/f1', float('nan')):.4f}"
                f"  (n={val_metrics.get('val/num_samples','?')})"
            )

        print(f"  Round time: {elapsed:.1f}s")
        print("─" * 60)

    def _check_wandb(self) -> bool:
        try:
            import wandb
            return wandb.run is not None
        except Exception:
            return False

    def _wandb_log(
        self,
        round_id: int,
        avg_metrics: Dict,
        kg_stats: Dict,
        kg_delta,
        val_metrics: Dict,
        round_time: float,
        lora_densities: List[float],
    ):
        try:
            import wandb

            record = {
                "round": round_id,
                # GRPO training (NaN on round 0)
                "train/policy_loss":    avg_metrics.get("policy_loss",    float("nan")),
                "train/avg_reward":     avg_metrics.get("avg_reward",     float("nan")),
                "train/kl_divergence":  avg_metrics.get("kl_divergence",  float("nan")),
                "train/num_entities":   avg_metrics.get("num_entities",   0),
                "train/num_hyperedges": avg_metrics.get("num_hyperedges", 0),
                # KG cumulative state
                "kg/entity_nodes":    kg_stats.get("num_entity_nodes",   0),
                "kg/hyperedge_nodes": kg_stats.get("num_hyperedge_nodes", 0),
                "kg/total_nodes": (
                    kg_stats.get("num_entity_nodes", 0)
                    + kg_stats.get("num_hyperedge_nodes", 0)
                ),
                "kg/density": kg_stats.get("density", 0.0),
                "kg/edges":   kg_stats.get("num_edges", 0),
                # Per-round KG delta — shows the paper's contribution directly
                **({} if kg_delta is None else kg_delta.as_log_dict()),
                # Validation (no GPT judge)
                **val_metrics,
                # System
                "system/round_time_s": round_time,
            }

            if lora_densities:
                avg_d = sum(lora_densities) / len(lora_densities)
                record["lora/avg_density"]  = avg_d
                record["lora/avg_sparsity"] = 1.0 - avg_d

            wandb.log(record, step=round_id)
        except Exception as e:
            logger.debug(f"[GraphR1Trainer] wandb.log failed: {e}")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_history(self) -> List[Dict]:
        return self._round_history

    def save_checkpoint(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        lora_server = getattr(self.server, "_lora_server", None)
        if lora_server is not None:
            lora_server.save(os.path.join(path, "global_lora.pt"))
        logger.info(f"[GraphR1Trainer] Checkpoint → {path}")

    def load_checkpoint(self, path: str):
        import os
        lora_path = os.path.join(path, "global_lora.pt")
        lora_server = getattr(self.server, "_lora_server", None)
        if lora_server is not None and os.path.exists(lora_path):
            lora_server.load(lora_path)
        logger.info(f"[GraphR1Trainer] Checkpoint loaded ← {path}")


# ---------------------------------------------------------------------------
# Helper for client.py to check kg_init_round flag
# ---------------------------------------------------------------------------

def is_kg_init_round(message_pool: dict) -> bool:
    """Return True when the current round is the KG initialisation round."""
    return message_pool.get("kg_init_round", False)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _average_metrics(metrics_list: List[Dict]) -> Dict:
    if not metrics_list:
        return {}
    result = {}
    all_keys = set(k for m in metrics_list for k in m)
    for key in all_keys:
        vals = [
            m[key] for m in metrics_list
            if key in m and not (isinstance(m[key], float) and math.isnan(m[key]))
        ]
        if vals:
            try:
                result[key] = sum(vals) / len(vals)
            except TypeError:
                result[key] = vals[0]
    return result
