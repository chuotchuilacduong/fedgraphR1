"""
fedgraphr1/utils/metrics.py
============================
FL-specific evaluation and monitoring metrics.

Tracks:
  - Per-round reward statistics from GRPO training
  - Knowledge Hypergraph quality (entity/hyperedge counts, density)
  - Communication cost (bandwidth consumed per round)
  - Retrieval quality proxies (answer F1, retrieval recall)

Plan reference: Section 5 (Roadmap — Validation metrics)
               Section 4.1.1 (Reward function R_format + R_answer)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("fedgraphr1")


# ---------------------------------------------------------------------------
# Per-round metric containers
# ---------------------------------------------------------------------------


@dataclass
class RoundMetrics:
    """Aggregated metrics for one FL round.

    Populated by GraphR1Trainer after each round and passed to the logger.
    """

    round_id: int

    # ── GRPO reward stats (averaged across sampled clients) ──────────────
    avg_reward: float = math.nan
    avg_format_reward: float = math.nan
    avg_answer_f1: float = math.nan

    # ── LoRA training stats ──────────────────────────────────────────────
    avg_policy_loss: float = math.nan
    avg_kl_divergence: float = math.nan

    # ── Hypergraph quality (server-side) ─────────────────────────────────
    global_num_entities: int = 0
    global_num_hyperedges: int = 0
    global_num_edges: int = 0
    global_graph_density: float = math.nan

    # ── Communication cost ───────────────────────────────────────────────
    total_uplink_bytes: int = 0    # sum across sampled clients
    total_downlink_bytes: int = 0  # sum across sampled clients

    # ── Per-client metrics (for logging detail) ───────────────────────────
    client_metrics: Dict[str, Dict] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary string for console output."""
        return (
            f"[Round {self.round_id}] "
            f"avg_reward={self.avg_reward:.4f}  "
            f"avg_f1={self.avg_answer_f1:.4f}  "
            f"KG_entities={self.global_num_entities}  "
            f"KG_hyperedges={self.global_num_hyperedges}"
        )


# ---------------------------------------------------------------------------
# Hypergraph quality helpers
# ---------------------------------------------------------------------------


def compute_kg_density(num_nodes: int, num_edges: int) -> float:
    """Graph density = |E| / (|V| * (|V| - 1)) for directed graph.

    Returns 0.0 for graphs with fewer than 2 nodes.
    """
    if num_nodes < 2:
        return 0.0
    return num_edges / (num_nodes * (num_nodes - 1))


def compute_kg_stats(graph) -> Dict:
    """Compute quality statistics for a NetworkX graph.

    Args:
        graph: nx.Graph or nx.DiGraph instance (the global Knowledge Hypergraph).

    Returns:
        Dict with keys: num_nodes, num_edges, num_entity_nodes,
        num_hyperedge_nodes, density.
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    entity_nodes = [
        n for n, d in graph.nodes(data=True) if d.get("role") == "entity"
    ]
    hyperedge_nodes = [
        n for n, d in graph.nodes(data=True) if d.get("role") == "hyperedge"
    ]

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_entity_nodes": len(entity_nodes),
        "num_hyperedge_nodes": len(hyperedge_nodes),
        "density": compute_kg_density(num_nodes, num_edges),
    }


# ---------------------------------------------------------------------------
# Reward aggregation helpers
# ---------------------------------------------------------------------------


def aggregate_client_reward_stats(
    per_client_metrics: Dict[str, Dict],
) -> Dict[str, float]:
    """Compute FL-round-level reward stats by averaging across clients.

    Args:
        per_client_metrics: {client_id: {"avg_reward": ..., "avg_f1": ..., ...}}

    Returns:
        Dict with "avg_reward", "avg_format_reward", "avg_answer_f1" keys.
    """
    keys = ["avg_reward", "avg_format_reward", "avg_answer_f1",
            "avg_policy_loss", "avg_kl_divergence"]
    result = {}
    for k in keys:
        vals = [
            m[k] for m in per_client_metrics.values()
            if k in m and not math.isnan(m[k])
        ]
        result[k] = sum(vals) / len(vals) if vals else math.nan
    return result


# ---------------------------------------------------------------------------
# Communication cost tracker
# ---------------------------------------------------------------------------


class BandwidthTracker:
    """Tracks bytes sent/received per round for bandwidth profiling.

    Plan §3.1.2 (Bandwidth estimates), §4.3.2 (Communication Cost Analysis)
    """

    def __init__(self):
        self._uplink: Dict[int, int] = defaultdict(int)    # round → bytes
        self._downlink: Dict[int, int] = defaultdict(int)

    def record_uplink(self, round_id: int, num_bytes: int):
        """Record *num_bytes* sent from a client to the server."""
        self._uplink[round_id] += num_bytes

    def record_downlink(self, round_id: int, num_bytes: int):
        """Record *num_bytes* sent from the server to a client."""
        self._downlink[round_id] += num_bytes

    def summary(self, round_id: int) -> str:
        up_mb = self._uplink[round_id] / (1024 ** 2)
        down_mb = self._downlink[round_id] / (1024 ** 2)
        return (
            f"Round {round_id} bandwidth — "
            f"uplink: {up_mb:.2f} MB  downlink: {down_mb:.2f} MB"
        )

    def total_uplink(self, round_id: int) -> int:
        return self._uplink[round_id]

    def total_downlink(self, round_id: int) -> int:
        return self._downlink[round_id]
