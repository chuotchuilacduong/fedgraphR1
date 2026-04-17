"""
fedgraphr1/server/kg_diff.py
=============================
KG Fingerprint and Delta Detection for bandwidth-efficient broadcast.

The server computes a fingerprint of G_H^global before and after each
round's aggregation step.  If the fingerprint is unchanged the server
skips the fragment broadcast entirely — saving the full KG serialisation
overhead for that round.

Fingerprint design (fast, stable):
  - entity count, hyperedge count, edge count
  - MD5 of sorted entity node names (detects new/removed entities)

A fingerprint change triggers a full fragment broadcast; no change →
clients reuse their cached local KG.

The KGDelta dataclass is logged to W&B so users can see per-round KG
growth (the paper's main contribution: KG improves each round).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("fedgraphr1")

# Type alias: (entity_count, hyperedge_count, edge_count, name_hash)
KGFingerprint = Tuple[int, int, int, str]

_EMPTY_FP: KGFingerprint = (0, 0, 0, "")


@dataclass
class KGDelta:
    """Change statistics between two consecutive KG snapshots.

    Attributes:
        prev_entities:       Entity count before this round's aggregation.
        curr_entities:       Entity count after this round's aggregation.
        prev_hyperedges:     Hyperedge count before.
        curr_hyperedges:     Hyperedge count after.
        prev_edges:          Edge count before.
        curr_edges:          Edge count after.
        new_entities:        Entities added this round.
        new_hyperedges:      Hyperedges added this round.
        new_edges:           Edges added this round.
        changed:             True if any structural change was detected.
        broadcast_skipped:   Set to True by the server when broadcast is skipped.
    """

    prev_entities: int = 0
    curr_entities: int = 0
    prev_hyperedges: int = 0
    curr_hyperedges: int = 0
    prev_edges: int = 0
    curr_edges: int = 0
    new_entities: int = 0
    new_hyperedges: int = 0
    new_edges: int = 0
    changed: bool = False
    broadcast_skipped: bool = False

    def as_log_dict(self) -> dict:
        """Return dict suitable for W&B / debug logging."""
        return {
            "kg/prev_entities":    self.prev_entities,
            "kg/curr_entities":    self.curr_entities,
            "kg/prev_hyperedges":  self.prev_hyperedges,
            "kg/curr_hyperedges":  self.curr_hyperedges,
            "kg/new_entities":     self.new_entities,
            "kg/new_hyperedges":   self.new_hyperedges,
            "kg/new_edges":        self.new_edges,
            "kg/changed":          int(self.changed),
            "kg/broadcast_skipped": int(self.broadcast_skipped),
        }

    def summary(self) -> str:
        if not self.changed:
            return "KG unchanged — broadcast skipped"
        parts = []
        if self.new_entities:
            parts.append(f"+{self.new_entities} entities")
        if self.new_hyperedges:
            parts.append(f"+{self.new_hyperedges} hyperedges")
        if self.new_edges:
            parts.append(f"+{self.new_edges} edges")
        label = ", ".join(parts) if parts else "structural change"
        return f"KG updated ({label}) → broadcast triggered"


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------


def compute_kg_fingerprint(graph) -> KGFingerprint:
    """Compute a fast, stable fingerprint of a NetworkX knowledge graph.

    The fingerprint captures structural identity: adding or removing any
    entity/hyperedge node or edge will change it.  Pure weight/description
    changes to existing nodes do NOT change the fingerprint (weight updates
    don't require re-sending the full graph structure).

    Args:
        graph: nx.Graph representing G_H^global (may be None for an empty KG).

    Returns:
        Tuple (entity_count, hyperedge_count, edge_count, name_hash).
    """
    if graph is None:
        return _EMPTY_FP

    entity_nodes = []
    hyperedge_nodes = []
    for node, data in graph.nodes(data=True):
        role = data.get("role", "")
        if role == "hyperedge":
            hyperedge_nodes.append(str(node))
        else:
            entity_nodes.append(str(node))

    edge_count = graph.number_of_edges()

    # Hash sorted entity names for O(1) comparison in subsequent rounds
    name_str = "|".join(sorted(entity_nodes)).encode("utf-8")
    name_hash = hashlib.md5(name_str).hexdigest()[:12]

    return (len(entity_nodes), len(hyperedge_nodes), edge_count, name_hash)


def compute_kg_delta(
    old_fp: Optional[KGFingerprint],
    new_fp: KGFingerprint,
) -> KGDelta:
    """Compute the structural delta between two KG fingerprints.

    Args:
        old_fp: Fingerprint from the previous round (None for round 0).
        new_fp: Fingerprint from the current round.

    Returns:
        KGDelta with change statistics and a `changed` flag.
    """
    if old_fp is None:
        old_fp = _EMPTY_FP

    old_e, old_h, old_edges, old_hash = old_fp
    new_e, new_h, new_edges, new_hash = new_fp

    changed = (old_fp != new_fp)

    delta = KGDelta(
        prev_entities=old_e,
        curr_entities=new_e,
        prev_hyperedges=old_h,
        curr_hyperedges=new_h,
        prev_edges=old_edges,
        curr_edges=new_edges,
        new_entities=max(0, new_e - old_e),
        new_hyperedges=max(0, new_h - old_h),
        new_edges=max(0, new_edges - old_edges),
        changed=changed,
    )

    logger.info(f"[KGDiff] {delta.summary()}")
    return delta
