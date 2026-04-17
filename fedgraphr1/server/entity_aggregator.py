"""
fedgraphr1/server/entity_aggregator.py
========================================
Server-side entity deduplication and merging across FL clients.

Implements §2.2.1 (Entity Aggregation Engine):
  1. EntityDeduplicator — exact + fuzzy dedup using FAISS ANN + Union-Find
  2. federated_merge_entities() — heuristic merge (prototype) or LLM merge
  3. federated_merge_hyperedges() — aggregate hyperedges from all clients

Design choices (§2.2.1 notes):
  - Fuzzy matching uses FAISS IndexFlatIP + Union-Find clustering (O(n·k)
    instead of O(n²) pairwise comparison).
  - Entity merge conflict resolution uses heuristic (longest description)
    for prototype speed.  LLM-based summarisation is available as opt-in
    (use_llm_summarize=True) for production.
  - Reuses _merge_nodes_then_upsert and _merge_hyperedges_then_upsert from
    graphr1/operate.py where possible, wrapping with sync adapters.

Plan reference: Section 2.2.1 (Entity Aggregation Engine)
               Phase 1 deliverable: server/entity_aggregator.py
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np

from fedgraphr1.shared_types import (
    ClientExtractionResult,
    ExtractedEdgeRecord,
    ExtractedEntityRecord,
    ExtractedHyperedgeRecord,
)

logger = logging.getLogger("fedgraphr1")


# ---------------------------------------------------------------------------
# 1. Entity Deduplicator
# ---------------------------------------------------------------------------


class EntityDeduplicator:
    """Deduplicate entities received from multiple clients.

    Two-phase deduplication (§2.2.1):
      Phase 1: Exact name match (case-insensitive, strip quotes).
      Phase 2: Fuzzy semantic match via FAISS ANN + Union-Find clustering.

    Args:
        similarity_threshold: Cosine similarity threshold for fuzzy merge.
            Default 0.85 from plan §2.2.1.
        embedding_model: FlagEmbedding model for computing description
            embeddings.  If None, only exact-name dedup is performed.

    Plan §2.2.1
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        embedding_model=None,
    ):
        self.threshold = similarity_threshold
        self.embedding_model = embedding_model

    def deduplicate(
        self,
        all_results: List[ClientExtractionResult],
    ) -> Dict[str, List[ExtractedEntityRecord]]:
        """Deduplicate entities from *all_results*.

        Returns:
            Dict mapping canonical entity name → list of records (merged).
        """
        # ── Phase 1: Exact name grouping ───────────────────────────────
        maybe_nodes: Dict[str, List[ExtractedEntityRecord]] = defaultdict(list)
        for result in all_results:
            for entity in result.entities:
                canonical = _canonical_name(entity.entity_name)
                maybe_nodes[canonical].append(entity)

        if len(maybe_nodes) <= 1 or self.embedding_model is None:
            return dict(maybe_nodes)

        # ── Phase 2: Fuzzy matching via FAISS ANN + Union-Find ──────────
        try:
            import faiss
        except ImportError:
            logger.warning(
                "[EntityDeduplicator] faiss not installed — "
                "skipping fuzzy deduplication."
            )
            return dict(maybe_nodes)

        entity_names = list(maybe_nodes.keys())
        descriptions = [
            maybe_nodes[name][0].description for name in entity_names
        ]
        embeddings = self.embedding_model.encode_corpus(descriptions)
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        k = min(5, len(entity_names))
        distances, indices = index.search(embeddings, k)

        # Union-Find merge
        parent = list(range(len(entity_names)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            for dist, j in zip(dists[1:], idxs[1:]):
                if j >= 0 and dist > self.threshold:
                    union(i, j)

        # Group by cluster root
        clusters: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(entity_names)):
            clusters[find(i)].append(i)

        merged: Dict[str, List[ExtractedEntityRecord]] = {}
        for root_idx, member_idxs in clusters.items():
            canonical = entity_names[root_idx]  # first (root) = canonical
            merged[canonical] = []
            for idx in member_idxs:
                merged[canonical].extend(maybe_nodes[entity_names[idx]])

        logger.info(
            f"[EntityDeduplicator] "
            f"{len(maybe_nodes)} → {len(merged)} entities after fuzzy dedup"
        )
        return merged


# ---------------------------------------------------------------------------
# 2. Server-side entity merge
# ---------------------------------------------------------------------------


async def federated_merge_entities(
    deduped_nodes: Dict[str, List[ExtractedEntityRecord]],
    global_kg,
    global_config: dict,
    use_llm_summarize: bool = False,
):
    """Merge deduplicated entity groups into the global KG.

    Conflict resolution (§2.2.1, Step 3):
      - entity_type:  majority voting
      - description:  longest-wins heuristic (prototype) or LLM summarise
      - weight:       sum across all records
      - source_id:    concatenate all unique source hashes

    Args:
        deduped_nodes: Output of EntityDeduplicator.deduplicate().
        global_kg: Server-side NetworkXStorage instance (global KG).
        global_config: graphr1 global_config dict (needed for LLM path).
        use_llm_summarize: If True, call LLM to merge conflicting descriptions.
            Expensive — use only in production.  Default False (prototype).

    Plan §2.2.1 (Step 3)
    """
    from graphr1.prompt import GRAPH_FIELD_SEP

    async def _merge_one(entity_name: str, records: List[ExtractedEntityRecord]):
        if use_llm_summarize:
            # Production path: delegate to graphr1's LLM-based merger
            from graphr1.operate import _merge_nodes_then_upsert
            await _merge_nodes_then_upsert(
                entity_name,
                [_record_to_node_dict(r) for r in records],
                global_kg,
                global_config,
            )
        else:
            # Prototype heuristic: longest description wins
            merged_desc = max((r.description for r in records), key=len)
            type_counts = Counter(r.entity_type for r in records)
            merged_type = type_counts.most_common(1)[0][0]
            merged_weight = sum(r.weight for r in records)
            source_id = GRAPH_FIELD_SEP.join(
                sorted({r.source_chunk_hash for r in records})
            )
            await global_kg.upsert_node(
                entity_name,
                node_data={
                    "role": "entity",
                    "entity_type": merged_type,
                    "description": merged_desc,
                    "weight": merged_weight,
                    "source_id": source_id,
                },
            )

    tasks = [_merge_one(name, recs) for name, recs in deduped_nodes.items()]
    await asyncio.gather(*tasks)
    logger.info(
        f"[federated_merge_entities] Merged {len(deduped_nodes)} entities."
    )


# ---------------------------------------------------------------------------
# 3. Server-side hyperedge merge
# ---------------------------------------------------------------------------


async def federated_merge_hyperedges(
    all_results: List[ClientExtractionResult],
    global_kg,
    global_config: dict,
):
    """Aggregate and merge hyperedges from all client extraction results.

    Reuses _merge_hyperedges_then_upsert from graphr1/operate.py
    (§2.2.1 Step 2: "tái sử dụng logic _merge_hyperedges_then_upsert").

    Plan §2.2.1 (Step 2)
    """
    from graphr1.operate import _merge_hyperedges_then_upsert

    maybe_edges: Dict[str, list] = defaultdict(list)
    for result in all_results:
        for hyperedge in result.hyperedges:
            maybe_edges[hyperedge.hyperedge_name].append({
                "hyper_relation": hyperedge.hyperedge_name,
                "weight": hyperedge.weight,
                "source_id": f"client_{result.client_id}",
            })

    tasks = [
        _merge_hyperedges_then_upsert(he_name, he_records, global_kg, global_config)
        for he_name, he_records in maybe_edges.items()
    ]
    await asyncio.gather(*tasks)
    logger.info(
        f"[federated_merge_hyperedges] Merged {len(maybe_edges)} hyperedges."
    )


# ---------------------------------------------------------------------------
# 4. Edge (hyperedge → entity) merge
# ---------------------------------------------------------------------------


async def federated_merge_edges(
    all_results: List[ClientExtractionResult],
    global_kg,
):
    """Insert hyperedge→entity edges into the global KG.

    Unlike nodes, edges are additive — we upsert with weight summation.

    Plan §2.1.1 (ExtractedEdgeRecord)
    """
    from graphr1.prompt import GRAPH_FIELD_SEP

    edge_groups: Dict[str, list] = defaultdict(list)
    for result in all_results:
        for edge in result.edges:
            key = f"{edge.src_id}|||{edge.tgt_id}"
            edge_groups[key].append(edge)

    for key, edge_list in edge_groups.items():
        src_id, tgt_id = key.split("|||", 1)
        merged_weight = sum(e.weight for e in edge_list)
        source_id = GRAPH_FIELD_SEP.join(
            sorted({e.source_chunk_hash for e in edge_list})
        )
        await global_kg.upsert_edge(
            src_id,
            tgt_id,
            edge_data={"weight": merged_weight, "source_id": source_id},
        )

    logger.info(
        f"[federated_merge_edges] Merged {len(edge_groups)} unique edges."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_name(entity_name: str) -> str:
    """Normalise entity name: uppercase, strip outer quotes."""
    return '"' + entity_name.upper().strip('"').strip("'").strip() + '"'


def _record_to_node_dict(record: ExtractedEntityRecord) -> dict:
    """Convert ExtractedEntityRecord to the dict format expected by operate.py."""
    return {
        "entity_name": record.entity_name,
        "entity_type": record.entity_type,
        "description": record.description,
        "weight": record.weight,
        "source_id": record.source_chunk_hash,
        "hyper_relation": "",
    }
