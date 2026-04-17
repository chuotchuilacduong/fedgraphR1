"""
fedgraphr1/server/hypergraph_builder.py
==========================================
Server-side global Knowledge Hypergraph builder.

Orchestrates the full entity/hyperedge aggregation pipeline:
  1. Deserialise client packets (via EntityPacker.unpack)
  2. Deduplicate entities (EntityDeduplicator)
  3. Merge entities into the global KG NetworkXStorage
  4. Merge hyperedges into the global KG
  5. Merge edges (hyperedge → entity links)
  6. Persist the updated global KG

The server maintains a single persistent global KG across all FL rounds,
incrementally enriching it with each round's client contributions.

Plan reference: Section 2.2.1 (Hypergraph Engine — BUILDER)
               Phase 1 deliverable: server/hypergraph_builder.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict
from typing import Dict, List, Optional

from fedgraphr1.shared_types import ClientExtractionResult
from fedgraphr1.server.entity_aggregator import (
    EntityDeduplicator,
    federated_merge_entities,
    federated_merge_hyperedges,
    federated_merge_edges,
)

logger = logging.getLogger("fedgraphr1")


class GlobalHypergraphBuilder:
    """Incrementally builds G_H^global from client extraction results.

    The builder holds a reference to the server-side GraphR1 instance
    whose storage backends (NetworkXStorage, JsonKVStorage) form the
    global KG.

    Args:
        graphr1_instance: A GraphR1 dataclass instance initialised with the
            server's working_dir.  Its chunk_entity_relation_graph is used
            as the global KG.
        embedding_model: Optional FlagEmbedding model for fuzzy entity dedup.
            If None, only exact-name deduplication is performed.
        use_llm_summarize: Whether to use LLM to merge conflicting entity
            descriptions.  Expensive — leave False for prototype.
        dedup_threshold: Cosine similarity threshold for fuzzy entity dedup.

    Plan §2.2.1
    """

    def __init__(
        self,
        graphr1_instance,
        embedding_model=None,
        use_llm_summarize: bool = False,
        dedup_threshold: float = 0.85,
    ):
        self._graphr1 = graphr1_instance
        self.use_llm_summarize = use_llm_summarize
        self.deduplicator = EntityDeduplicator(
            similarity_threshold=dedup_threshold,
            embedding_model=embedding_model,
        )
        self._round_results: List[ClientExtractionResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        client_results: List[ClientExtractionResult],
        round_id: int = 0,
    ):
        """Integrate *client_results* from the current round into G_H^global.

        Args:
            client_results: List of extraction results from sampled clients.
            round_id: Current FL round number (for logging).

        Plan §2.2.1
        """
        loop = _get_or_create_event_loop()
        loop.run_until_complete(self.aupdate(client_results, round_id))

    async def aupdate(
        self,
        client_results: List[ClientExtractionResult],
        round_id: int = 0,
    ):
        """Async variant of update()."""
        if not client_results:
            logger.warning(f"[Round {round_id}] No client results to aggregate.")
            return

        g = self._graphr1
        from dataclasses import asdict as _asdict
        global_config = _asdict(g)

        logger.info(
            f"[Round {round_id}] Aggregating {len(client_results)} client results …"
        )

        # ── Step 1: Deduplicate entities ───────────────────────────────
        deduped_nodes = self.deduplicator.deduplicate(client_results)
        logger.info(
            f"[Round {round_id}] Deduped: {sum(len(r.entities) for r in client_results)} → "
            f"{len(deduped_nodes)} unique entities"
        )

        # ── Step 2: Merge entities into global KG ──────────────────────
        await federated_merge_entities(
            deduped_nodes,
            g.chunk_entity_relation_graph,
            global_config,
            use_llm_summarize=self.use_llm_summarize,
        )

        # ── Step 3: Merge hyperedges ───────────────────────────────────
        await federated_merge_hyperedges(
            client_results,
            g.chunk_entity_relation_graph,
            global_config,
        )

        # ── Step 4: Merge edges ────────────────────────────────────────
        await federated_merge_edges(
            client_results,
            g.chunk_entity_relation_graph,
        )

        # ── Step 5: Persist to disk ────────────────────────────────────
        await g.chunk_entity_relation_graph.index_done_callback()
        # Also persist KV stores
        for store in [g.entities_vdb, g.hyperedges_vdb]:
            if store is not None:
                await store.index_done_callback()

        # Track cumulative results for quality monitoring
        self._round_results.extend(client_results)

        kg = g.chunk_entity_relation_graph
        n_nodes = kg._graph.number_of_nodes() if hasattr(kg, "_graph") else "?"
        n_edges = kg._graph.number_of_edges() if hasattr(kg, "_graph") else "?"
        logger.info(
            f"[Round {round_id}] Global KG updated: "
            f"{n_nodes} nodes, {n_edges} edges"
        )

    def get_global_graph(self):
        """Return the underlying nx.Graph of the global KG.

        Used by HypergraphPartitioner to build per-client fragments.
        """
        kg = self._graphr1.chunk_entity_relation_graph
        if hasattr(kg, "_graph"):
            return kg._graph
        raise AttributeError(
            "NetworkXStorage has no _graph attribute — "
            "check graphr1/storage.py version."
        )

    def kg_stats(self) -> Dict:
        """Return quality statistics about the current global KG."""
        from fedgraphr1.utils.metrics import compute_kg_stats
        try:
            g = self.get_global_graph()
            return compute_kg_stats(g)
        except Exception as e:
            logger.warning(f"[HypergraphBuilder] Could not compute stats: {e}")
            return {}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
