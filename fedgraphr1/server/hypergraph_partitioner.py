"""
fedgraphr1/server/hypergraph_partitioner.py
=============================================
Server-side partitioning of the global Knowledge Hypergraph into
per-client fragments for distribution.

Two strategies (§2.2.2):
  A. full_broadcast — every client gets the full G_H^global (prototype).
  B. relevance_based — each client receives a subgraph centred around
     the entities it contributed, with k-hop expansion and top-K global
     high-centrality nodes added.

Plan reference: Section 2.2.2 (Hypergraph Partitioner)
               Phase 1/2 deliverable: server/hypergraph_partitioner.py
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import networkx as nx

from fedgraphr1.shared_types import (
    EntityMetadata,
    FragmentStats,
    HypergraphFragment,
)
from fedgraphr1.utils.metrics import compute_kg_stats

logger = logging.getLogger("fedgraphr1")


class HypergraphPartitioner:
    """Partition the global KG into client-specific fragments.

    Args:
        global_graph: The full global Knowledge Hypergraph (nx.Graph).
        strategy: "full_broadcast" or "relevance_based".
        expansion_hops: Number of hops to expand from client-contributed
            entities when using relevance_based strategy.
        top_k_global: Number of high-centrality global nodes to always
            include in relevance_based fragments.

    Plan §2.2.2
    """

    def __init__(
        self,
        global_graph: nx.Graph,
        strategy: str = "full_broadcast",
        expansion_hops: int = 1,
        top_k_global: int = 100,
    ):
        self.global_graph = global_graph
        self.strategy = strategy
        self.expansion_hops = expansion_hops
        self.top_k_global = top_k_global

        # Cache degree centrality (expensive to compute; recomputed when graph changes)
        self._cached_centrality: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def partition_for_client(
        self,
        client_id: str,
        client_entity_names: Optional[Set[str]] = None,
        round_number: int = 0,
    ) -> HypergraphFragment:
        """Build a HypergraphFragment for *client_id*.

        Args:
            client_id: String client identifier.
            client_entity_names: Set of entity names contributed by this
                client in the current (or previous) round.  Required for
                relevance_based strategy; ignored for full_broadcast.
            round_number: Current FL round.

        Returns:
            HypergraphFragment ready for the server → client downlink.

        Plan §2.2.2
        """
        if self.strategy == "full_broadcast":
            fragment_graph = self._full_broadcast()
        elif self.strategy == "relevance_based":
            if client_entity_names is None:
                logger.warning(
                    f"[Partitioner] relevance_based requested but no entity "
                    f"names provided for client {client_id}. "
                    "Falling back to full_broadcast."
                )
                fragment_graph = self._full_broadcast()
            else:
                fragment_graph = self._relevance_based(client_entity_names)
        else:
            raise ValueError(
                f"Unknown partitioning strategy '{self.strategy}'. "
                "Choose: 'full_broadcast' or 'relevance_based'."
            )

        # Build entity metadata list from fragment nodes
        entity_meta = _build_entity_metadata(fragment_graph)

        # Build stats
        stats_dict = compute_kg_stats(fragment_graph)
        global_nodes = self.global_graph.number_of_nodes()
        coverage = (
            fragment_graph.number_of_nodes() / global_nodes
            if global_nodes > 0 else 1.0
        )
        stats = FragmentStats(
            num_nodes=stats_dict["num_nodes"],
            num_edges=stats_dict["num_edges"],
            num_entity_nodes=stats_dict["num_entity_nodes"],
            num_hyperedge_nodes=stats_dict["num_hyperedge_nodes"],
            coverage_ratio=coverage,
        )

        logger.info(
            f"[Partitioner] Fragment for client {client_id} "
            f"({self.strategy}): "
            f"{stats.num_nodes} nodes, {stats.num_edges} edges, "
            f"coverage={coverage:.1%}"
        )

        return HypergraphFragment(
            client_id=client_id,
            round_number=round_number,
            graph_data=fragment_graph,   # nx.Graph in simulation
            entity_metadata=entity_meta,
            stats=stats,
            is_compressed=False,
        )

    def invalidate_centrality_cache(self):
        """Call after the global graph is updated to force recomputation."""
        self._cached_centrality = None

    # ------------------------------------------------------------------
    # Partitioning strategies
    # ------------------------------------------------------------------

    def _full_broadcast(self) -> nx.Graph:
        """Return a copy of the full global graph.

        Plan §2.2.2 Strategy A
        """
        return self.global_graph.copy()

    def _relevance_based(self, client_entities: Set[str]) -> nx.Graph:
        """Return a relevance-based subgraph for one client.

        Steps (§2.2.2 Strategy B):
          1. Core entities: nodes contributed by this client.
          2. K-hop expansion around core entities.
          3. Top-K global nodes by degree centrality.
          4. Return induced subgraph.

        Plan §2.2.2 Strategy B
        """
        selected_nodes: Set[str] = set()

        # 1. Core entities
        for entity in client_entities:
            if self.global_graph.has_node(entity):
                selected_nodes.add(entity)

        # 2. K-hop expansion
        frontier = set(selected_nodes)
        for _ in range(self.expansion_hops):
            new_nodes: Set[str] = set()
            for node in frontier:
                new_nodes.update(self.global_graph.neighbors(node))
            selected_nodes.update(new_nodes)
            frontier = new_nodes

        # 3. Top-K global nodes by degree centrality
        centrality = self._get_centrality()
        top_global = sorted(
            centrality.items(), key=lambda x: x[1], reverse=True
        )[:self.top_k_global]
        for node, _ in top_global:
            selected_nodes.add(node)

        # 4. Induced subgraph
        return self.global_graph.subgraph(selected_nodes).copy()

    def _get_centrality(self) -> Dict[str, float]:
        """Compute (or return cached) degree centrality for the global graph."""
        if self._cached_centrality is None:
            self._cached_centrality = nx.degree_centrality(self.global_graph)
        return self._cached_centrality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_entity_metadata(graph: nx.Graph) -> List[EntityMetadata]:
    """Build EntityMetadata list from graph nodes (entity nodes only)."""
    result = []
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get("role", "entity") == "entity":
            result.append(EntityMetadata(
                entity_name=node_id,
                entity_type=node_data.get("entity_type", "UNKNOWN"),
                description=node_data.get("description", ""),
                weight=float(node_data.get("weight", 1.0)),
            ))
    return result
