"""
fedgraphr1/client/federated_search_tool.py
============================================
Local FAISS-based retrieval tool for the federated setting.

Replaces the centralised HTTP API call (`localhost:8001`) from
graphr1/agent/tool/tools/search_tool.py with a local search against
the FAISS indices and KV stores built from the Server-distributed
Hypergraph fragment.

Key differences from the centralised SearchTool:
  - No HTTP call; all retrieval is in-process.
  - Entity/hyperedge indices are rebuilt each round from the received fragment.
  - O(1) entity/hyperedge lookup via pre-built indexed lists (not dict.values()).
  - Mirrors the n-ary fact format returned by the original SearchTool so the
    ToolGenerationManager / agentic loop requires no modification.
  - Extends agent.tool.tool_base.Tool so it plugs directly into ToolEnv
    without any changes to ToolGenerationManager.

Plan reference: Section 2.1.3 (FederatedSearchTool)
               Phase 2 deliverable: client/federated_search_tool.py
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

logger = logging.getLogger("fedgraphr1")

# ---------------------------------------------------------------------------
# Graceful fallback if agent package is not on sys.path
# (keeps fedgraphr1 importable as standalone package)
# ---------------------------------------------------------------------------
try:
    from agent.tool.tool_base import Tool as _BaseTool
except ImportError:
    class _BaseTool:  # type: ignore[no-redef]
        """Minimal shim when agent.tool is unavailable."""
        def __init__(self, name="", description="", parameters=None):
            self.name = name
            self.description = description
            self.parameters = parameters or {"type": "object", "properties": {}, "required": []}

        def validate_args(self, args: Dict):
            if not isinstance(args, dict):
                return False, "Arguments must be a dictionary"
            if "query" not in args:
                return False, "Missing required parameter: query"
            return True, "Parameters valid"

        def calculate_reward(self, args: Dict, result: str) -> float:
            return 0.0

        def get_description(self) -> Dict:
            return {"name": self.name, "description": self.description, "parameters": self.parameters}

        def get_simple_description(self) -> str:
            return f"Tool name: {self.name}\nDescription: {self.description}"

        def batch_execute(self, args_list: List[Dict]) -> List[str]:
            return [self.execute(args) for args in args_list]


# Top-k entities / hyperedges to return per query (mirrors centralized default)
_DEFAULT_ENTITY_TOP_K = 10
_DEFAULT_HE_TOP_K = 10


class FederatedSearchTool(_BaseTool):
    """Local retrieval tool using FAISS indices from the Hypergraph fragment.

    Drop-in replacement for `SearchTool` in the agentic reasoning loop.
    Extends `agent.tool.tool_base.Tool` so it can be placed directly into a
    `ToolEnv` — ToolGenerationManager requires zero changes.

    Args:
        working_dir: Client local directory containing:
            - index_entity.bin
            - index_hyperedge.bin
            - kv_store_entities.json
            - kv_store_hyperedges.json
            - graph_chunk_entity_relation.graphml
        embedding_model: FlagEmbedding model for query encoding.
            Can be set later via set_embedding_model().
        top_k_entities: Number of entities to retrieve per query.
        top_k_hyperedges: Number of hyperedges to retrieve per query.

    Plan §2.1.3
    """

    _TOOL_NAME = "search"
    _TOOL_DESCRIPTION = (
        "Search for information in the local federated knowledge graph."
    )
    _TOOL_PARAMETERS = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            }
        },
        "required": ["query"],
    }

    def __init__(
        self,
        working_dir: str,
        embedding_model=None,
        top_k_entities: int = _DEFAULT_ENTITY_TOP_K,
        top_k_hyperedges: int = _DEFAULT_HE_TOP_K,
    ):
        super().__init__(
            name=self._TOOL_NAME,
            description=self._TOOL_DESCRIPTION,
            parameters=self._TOOL_PARAMETERS,
        )
        self.working_dir = working_dir
        self.embedding_model = embedding_model
        self.top_k_entities = top_k_entities
        self.top_k_hyperedges = top_k_hyperedges

        # Lazily loaded indices / KV data
        self._entity_index = None
        self._hyperedge_index = None
        self._entity_list: List[dict] = []   # O(1) indexed access
        self._hyperedge_list: List[dict] = []
        self._kg_graph: Optional[nx.Graph] = None

        self._loaded = False

    # ------------------------------------------------------------------
    # Tool.execute interface (required by agent.tool.tool_base.Tool)
    # ------------------------------------------------------------------

    def execute(self, args: Dict) -> str:
        """Single-query retrieval (delegates to batch_execute)."""
        return self.batch_execute([args])[0]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_embedding_model(self, model):
        """Attach or replace the embedding model.

        Called by GraphR1Client after the embedding model is initialised.
        After attaching, call load() or reload() to rebuild the index.
        """
        self.embedding_model = model

    def load(self):
        """Load FAISS indices, KV stores, and graph from *working_dir*.

        Called once after HypergraphReceiver.receive() completes.
        Safe to call even if indices don't exist yet (will load whatever
        is present and mark as loaded so queries return empty results
        rather than crashing).
        """
        try:
            import faiss
        except ImportError:
            logger.warning(
                "[FederatedSearchTool] faiss not installed — "
                "retrieval will return empty results."
            )
            self._loaded = False
            return

        entity_index_path = os.path.join(self.working_dir, "index_entity.bin")
        he_index_path = os.path.join(self.working_dir, "index_hyperedge.bin")
        entity_kv_path = os.path.join(self.working_dir, "kv_store_entities.json")
        he_kv_path = os.path.join(self.working_dir, "kv_store_hyperedges.json")
        graph_path = os.path.join(
            self.working_dir, "graph_chunk_entity_relation.graphml"
        )

        # Load entity index + KV
        if os.path.exists(entity_index_path):
            self._entity_index = faiss.read_index(entity_index_path)
        if os.path.exists(entity_kv_path):
            with open(entity_kv_path, encoding="utf-8") as f:
                kv = json.load(f)
            self._entity_list = list(kv.values())

        # Load hyperedge index + KV
        if os.path.exists(he_index_path):
            self._hyperedge_index = faiss.read_index(he_index_path)
        if os.path.exists(he_kv_path):
            with open(he_kv_path, encoding="utf-8") as f:
                he_kv = json.load(f)
            self._hyperedge_list = list(he_kv.values())

        # Load graph
        if os.path.exists(graph_path):
            self._kg_graph = nx.read_graphml(graph_path)

        self._loaded = True
        logger.info(
            f"[FederatedSearchTool] Loaded from {self.working_dir} — "
            f"{len(self._entity_list)} entities, "
            f"{len(self._hyperedge_list)} hyperedges"
        )

    def reload(self):
        """Reload all indices from disk (call after each round's fragment update)."""
        self._entity_index = None
        self._hyperedge_index = None
        self._entity_list = []
        self._hyperedge_list = []
        self._kg_graph = None
        self._loaded = False
        self.load()

    # ------------------------------------------------------------------
    # Search interface (mirrors SearchTool.batch_execute)
    # ------------------------------------------------------------------

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """Execute a batch of retrieval queries locally.

        Args:
            args_list: List of dicts, each with key "query" (str).

        Returns:
            List of knowledge context strings (one per query), formatted
            as "<knowledge>...</knowledge>" blocks matching the original
            SearchTool output format consumed by ToolGenerationManager.

        Plan §2.1.3
        """
        if not self._loaded:
            logger.warning(
                "[FederatedSearchTool] Tool not loaded — returning empty results. "
                "Call load() after HypergraphReceiver.receive()."
            )
            return ["<knowledge>Knowledge graph not yet loaded.</knowledge>" for _ in args_list]

        if self.embedding_model is None:
            logger.warning(
                "[FederatedSearchTool] No embedding model — "
                "returning empty results."
            )
            return ["<knowledge>Embedding model not available.</knowledge>" for _ in args_list]

        queries = [a.get("query", "") for a in args_list]
        query_embeddings = self.embedding_model.encode_queries(queries)
        query_embeddings = np.array(query_embeddings, dtype=np.float32)
        # Normalise for cosine similarity (IndexFlatIP with L2-normalised vecs)
        faiss_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        faiss_norm = np.where(faiss_norm == 0, 1.0, faiss_norm)
        query_embeddings = query_embeddings / faiss_norm

        results = []
        for i, q_emb in enumerate(query_embeddings):
            q_emb = q_emb.reshape(1, -1)
            matched_entities = self._search_entities(q_emb)
            matched_hyperedges = self._search_hyperedges(q_emb)
            knowledge = self._build_knowledge(
                matched_entities, matched_hyperedges, queries[i]
            )
            results.append(knowledge)
        return results

    # ------------------------------------------------------------------
    # Internal retrieval helpers
    # ------------------------------------------------------------------

    def _search_entities(self, q_emb: np.ndarray) -> List[str]:
        """Return top-k entity names for query embedding *q_emb*."""
        if self._entity_index is None or not self._entity_list:
            return []
        k = min(self.top_k_entities, len(self._entity_list))
        _, ids = self._entity_index.search(q_emb, k)
        names = []
        for idx in ids[0]:
            if 0 <= idx < len(self._entity_list):
                names.append(
                    self._entity_list[idx].get("entity_name", "")
                )
        return names

    def _search_hyperedges(self, q_emb: np.ndarray) -> List[str]:
        """Return top-k hyperedge content strings for query embedding *q_emb*."""
        if self._hyperedge_index is None or not self._hyperedge_list:
            return []
        k = min(self.top_k_hyperedges, len(self._hyperedge_list))
        _, ids = self._hyperedge_index.search(q_emb, k)
        contents = []
        for idx in ids[0]:
            if 0 <= idx < len(self._hyperedge_list):
                contents.append(
                    self._hyperedge_list[idx].get("content", "")
                )
        return contents

    def _build_knowledge(
        self,
        matched_entities: List[str],
        matched_hyperedges: List[str],
        query: str,
    ) -> str:
        """Compose a knowledge context string from retrieval results.

        Format mirrors the original SearchTool output so ToolGenerationManager
        can process it without modification.

        Returns:
            String wrapped in <knowledge>...</knowledge> tags.
        """
        lines = []

        # Add hyperedge facts (direct knowledge fragments)
        for he_content in matched_hyperedges:
            if he_content:
                lines.append(f"Fact: {he_content}")

        # Add entity descriptions (with 1-hop context from KG if available)
        for entity_name in matched_entities:
            if not entity_name:
                continue
            entity_data = self._get_entity_data(entity_name)
            if entity_data:
                desc = entity_data.get("description", "")
                if desc:
                    lines.append(f"Entity [{entity_name}]: {desc}")
                # Add connected hyperedges from the KG (1-hop context)
                if self._kg_graph is not None and self._kg_graph.has_node(entity_name):
                    for neighbor in self._kg_graph.neighbors(entity_name):
                        n_data = self._kg_graph.nodes[neighbor]
                        if n_data.get("role") == "hyperedge":
                            lines.append(f"Related fact: {neighbor}")

        if not lines:
            return "<knowledge>No relevant knowledge found.</knowledge>"

        content = "\n".join(lines[:50])  # cap at 50 lines
        return f"<knowledge>\n{content}\n</knowledge>"

    def _get_entity_data(self, entity_name: str) -> Optional[dict]:
        """Look up entity metadata by name (O(n) scan — acceptable for ≤10K)."""
        for e in self._entity_list:
            if e.get("entity_name") == entity_name:
                return e
        return None
