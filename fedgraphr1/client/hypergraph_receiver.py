"""
fedgraphr1/client/hypergraph_receiver.py
==========================================
Client-side reconstruction of local Knowledge Hypergraph from a
HypergraphFragment received from the FL server.

Steps (§3.2.3 Client-side Reconstruction):
  1. Deserialise the NetworkX graph from the fragment (in simulation:
     the nx.Graph object is passed directly; in distributed: decompress
     GraphML bytes).
  2. Write the graph to disk as GraphML for the local NetworkXStorage.
  3. Rebuild the entity KV store (kv_store_entities.json).
  4. Rebuild the hyperedge KV store (kv_store_hyperedges.json).
  5. Build FAISS indices locally using the BGE embedding model.
     (Server does NOT send pre-built indices — saves ~40 MB per client
      per round; rebuilding locally takes ~1-2s on GPU.)

Design note: The FAISS index build requires FlagEmbedding.
If FlagEmbedding is not installed, the receiver falls back to storing
the KV data without an index (retrieval will use linear scan).

Plan reference: Section 3.2.3 (HypergraphReceiver)
               Phase 2 deliverable: client/hypergraph_receiver.py
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from fedgraphr1.shared_types import HypergraphFragment

logger = logging.getLogger("fedgraphr1")


class HypergraphReceiver:
    """Receive a HypergraphFragment from the server and rebuild local stores.

    Args:
        working_dir: Client-local directory where KV stores and FAISS
            indices are written.
        embedding_model: Optional FlagEmbedding model (already loaded)
            for building FAISS indices.  If None, index build is skipped.

    Plan §3.2.3
    """

    def __init__(
        self,
        working_dir: str,
        embedding_model=None,
    ):
        self.working_dir = working_dir
        self.embedding_model = embedding_model
        os.makedirs(working_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def receive(self, fragment: HypergraphFragment) -> nx.Graph:
        """Reconstruct local stores from *fragment* and return the local KG.

        Args:
            fragment: HypergraphFragment from the server.

        Returns:
            Local nx.Graph (the rebuilt knowledge hypergraph).
        """
        import asyncio

        # ── 1. Deserialise the graph ────────────────────────────────────
        local_kg = self._deserialise_graph(fragment)

        # ── 2. Write GraphML to disk (for NetworkXStorage compatibility) ─
        graphml_path = os.path.join(
            self.working_dir, "graph_chunk_entity_relation.graphml"
        )
        nx.write_graphml(local_kg, graphml_path)
        logger.info(
            f"[HypergraphReceiver] Wrote GraphML: "
            f"{local_kg.number_of_nodes()} nodes, "
            f"{local_kg.number_of_edges()} edges → {graphml_path}"
        )

        # ── 3 & 4. Rebuild KV stores ───────────────────────────────────
        entities_kv, hyperedges_kv = self._rebuild_kv_stores(local_kg)

        self._write_json(
            entities_kv,
            os.path.join(self.working_dir, "kv_store_entities.json"),
        )
        self._write_json(
            hyperedges_kv,
            os.path.join(self.working_dir, "kv_store_hyperedges.json"),
        )

        # ── 5. Build FAISS indices ─────────────────────────────────────
        if self.embedding_model is not None:
            self._build_faiss_indices(entities_kv, hyperedges_kv)
        else:
            logger.warning(
                "[HypergraphReceiver] No embedding model — FAISS index "
                "not built.  FederatedSearchTool will use linear scan."
            )

        return local_kg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deserialise_graph(self, fragment: HypergraphFragment) -> nx.Graph:
        """Extract the nx.Graph from *fragment*.

        In simulation mode `fragment.graph_data` is already an nx.Graph.
        In distributed mode it is compressed GraphML bytes.
        """
        if isinstance(fragment.graph_data, nx.Graph):
            # Simulation: graph passed directly (no network serialisation)
            return fragment.graph_data

        if isinstance(fragment.graph_data, (bytes, bytearray)):
            # Distributed: decompress GraphML bytes
            from fedgraphr1.utils.compression import decompress
            graphml_bytes = decompress(fragment.graph_data)
            buf = io.BytesIO(graphml_bytes)
            return nx.read_graphml(buf)

        # Fallback: empty graph (should not reach here)
        logger.error(
            "[HypergraphReceiver] Unknown graph_data type: "
            f"{type(fragment.graph_data)}"
        )
        return nx.Graph()

    def _rebuild_kv_stores(
        self, local_kg: nx.Graph
    ) -> tuple[Dict, Dict]:
        """Walk *local_kg* nodes and build entity/hyperedge KV dicts.

        Keys are compute_mdhash_id("ent-" + entity_name) style hashes
        matching the format used by graphr1/storage.py JsonKVStorage.
        """
        try:
            from graphr1.utils import compute_mdhash_id
        except ImportError:
            # Fallback: use MD5 directly
            from hashlib import md5
            def compute_mdhash_id(text, prefix=""):
                return prefix + md5(text.encode()).hexdigest()

        entities_kv: Dict[str, dict] = {}
        hyperedges_kv: Dict[str, dict] = {}

        for node_id, node_data in local_kg.nodes(data=True):
            role = node_data.get("role", "entity")
            if role == "hyperedge":
                key = compute_mdhash_id(node_id, prefix="rel-")
                hyperedges_kv[key] = {
                    "hyperedge_name": node_id,
                    "content": node_id,
                    "weight": node_data.get("weight", 1.0),
                }
            else:
                key = compute_mdhash_id(node_id, prefix="ent-")
                entities_kv[key] = {
                    "entity_name": node_id,
                    "entity_type": node_data.get("entity_type", "UNKNOWN"),
                    "description": node_data.get("description", ""),
                    "content": node_id + " " + node_data.get("description", ""),
                    "weight": node_data.get("weight", 1.0),
                }

        return entities_kv, hyperedges_kv

    def _build_faiss_indices(
        self,
        entities_kv: Dict[str, dict],
        hyperedges_kv: Dict[str, dict],
    ):
        """Build FAISS IndexFlatIP for entities and hyperedges.

        Uses the BGE embedding model (already loaded) to encode content.
        Writes index_entity.bin and index_hyperedge.bin to working_dir.

        Plan §3.2.3: "Client tự encode_corpus() → ~1-2s on GPU"
        """
        try:
            import faiss
        except ImportError:
            logger.warning(
                "[HypergraphReceiver] faiss not installed — skipping index build."
            )
            return

        model = self.embedding_model

        def _build_one(kv: Dict[str, dict], content_key: str, fname: str):
            if not kv:
                return
            contents = [v[content_key] for v in kv.values()]
            embeddings = model.encode_corpus(contents)
            embeddings = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            path = os.path.join(self.working_dir, fname)
            faiss.write_index(index, path)
            logger.info(
                f"[HypergraphReceiver] Built FAISS index: "
                f"{len(contents)} vectors → {path}"
            )

        _build_one(entities_kv, "content", "index_entity.bin")
        _build_one(hyperedges_kv, "content", "index_hyperedge.bin")

    @staticmethod
    def _write_json(data: dict, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
