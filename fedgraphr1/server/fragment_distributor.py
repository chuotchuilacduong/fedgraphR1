"""
fedgraphr1/server/fragment_distributor.py
==========================================
Server-side compression and distribution of Hypergraph fragments.

In simulation mode (in-memory message pool), no actual serialisation
is needed — the nx.Graph is passed directly.  The distributor provides
an optional compressed-bytes path for distributed deployment testing.

Key design (§3.2.1):
  - Server does NOT transmit pre-built FAISS indices (~40 MB each).
  - Client rebuilds FAISS locally from the GraphML + BGE model (~1-2s GPU).
  - For fragments < 10 MB: single compressed payload.
  - For fragments ≥ 10 MB: chunk into 4 MB stream pieces.

Plan reference: Section 3.2.1 (HypergraphFragment packaging)
               Section 3.2.2 (Fragment Compression & Streaming)
               Phase 2 deliverable: server/fragment_distributor.py
"""

from __future__ import annotations

import io
import json
import logging
from typing import Iterator, List

import networkx as nx

from fedgraphr1.shared_types import EntityMetadata, HypergraphFragment
from fedgraphr1.utils.compression import compress, decompress

logger = logging.getLogger("fedgraphr1")

# Streaming chunk size (4 MB per §3.2.2)
_CHUNK_SIZE = 4 * 1024 * 1024
# Threshold above which streaming is used (10 MB per §3.2.2)
_STREAM_THRESHOLD = 10 * 1024 * 1024


class HypergraphFragmentDistributor:
    """Compress and distribute Hypergraph fragments to clients.

    In simulation mode the fragment's graph_data (nx.Graph) is passed
    directly through the message_pool without serialisation.  For
    distributed deployment, call distribute_bytes() to get compressed
    GraphML bytes that can be sent over the network.

    Plan §3.2.2
    """

    def __init__(self, compression_level: int = 3):
        self.compression_level = compression_level

    # ------------------------------------------------------------------
    # Simulation mode (direct nx.Graph pass-through)
    # ------------------------------------------------------------------

    def distribute_in_memory(
        self,
        fragment: HypergraphFragment,
    ) -> HypergraphFragment:
        """Return *fragment* as-is for in-memory simulation.

        No serialisation overhead — the nx.Graph object is passed
        directly via the message_pool dict.

        Plan §3.2.1 Note: "lora_weights là plain Python state_dict"
        """
        # Mark as uncompressed so HypergraphReceiver uses the direct path
        fragment.is_compressed = False
        return fragment

    # ------------------------------------------------------------------
    # Distributed mode (compressed bytes)
    # ------------------------------------------------------------------

    def to_bytes(self, fragment: HypergraphFragment) -> bytes:
        """Serialise and compress a HypergraphFragment into bytes.

        Output can be transported over TCP/gRPC and reconstructed by
        HypergraphReceiver.receive() (which handles is_compressed=True).

        Plan §3.2.2
        """
        if not isinstance(fragment.graph_data, nx.Graph):
            raise TypeError(
                "fragment.graph_data must be nx.Graph for serialisation."
            )

        # Serialise graph to GraphML
        buf = io.BytesIO()
        nx.write_graphml(fragment.graph_data, buf)
        graphml_bytes = buf.getvalue()

        # Serialise entity metadata
        meta_json = json.dumps(
            [
                {
                    "entity_name": m.entity_name,
                    "entity_type": m.entity_type,
                    "description": m.description,
                    "weight": m.weight,
                }
                for m in fragment.entity_metadata
            ],
            ensure_ascii=False,
        ).encode("utf-8")

        # Combine into a single payload
        payload = {
            "client_id": fragment.client_id,
            "round_number": fragment.round_number,
            "graphml": graphml_bytes.decode("utf-8"),
            "entity_metadata": json.loads(meta_json.decode("utf-8")),
            "stats": {
                "num_nodes": fragment.stats.num_nodes,
                "num_edges": fragment.stats.num_edges,
                "num_entity_nodes": fragment.stats.num_entity_nodes,
                "num_hyperedge_nodes": fragment.stats.num_hyperedge_nodes,
                "coverage_ratio": fragment.stats.coverage_ratio,
            },
        }

        raw_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        compressed = compress(raw_bytes, level=self.compression_level)

        original_kb = len(raw_bytes) / 1024
        compressed_kb = len(compressed) / 1024
        logger.info(
            f"[FragmentDistributor] Serialised fragment for "
            f"client {fragment.client_id}: "
            f"raw={original_kb:.1f} KB → compressed={compressed_kb:.1f} KB"
        )
        return compressed

    def stream_bytes(
        self,
        fragment: HypergraphFragment,
    ) -> Iterator[bytes]:
        """Stream compressed fragment bytes in chunks (§3.2.2).

        Yields 4 MB chunks for large fragments; single yield for small ones.
        """
        compressed = self.to_bytes(fragment)
        total = len(compressed)

        if total < _STREAM_THRESHOLD:
            yield compressed
            return

        logger.info(
            f"[FragmentDistributor] Streaming large fragment "
            f"({total / 1024 / 1024:.1f} MB) in chunks"
        )
        for start in range(0, total, _CHUNK_SIZE):
            yield compressed[start: start + _CHUNK_SIZE]

    @staticmethod
    def from_bytes(data: bytes) -> HypergraphFragment:
        """Reconstruct a HypergraphFragment from compressed bytes.

        Called on the CLIENT side inside HypergraphReceiver to reconstruct
        the fragment from distributed bytes.

        Plan §3.2.3
        """
        from fedgraphr1.shared_types import FragmentStats

        raw = decompress(data)
        payload = json.loads(raw.decode("utf-8"))

        # Reconstruct nx.Graph from GraphML string
        graphml_str = payload["graphml"]
        buf = io.BytesIO(graphml_str.encode("utf-8"))
        graph = nx.read_graphml(buf)

        # Reconstruct entity metadata
        entity_metadata = [
            EntityMetadata(
                entity_name=m["entity_name"],
                entity_type=m["entity_type"],
                description=m["description"],
                weight=float(m["weight"]),
            )
            for m in payload.get("entity_metadata", [])
        ]

        stats_dict = payload.get("stats", {})
        stats = FragmentStats(
            num_nodes=stats_dict.get("num_nodes", 0),
            num_edges=stats_dict.get("num_edges", 0),
            num_entity_nodes=stats_dict.get("num_entity_nodes", 0),
            num_hyperedge_nodes=stats_dict.get("num_hyperedge_nodes", 0),
            coverage_ratio=stats_dict.get("coverage_ratio", 0.0),
        )

        return HypergraphFragment(
            client_id=payload["client_id"],
            round_number=payload["round_number"],
            graph_data=graph,
            entity_metadata=entity_metadata,
            stats=stats,
            is_compressed=False,
        )
