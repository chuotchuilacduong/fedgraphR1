"""
fedgraphr1/client/entity_packer.py
====================================
Client-side serialisation and compression of ClientExtractionResult.

Implements the bandwidth-optimisation pipeline from §3.1.2:
  1. Weight-threshold filtering  (drop low-importance entities)
  2. Description truncation      (cap at max_description_tokens)
  3. Delta transmission          (only send new entities since last round)
  4. JSON serialisation          (Protobuf avoided for prototype simplicity)
  5. zstd compression            (via fedgraphr1.utils.compression)

Counterpart on the server is entity_aggregator.py which calls unpack().

Design note: Protobuf (§3.1.1) is the production target but requires
compiled .proto stubs.  For the simulation prototype we use JSON + zstd
which achieves comparable compression and avoids the build dependency.
The public interface (pack/unpack bytes) is identical so swapping to
Protobuf later requires only changing the inner serialisation logic.

Plan reference: Section 3.1.2 (Bandwidth Optimization)
               Phase 1 deliverable: client/entity_packer.py
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict
from typing import Optional, Set

from fedgraphr1.shared_types import (
    ClientExtractionResult,
    ExtractedEdgeRecord,
    ExtractedEntityRecord,
    ExtractedHyperedgeRecord,
)
from fedgraphr1.utils.compression import compress, decompress

logger = logging.getLogger("fedgraphr1")

# Default tiktoken model used for description truncation (same as GraphR1)
_TIKTOKEN_MODEL = "gpt-4o-mini"


class EntityPacker:
    """Pack a ClientExtractionResult into compressed bytes for upload.

    Maintains delta caches across rounds to enable delta transmission
    (only new entities/hyperedges/edges are sent after round 0).

    Args:
        max_description_tokens: Hard cap on description length (in tokens).
            Default 200 matches plan §3.1.2.
        weight_threshold: Drop entities with weight < this value.
            Default 30.0 (plan §3.1.2).
        compression_level: zstd level (1–22).  Default 3 (fast + decent).
        use_delta: Enable delta transmission.  Should be True in production;
            set False to force full snapshot each round (e.g. round 0).

    Plan §3.1.2
    """

    def __init__(
        self,
        max_description_tokens: int = 200,
        weight_threshold: float = 30.0,
        compression_level: int = 3,
        use_delta: bool = True,
    ):
        self.max_desc_tokens = max_description_tokens
        self.weight_threshold = weight_threshold
        self.compression_level = compression_level
        self.use_delta = use_delta

        # Delta caches — names/hashes of items sent in previous rounds
        self._prev_entity_names: Set[str] = set()
        self._prev_hyperedge_names: Set[str] = set()
        self._prev_edge_hashes: Set[str] = set()

    # ------------------------------------------------------------------
    # Pack (client → server)
    # ------------------------------------------------------------------

    def pack(
        self,
        extraction_result: ClientExtractionResult,
        force_full: bool = False,
    ) -> bytes:
        """Serialise and compress *extraction_result*.

        Args:
            extraction_result: Full extraction output from ClientEntityExtractor.
            force_full: If True, send all data (ignore delta cache).
                        Automatically True for round 0.

        Returns:
            Compressed bytes ready for upload to the server.
        """
        result = copy.deepcopy(extraction_result)
        is_round_zero = extraction_result.round_number == 0 or force_full
        apply_delta = self.use_delta and not is_round_zero

        # ── Step 1: Weight filtering ────────────────────────────────────
        result.entities = [
            e for e in result.entities
            if e.weight >= self.weight_threshold
        ]

        # ── Step 2: Description truncation ─────────────────────────────
        try:
            from graphr1.utils import (
                encode_string_by_tiktoken,
                decode_tokens_by_tiktoken,
            )
            for entity in result.entities:
                tokens = encode_string_by_tiktoken(
                    entity.description, model_name=_TIKTOKEN_MODEL
                )
                if len(tokens) > self.max_desc_tokens:
                    entity.description = decode_tokens_by_tiktoken(
                        tokens[: self.max_desc_tokens],
                        model_name=_TIKTOKEN_MODEL,
                    )
        except ImportError:
            # Fallback: naive character truncation (~4 chars per token)
            char_cap = self.max_desc_tokens * 4
            for entity in result.entities:
                if len(entity.description) > char_cap:
                    entity.description = entity.description[:char_cap]

        # ── Step 3: Delta transmission ─────────────────────────────────
        if apply_delta:
            current_entity_names = {e.entity_name for e in result.entities}
            result.entities = [
                e for e in result.entities
                if e.entity_name not in self._prev_entity_names
            ]
            self._prev_entity_names = current_entity_names

            current_he_names = {h.hyperedge_name for h in result.hyperedges}
            result.hyperedges = [
                h for h in result.hyperedges
                if h.hyperedge_name not in self._prev_hyperedge_names
            ]
            self._prev_hyperedge_names = current_he_names

            current_edge_hashes = {
                _edge_hash(e) for e in result.edges
            }
            result.edges = [
                e for e in result.edges
                if _edge_hash(e) not in self._prev_edge_hashes
            ]
            self._prev_edge_hashes = current_edge_hashes
        else:
            # Full snapshot — update caches for next round
            self._prev_entity_names = {e.entity_name for e in result.entities}
            self._prev_hyperedge_names = {
                h.hyperedge_name for h in result.hyperedges
            }
            self._prev_edge_hashes = {_edge_hash(e) for e in result.edges}

        # ── Step 4: JSON serialisation ─────────────────────────────────
        payload = _result_to_dict(result)
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        # ── Step 5: zstd compression ────────────────────────────────────
        compressed = compress(raw, level=self.compression_level)

        logger.debug(
            f"[Client {extraction_result.client_id}] Packed "
            f"{len(result.entities)} entities, "
            f"{len(result.hyperedges)} hyperedges, "
            f"{len(result.edges)} edges — "
            f"raw={len(raw)/1024:.1f} KB  "
            f"compressed={len(compressed)/1024:.1f} KB"
        )
        return compressed

    # ------------------------------------------------------------------
    # Unpack (server side)
    # ------------------------------------------------------------------

    @staticmethod
    def unpack(data: bytes) -> ClientExtractionResult:
        """Deserialise bytes from pack() back into a ClientExtractionResult.

        This is a static method so the server can call it without a packer
        instance (no delta cache needed on the server side).

        Args:
            data: Bytes produced by pack().

        Returns:
            ClientExtractionResult.
        """
        raw = decompress(data)
        payload = json.loads(raw.decode("utf-8"))
        return _dict_to_result(payload)

    def reset_delta_cache(self):
        """Clear delta caches (call at the start of a new session)."""
        self._prev_entity_names.clear()
        self._prev_hyperedge_names.clear()
        self._prev_edge_hashes.clear()


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _edge_hash(edge: ExtractedEdgeRecord) -> str:
    """Stable string hash for an edge (used in delta cache)."""
    return f"{edge.src_id}|{edge.tgt_id}|{edge.weight}"


def _result_to_dict(result: ClientExtractionResult) -> dict:
    """Convert ClientExtractionResult → plain dict for JSON serialisation."""
    return {
        "client_id": result.client_id,
        "round_number": result.round_number,
        "metadata": result.metadata,
        "entities": [
            {
                "entity_name": e.entity_name,
                "entity_type": e.entity_type,
                "description": e.description,
                "weight": e.weight,
                "source_chunk_hash": e.source_chunk_hash,
            }
            for e in result.entities
        ],
        "hyperedges": [
            {
                "hyperedge_name": h.hyperedge_name,
                "weight": h.weight,
                "source_chunk_hash": h.source_chunk_hash,
            }
            for h in result.hyperedges
        ],
        "edges": [
            {
                "src_id": ed.src_id,
                "tgt_id": ed.tgt_id,
                "weight": ed.weight,
                "source_chunk_hash": ed.source_chunk_hash,
            }
            for ed in result.edges
        ],
    }


def _dict_to_result(payload: dict) -> ClientExtractionResult:
    """Convert plain dict (from JSON) back to ClientExtractionResult."""
    entities = [
        ExtractedEntityRecord(
            entity_name=e["entity_name"],
            entity_type=e["entity_type"],
            description=e["description"],
            weight=float(e["weight"]),
            source_chunk_hash=e["source_chunk_hash"],
        )
        for e in payload.get("entities", [])
    ]
    hyperedges = [
        ExtractedHyperedgeRecord(
            hyperedge_name=h["hyperedge_name"],
            weight=float(h["weight"]),
            source_chunk_hash=h["source_chunk_hash"],
        )
        for h in payload.get("hyperedges", [])
    ]
    edges = [
        ExtractedEdgeRecord(
            src_id=ed["src_id"],
            tgt_id=ed["tgt_id"],
            weight=float(ed["weight"]),
            source_chunk_hash=ed["source_chunk_hash"],
        )
        for ed in payload.get("edges", [])
    ]
    return ClientExtractionResult(
        client_id=payload["client_id"],
        round_number=payload["round_number"],
        entities=entities,
        hyperedges=hyperedges,
        edges=edges,
        metadata=payload.get("metadata", {}),
    )
