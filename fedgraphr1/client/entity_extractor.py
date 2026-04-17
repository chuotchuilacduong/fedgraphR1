"""
fedgraphr1/client/entity_extractor.py
======================================
Client-side entity extraction wrapper (π_ext).

Wraps the existing `extract_entities()` pipeline from graphr1/operate.py
and translates its output into a `ClientExtractionResult` for upload to
the FL server.

Design choices:
  - Reuses graphr1/operate.py's extraction logic 100% — no rewrite.
  - Uses a temporary local NetworkXStorage to collect raw extraction
    output before packaging it as ClientExtractionResult.
  - Stores only MD5 hashes of source chunks (not raw text) in the
    extraction records, honouring the privacy principle from §2.1.1.
  - Calls chunking_by_token_size from graphr1/operate.py directly.

Plan reference: Section 2.1.1 (Entity Extraction Layer π_ext)
               Phase 1 deliverable: client/entity_extractor.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict
from functools import partial
from hashlib import md5
from typing import Dict, List, Optional

from fedgraphr1.shared_types import (
    ClientExtractionResult,
    ExtractedEdgeRecord,
    ExtractedEntityRecord,
    ExtractedHyperedgeRecord,
)

logger = logging.getLogger("fedgraphr1")


class ClientEntityExtractor:
    """Runs the Graph-R1 entity extraction pipeline on local documents.

    Wraps `graphr1.operate.extract_entities()` and converts the output
    to a `ClientExtractionResult` ready for upload.

    Args:
        graphr1_instance: A fully initialised `GraphR1` dataclass from
            graphr1/graphr1.py.  The extractor borrows its LLM function,
            tokeniser settings, and storage instances.
        client_id: String identifier for this client (for logging / metadata).
        working_dir: Per-client directory for temporary local storage.
            Defaults to graphr1_instance.working_dir.

    Plan §2.1.1
    """

    def __init__(
        self,
        graphr1_instance,
        client_id: str,
        working_dir: Optional[str] = None,
    ):
        self._graphr1 = graphr1_instance
        self.client_id = client_id
        self.working_dir = working_dir or graphr1_instance.working_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        documents: List[str],
        round_number: int = 0,
    ) -> ClientExtractionResult:
        """Extract entities from *documents* and return a packaged result.

        Args:
            documents: List of raw text strings (local corpus D_k).
            round_number: Current FL round (recorded in metadata).

        Returns:
            ClientExtractionResult ready to be handed to EntityPacker.
        """
        loop = _get_or_create_event_loop()
        return loop.run_until_complete(
            self.aextract(documents, round_number)
        )

    async def aextract(
        self,
        documents: List[str],
        round_number: int = 0,
    ) -> ClientExtractionResult:
        """Async variant of extract()."""
        from dataclasses import asdict as _asdict
        from graphr1.operate import chunking_by_token_size, extract_entities
        from graphr1.utils import compute_mdhash_id

        g = self._graphr1
        global_config = _asdict(g)

        # ── 1. Chunk documents ──────────────────────────────────────────
        inserting_chunks: Dict[str, dict] = {}
        chunk_hash_map: Dict[str, str] = {}   # chunk_id → md5 of content

        for doc_text in documents:
            chunks = chunking_by_token_size(
                doc_text,
                overlap_token_size=g.chunk_overlap_token_size,
                max_token_size=g.chunk_token_size,
                tiktoken_model=g.tiktoken_model_name,
            )
            for chunk in chunks:
                content = chunk["content"].strip()
                if not content:
                    continue
                chunk_id = compute_mdhash_id(content, prefix="chunk-")
                inserting_chunks[chunk_id] = {"content": content}
                chunk_hash_map[chunk_id] = md5(content.encode()).hexdigest()

        if not inserting_chunks:
            logger.warning(f"[Client {self.client_id}] No chunks produced.")
            return ClientExtractionResult(
                client_id=self.client_id,
                round_number=round_number,
                metadata={"extraction_model": g.llm_model_name},
            )

        logger.info(
            f"[Client {self.client_id}] Extracting from "
            f"{len(inserting_chunks)} chunks …"
        )

        # ── 2. Run extraction into local KG ────────────────────────────
        # extract_entities() writes into the graphr1 instance's storages.
        # We reset them before extraction so we only collect this round's data.
        await g.chunk_entity_relation_graph.drop()

        maybe_new_kg = await extract_entities(
            inserting_chunks,
            knowledge_graph_inst=g.chunk_entity_relation_graph,
            entity_vdb=g.entities_vdb,
            hyperedge_vdb=g.hyperedges_vdb,
            global_config=global_config,
        )
        if maybe_new_kg is None:
            logger.warning(
                f"[Client {self.client_id}] extract_entities returned None."
            )
            return ClientExtractionResult(
                client_id=self.client_id,
                round_number=round_number,
                metadata={"extraction_model": g.llm_model_name},
            )

        # ── 3. Walk the KG and build typed records ─────────────────────
        entities: List[ExtractedEntityRecord] = []
        hyperedges: List[ExtractedHyperedgeRecord] = []
        edges: List[ExtractedEdgeRecord] = []

        kg = maybe_new_kg  # NetworkXStorage

        # Iterate over all nodes; classify by 'role' attribute
        for node_id, node_data in (await _all_nodes(kg)):
            role = node_data.get("role", "entity")
            source_raw = node_data.get("source_id", "")
            # source_id in the KG is already a chunk_id (MD5-prefixed).
            # We store it directly as the source_chunk_hash.
            source_hash = chunk_hash_map.get(source_raw, source_raw)

            if role == "hyperedge":
                hyperedges.append(ExtractedHyperedgeRecord(
                    hyperedge_name=node_id,
                    weight=float(node_data.get("weight", 1.0)),
                    source_chunk_hash=source_hash,
                ))
            else:
                entities.append(ExtractedEntityRecord(
                    entity_name=node_id,
                    entity_type=node_data.get("entity_type", "UNKNOWN"),
                    description=node_data.get("description", ""),
                    weight=float(node_data.get("weight", 50.0)),
                    source_chunk_hash=source_hash,
                ))

        # Edges connect hyperedge nodes → entity nodes
        for src, tgt, edge_data in (await _all_edges(kg)):
            edges.append(ExtractedEdgeRecord(
                src_id=src,
                tgt_id=tgt,
                weight=float(edge_data.get("weight", 1.0)),
                source_chunk_hash=chunk_hash_map.get(
                    edge_data.get("source_id", ""), ""
                ),
            ))

        result = ClientExtractionResult(
            client_id=self.client_id,
            round_number=round_number,
            entities=entities,
            hyperedges=hyperedges,
            edges=edges,
            metadata={
                "extraction_model": g.llm_model_name,
                "num_chunks": len(inserting_chunks),
                "num_gleaning_rounds": g.entity_extract_max_gleaning,
            },
        )
        logger.info(
            f"[Client {self.client_id}] Extraction done — "
            f"{result.num_entities} entities, "
            f"{result.num_hyperedges} hyperedges, "
            f"{result.num_edges} edges."
        )
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _all_nodes(kg):
    """Return list of (node_id, node_data) from a NetworkXStorage."""
    # NetworkXStorage wraps nx.Graph; access underlying graph directly.
    if hasattr(kg, "_graph"):
        return list(kg._graph.nodes(data=True))
    # Fallback: iterate via the storage's public interface
    keys = await kg.all_keys() if hasattr(kg, "all_keys") else []
    result = []
    for k in keys:
        data = await kg.get_node(k)
        if data is not None:
            result.append((k, data))
    return result


async def _all_edges(kg):
    """Return list of (src, tgt, edge_data) from a NetworkXStorage."""
    if hasattr(kg, "_graph"):
        return list(kg._graph.edges(data=True))
    return []


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop (safe for both main and worker threads)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
