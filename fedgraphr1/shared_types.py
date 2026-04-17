"""
fedgraphr1/shared_types.py
===========================
Shared dataclasses for the Federated Graph-R1 system.

These types define the data structures exchanged between:
  - Client extraction layer  →  ClientExtractionResult
  - Client → Server (uplink) →  serialised via entity_packer.py
  - Server → Client (downlink) via fragment_distributor.py

Plan reference: Section 2.1.1 (ExtractedEntityRecord dataclasses)
               Section 3.1.1 (EntityPacket design)
               Section 3.2.1 (HypergraphFragment design)

NOTE: This module was previously named types.py but was renamed to
shared_types.py to avoid shadowing Python's stdlib 'types' module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Client-side extraction records
# ---------------------------------------------------------------------------


@dataclass
class ExtractedEntityRecord:
    """One entity extracted at a Client.

    Mirrors the output of _handle_single_entity_extraction() in
    graphr1/operate.py but stripped of raw chunk content — only the
    MD5 hash of the source chunk is retained for traceability.

    Plan §2.1.1
    """

    entity_name: str        # e.g. '"ALEX"' (uppercased, quoted)
    entity_type: str        # e.g. '"PERSON"'
    description: str        # semantic description text
    weight: float           # key_score from extraction prompt
    source_chunk_hash: str  # MD5 of source chunk text  (NOT the raw text)


@dataclass
class ExtractedHyperedgeRecord:
    """One hyper-relation (hyperedge) extracted at a Client.

    Corresponds to _handle_single_hyperrelation_extraction() output.
    Plan §2.1.1
    """

    hyperedge_name: str     # e.g. '<hyperedge>Alex clenched ...'
    weight: float
    source_chunk_hash: str


@dataclass
class ExtractedEdgeRecord:
    """One directed edge connecting a hyperedge to an entity.

    Created when an entity record's `hyper_relation` field is non-empty.
    Plan §2.1.1
    """

    src_id: str             # hyperedge_name
    tgt_id: str             # entity_name
    weight: float
    source_chunk_hash: str


@dataclass
class ClientExtractionResult:
    """Full extraction result from a single Client for one FL round.

    Uploaded to the Server via EntityPacker.pack().
    Plan §2.1.1
    """

    client_id: str
    round_number: int
    entities: List[ExtractedEntityRecord] = field(default_factory=list)
    hyperedges: List[ExtractedHyperedgeRecord] = field(default_factory=list)
    edges: List[ExtractedEdgeRecord] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)  # timestamp, model, config …

    def __post_init__(self):
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = time.time()

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_hyperedges(self) -> int:
        return len(self.hyperedges)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def entity_names(self) -> List[str]:
        return [e.entity_name for e in self.entities]


# ---------------------------------------------------------------------------
# Server-side hypergraph fragment descriptor
# ---------------------------------------------------------------------------


@dataclass
class FragmentStats:
    """Statistics of a HypergraphFragment (metadata only, no graph data).

    Plan §3.2.1
    """

    num_nodes: int = 0
    num_edges: int = 0
    num_entity_nodes: int = 0
    num_hyperedge_nodes: int = 0
    coverage_ratio: float = 0.0   # fraction of global graph nodes included


@dataclass
class EntityMetadata:
    """Lightweight entity descriptor sent inside a HypergraphFragment.

    Used to rebuild the client-side KV store without transmitting the full
    NetworkX graph for every attribute.  Plan §3.2.1
    """

    entity_name: str
    entity_type: str
    description: str
    weight: float


@dataclass
class HypergraphFragment:
    """Server → Client payload carrying a (sub)graph and entity metadata.

    In simulation mode the actual nx.Graph object is passed directly
    (no serialisation required).  For distributed deployment, replace
    `graph_data` with compressed GraphML bytes.

    Plan §3.2.1
    """

    client_id: str
    round_number: int
    # In simulation: nx.Graph object.  In distributed: compressed GraphML bytes.
    graph_data: object = None
    entity_metadata: List[EntityMetadata] = field(default_factory=list)
    stats: FragmentStats = field(default_factory=FragmentStats)
    is_compressed: bool = False     # True only in distributed deployment


# ---------------------------------------------------------------------------
# Model synchronisation payload
# ---------------------------------------------------------------------------


@dataclass
class LoRAPayload:
    """LoRA weight packet exchanged in the message pool.

    Plan §4.3.1  /  §6.4 Message Pool Schema
    """

    client_id: str
    round_number: int
    # str → tensor mapping (parameter name → weight tensor)
    lora_state_dict: Dict = field(default_factory=dict)
    num_samples: int = 0           # local training samples (for FedAvg weighting)


# ---------------------------------------------------------------------------
# Server downlink / client uplink message schemas
# (mirrors §6.4 Message Pool Schema)
# ---------------------------------------------------------------------------


@dataclass
class ServerMessage:
    """Payload written to message_pool["server"] or message_pool["server_for_{cid}"].

    Plan §6.4
    """

    round_number: int
    lora_weights: Optional[Dict] = None        # global LoRA state dict
    hypergraph_fragment: Optional[HypergraphFragment] = None
    config: Dict = field(default_factory=dict)


@dataclass
class ClientMessage:
    """Payload written to message_pool["client_{cid}"].

    Plan §6.4
    """

    client_id: str
    round_number: int
    lora_weights: Optional[Dict] = None        # local LoRA delta
    extraction_result: Optional[ClientExtractionResult] = None
    num_samples: int = 0
    metrics: Dict = field(default_factory=dict)
