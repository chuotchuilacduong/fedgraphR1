"""
fedgraphr1/fl/client.py
=========================
GraphR1Client — the main federated client for Graph-R1.

Implements BaseClient from fl/base.py following FedGM's pattern
(FedGM/flcore/fedgm/client.py as style reference).

Per-round execution sequence (§2.3 / §4.3.3):
  1. execute()
     a. Load global LoRA weights from message_pool["server"]
     b. Reconstruct local KG from HypergraphFragment
     c. Run E epochs of local GRPO training (FederatedRayPPOTrainer)
     d. Extract entities from local documents (ClientEntityExtractor)
  2. send_message()
     a. Pack extraction result (EntityPacker)
     b. Write to message_pool["client_{cid}"]

Plan reference: Section 6.4 (FedGM-Style FL Implementation)
               Phase 1 deliverable: fl/client.py
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional

from fedgraphr1.fl.base import BaseClient
from fedgraphr1.shared_types import ClientMessage
from fedgraphr1.utils.debug_logger import step

logger = logging.getLogger("fedgraphr1")


class GraphR1Client(BaseClient):
    """Federated Graph-R1 client.

    Args:
        client_id: Integer client index (0-based).
        args: Parsed CLI namespace from fl/config.py get_args().
        local_data: List of document strings for this client's partition.
        message_pool: Shared in-memory dict (FedGM message-pool pattern).
        device: torch.device or string.
        graphr1_instance: Optional pre-initialised GraphR1 instance.
            If None, a new one is created with a per-client working_dir.
        local_docs: Optional pre-loaded documents (overrides local_data).

    Plan §6.4  "GraphR1Client(BaseClient)"
    """

    def __init__(
        self,
        client_id: int,
        args,
        local_data,
        message_pool: dict,
        device,
        graphr1_instance=None,
        embedding_model=None,
        local_examples=None,
    ):
        super().__init__(client_id, args, message_pool, device)

        self.local_data     = local_data     or []  # List[str]  — doc strings for π_ext extraction
        self.local_examples = local_examples or []  # List[dict] — QA dicts for π_θ GRPO training
        self._working_dir = os.path.join(
            getattr(args, "working_dir", "expr/fedgraphr1"),
            f"client_{client_id}",
        )
        os.makedirs(self._working_dir, exist_ok=True)

        # ── Embedding model (shared across receiver + search tool) ───────
        self._embedding_model = embedding_model

        # ── GraphR1 instance (per-client local KG storage) ──────────────
        self._graphr1 = graphr1_instance or self._init_graphr1()

        # ── Sub-modules ─────────────────────────────────────────────────
        from fedgraphr1.client.entity_extractor import ClientEntityExtractor
        from fedgraphr1.client.entity_packer import EntityPacker
        from fedgraphr1.client.hypergraph_receiver import HypergraphReceiver
        from fedgraphr1.client.federated_search_tool import FederatedSearchTool

        self._extractor = ClientEntityExtractor(
            graphr1_instance=self._graphr1,
            client_id=str(client_id),
            working_dir=self._working_dir,
        )
        self._packer = EntityPacker(
            max_description_tokens=getattr(args, "max_desc_tokens", 200),
            weight_threshold=getattr(args, "entity_weight_threshold", 30.0),
            use_delta=True,
        )
        self._receiver = HypergraphReceiver(
            working_dir=self._working_dir,
            embedding_model=self._embedding_model,
        )
        # Search tool is pre-created so ToolEnv can hold a reference to it
        # even before the first fragment arrives (it will return empty results
        # until load() is called after the first fragment).
        self._search_tool = FederatedSearchTool(
            working_dir=self._working_dir,
            embedding_model=self._embedding_model,
        )

        # ── GRPO client (LoRA lifecycle) ─────────────────────────────────
        self._grpo_client = self._init_grpo_client()

        # ── Round state ──────────────────────────────────────────────────
        self._last_extraction = None
        self._last_metrics: Dict = {}
        self._num_local_samples = len(self.local_examples or self.local_data)

        # ── Pre-built KG cache (lazily loaded when --pretrained_kg_dir set) ─
        self._pretrained_entities: Optional[List] = None
        self._pretrained_hyperedges: Optional[List] = None

    # ------------------------------------------------------------------
    # BaseClient interface
    # ------------------------------------------------------------------

    def execute(self):
        """Run one round of local work.

        Reads:  message_pool["server"]  or  message_pool[f"server_for_{id}"]
        Writes: self._last_extraction, self._last_metrics

        Plan §6.4
        """
        round_id = self.message_pool.get("round", 0)
        cid = self.client_id
        step(f"  Client {cid} │ execute() round={round_id}")

        # ── a. Load server payload ──────────────────────────────────────
        server_msg = (
            self.message_pool.get(f"server_for_{self.client_id}")
            or self.message_pool.get("server")
            or {}
        )

        # Load global LoRA weights
        lora_weights = server_msg.get("lora_weights")
        if lora_weights and self._grpo_client is not None:
            self._grpo_client.set_lora_state_dict(lora_weights)
            step(f"  Client {cid} │ [a] LoRA loaded",
                 {"tensors": len(lora_weights)})
            logger.info(
                f"[Client {self.client_id}] Loaded global LoRA weights "
                f"(round {round_id})."
            )
        else:
            step(f"  Client {cid} │ [a] LoRA skipped",
                 {"reason": "no weights in server msg" if not lora_weights else "no grpo_client"})

        # ── b. Reconstruct local KG from fragment ──────────────────────
        # kg_unchanged=True means the server determined the global KG did
        # not change — client keeps its cached local KG (bandwidth saving).
        kg_unchanged = server_msg.get("kg_unchanged", False)
        fragment = server_msg.get("hypergraph_fragment")
        if fragment is not None:
            local_kg = self._receiver.receive(fragment)
            self._search_tool.reload()
            n_nodes = local_kg.number_of_nodes()
            step(f"  Client {cid} │ [b] KG updated",
                 {"nodes": n_nodes, "edges": local_kg.number_of_edges()})
            logger.info(
                f"[Client {self.client_id}] Local KG updated ({n_nodes} nodes)."
            )
        elif kg_unchanged:
            step(f"  Client {cid} │ [b] KG reuse",
                 {"reason": "server: KG unchanged — cached fragment reused"})
        else:
            step(f"  Client {cid} │ [b] KG skipped", {"reason": "no fragment"})

        # ── c. Local GRPO training ─────────────────────────────────────
        # Skipped on round 0 (KG init round): the model has not yet seen any
        # KG context, so training would produce noise rather than signal.
        # From round 1 onward, clients train with the updated local KG available
        # as retrieval context.
        # π_θ (Qwen2.5/LoRA) trains on QA dicts (local_examples).
        # π_ext (TimelyGPT/GPT-4o-mini) is fixed — NOT trained here.
        kg_init_round = self.message_pool.get("kg_init_round", False)
        training_metrics: Dict = {}
        num_epochs = getattr(self.args, "num_epochs", 1)
        # GRPO uses QA dicts (local_examples); fall back to doc strings only as last resort
        train_data = self.local_examples if self.local_examples else self.local_data
        if kg_init_round:
            step(f"  Client {cid} │ [c] GRPO skipped", {"reason": "round 0 KG init"})
        elif self._grpo_client is not None and train_data:
            step(f"  Client {cid} │ [c] GRPO training",
                 {"epochs": num_epochs, "samples": len(train_data)})
            training_metrics = self._run_local_grpo(round_id, num_epochs)
            step(f"  Client {cid} │ [c] GRPO done",
                 {k: f"{v:.4f}" if isinstance(v, float) else v
                  for k, v in training_metrics.items()})
        else:
            step(f"  Client {cid} │ [c] GRPO skipped",
                 {"reason": "no model" if self._grpo_client is None else "no train data"})

        # ── d. Entity extraction (batched per round) ───────────────────
        # π_ext processes only the document slice assigned to this round.
        # When --pretrained_kg_dir is set, the pre-built KV stores are sliced
        # directly instead of calling TimelyGPT — the rest of the FL loop
        # (server aggregation, fragment broadcast, GRPO) runs unchanged.
        num_rounds = max(1, getattr(self.args, "num_rounds", 1))
        if getattr(self.args, "pretrained_kg_dir", None):
            self._last_extraction = self._extract_from_pretrained_kg(round_id)
            step(f"  Client {cid} │ [d] Pre-built KG batch {round_id + 1}/{num_rounds}",
                 {"entities":   self._last_extraction.num_entities,
                  "hyperedges": self._last_extraction.num_hyperedges})
            logger.info(
                f"[Client {self.client_id}] Pre-built KG: "
                f"{self._last_extraction.num_entities} entities "
                f"(batch {round_id + 1}/{num_rounds})."
            )
        else:
            batch = self._get_extraction_batch(round_id)
            if batch:
                step(f"  Client {cid} │ [d] Extract batch {round_id + 1}/{num_rounds}",
                     {"docs": len(batch)})
                self._last_extraction = self._extractor.extract(
                    documents=batch,
                    round_number=round_id,
                )
                step(f"  Client {cid} │ [d] Extraction done",
                     {"entities":   self._last_extraction.num_entities,
                      "hyperedges": self._last_extraction.num_hyperedges,
                      "edges":      self._last_extraction.num_edges})
                logger.info(
                    f"[Client {self.client_id}] Extracted "
                    f"{self._last_extraction.num_entities} entities "
                    f"(batch {round_id + 1}/{num_rounds})."
                )
            else:
                step(f"  Client {cid} │ [d] Extraction skipped",
                     {"reason": f"all batches done (round {round_id} >= {num_rounds})"
                                if self.local_data else "no local data"})
                from fedgraphr1.shared_types import ClientExtractionResult
                self._last_extraction = ClientExtractionResult(
                    client_id=str(self.client_id),
                    round_number=round_id,
                )

        self._last_metrics = {
            "round": round_id,
            "num_entities": self._last_extraction.num_entities,
            "num_hyperedges": self._last_extraction.num_hyperedges,
            **training_metrics,
        }

    def send_message(self):
        """Write client results to message_pool.

        Writes: message_pool["client_{client_id}"]

        Plan §6.4  message_pool["client_{cid}"]
        """
        round_id = self.message_pool.get("round", 0)

        # Extract LoRA weights for upload (optionally sparsified, P4.6)
        lora_weights = {}
        lora_density = None
        if self._grpo_client is not None:
            top_k = getattr(self.args, "lora_top_k_ratio", None)
            lora_weights = self._grpo_client.get_lora_state_dict(
                top_k_ratio=top_k
            )
            if lora_weights:
                from fedgraphr1.server.lora_sparsifier import TopKSparsifier
                lora_density = TopKSparsifier.density(lora_weights)

        self.message_pool[f"client_{self.client_id}"] = {
            "lora_weights": lora_weights,
            "lora_density": lora_density,   # fraction non-zero (P4.6); None = no LoRA
            # Keep extraction_result as Python object (no serialisation in simulation)
            "extraction_result": self._last_extraction,
            "num_samples": self._num_local_samples,
            "metrics": self._last_metrics,
        }
        logger.debug(
            f"[Client {self.client_id}] Sent message: "
            f"{self._last_extraction.num_entities} entities, "
            f"{len(lora_weights)} LoRA tensors"
            + (f", density={lora_density:.3f}" if lora_density is not None else "")
            + "."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_extraction_batch(self, round_id: int) -> List[str]:
        """Return the document slice assigned to this round.

        Divides local_data evenly into num_rounds sliding batches.
        Round 0 → docs[0:B], Round 1 → docs[B:2B], ..., Round R-1 → docs[last].
        Returns [] once all batches are exhausted (KG stable from that point).
        Not called when --pretrained_kg_dir is set (see _extract_from_pretrained_kg).
        """
        if not self.local_data:
            return []
        num_rounds = max(1, getattr(self.args, "num_rounds", 1))
        n = len(self.local_data)
        batch_size = max(1, math.ceil(n / num_rounds))
        start = round_id * batch_size
        if start >= n:
            return []
        return self.local_data[start : min(start + batch_size, n)]

    def _load_pretrained_kv(self):
        """Lazily load and cache the pre-built entity/hyperedge KV stores.

        Called on the first invocation of _extract_from_pretrained_kg().
        The two lists are held in memory for the lifetime of the client so
        they are only read from disk once regardless of the number of rounds.
        """
        if self._pretrained_entities is not None:
            return  # already cached

        import json
        kg_dir = os.path.abspath(getattr(self.args, "pretrained_kg_dir", ""))
        max_size = getattr(self.args, "max_pretrained_kg_size", None)

        entity_kv_path = os.path.join(kg_dir, "kv_store_entities.json")
        with open(entity_kv_path, encoding="utf-8") as f:
            kv = json.load(f)
        self._pretrained_entities = list(kv.values())
        if max_size:
            self._pretrained_entities = self._pretrained_entities[:max_size]

        he_kv_path = os.path.join(kg_dir, "kv_store_hyperedges.json")
        with open(he_kv_path, encoding="utf-8") as f:
            he_kv = json.load(f)
        self._pretrained_hyperedges = list(he_kv.values())
        if max_size:
            self._pretrained_hyperedges = self._pretrained_hyperedges[:max_size]

        logger.info(
            f"[Client {self.client_id}] Pre-built KV loaded: "
            f"{len(self._pretrained_entities)} entities, "
            f"{len(self._pretrained_hyperedges)} hyperedges "
            f"from {kg_dir}"
            + (f" (capped at {max_size})" if max_size else "")
        )

    def _extract_from_pretrained_kg(self, round_id: int) -> "ClientExtractionResult":
        """Build a ClientExtractionResult from the pre-built KV stores.

        Partitions the global entity/hyperedge pools across
        num_clients × num_rounds chunks so that each (client, round) pair
        receives a distinct non-overlapping slice:
            chunk_idx = client_id * num_rounds + round_id

        This feeds the server's aggregation + fragment-broadcast pipeline
        exactly as if TimelyGPT had run on raw documents — no other part of
        the FL loop changes.
        """
        import hashlib
        from fedgraphr1.shared_types import (
            ClientExtractionResult,
            ExtractedEntityRecord,
            ExtractedHyperedgeRecord,
        )

        self._load_pretrained_kv()

        num_rounds  = max(1, getattr(self.args, "num_rounds", 1))
        num_clients = max(1, getattr(self.args, "num_clients", 1))
        chunk_idx   = self.client_id * num_rounds + round_id
        total_chunks = num_clients * num_rounds

        def _slice(items):
            n = len(items)
            if n == 0:
                return []
            chunk_size = max(1, math.ceil(n / total_chunks))
            start = chunk_idx * chunk_size
            if start >= n:
                return []
            return items[start : min(start + chunk_size, n)]

        entity_slice = _slice(self._pretrained_entities)
        he_slice     = _slice(self._pretrained_hyperedges)

        entities = [
            ExtractedEntityRecord(
                entity_name=rec.get("entity_name", ""),
                entity_type="",
                description=rec.get("content", ""),
                weight=1.0,
                source_chunk_hash=hashlib.md5(
                    rec.get("entity_name", "").encode()
                ).hexdigest(),
            )
            for rec in entity_slice
        ]

        hyperedges = [
            ExtractedHyperedgeRecord(
                hyperedge_name=rec.get("hyperedge_name", ""),
                weight=1.0,
                source_chunk_hash=hashlib.md5(
                    rec.get("hyperedge_name", "").encode()
                ).hexdigest(),
            )
            for rec in he_slice
        ]

        return ClientExtractionResult(
            client_id=str(self.client_id),
            round_number=round_id,
            entities=entities,
            hyperedges=hyperedges,
            metadata={
                "source":       "pretrained_kg",
                "kg_dir":       getattr(self.args, "pretrained_kg_dir", ""),
                "chunk_idx":    chunk_idx,
                "total_chunks": total_chunks,
            },
        )

    def _init_graphr1(self):
        """Create a per-client GraphR1 instance with local working_dir.

        Uses TimelyGPT when TIMELY_API_KEY (or legacy TGPT_API_KEY) is set,
        falls back to standard OpenAI otherwise.
        """
        try:
            from graphr1.graphr1 import GraphR1
            from fedgraphr1.fl.config import _load_dotenv

            import os
            _load_dotenv()
            if os.environ.get("TIMELY_API_KEY") or os.environ.get("TGPT_API_KEY"):
                from graphr1.llm import timelygpt_complete, timelygpt_embedding
                llm_func = timelygpt_complete
                emb_func = timelygpt_embedding
                # TimelyGPT has a server-side execution timeout.
                # Use smaller chunks (fewer tokens per prompt) and skip gleaning
                # rounds to keep each call fast enough to avoid 504 errors.
                _cts = getattr(self.args, "chunk_token_size", None)
                chunk_token_size = _cts if _cts is not None else 400
                _mg = getattr(self.args, "max_gleaning", None)
                max_gleaning = _mg if _mg is not None else 0
            else:
                from graphr1.llm import gpt_4o_mini_complete, openai_embedding
                llm_func = gpt_4o_mini_complete
                emb_func = openai_embedding
                _cts = getattr(self.args, "chunk_token_size", None)
                chunk_token_size = _cts if _cts is not None else 1200
                _mg = getattr(self.args, "max_gleaning", None)
                max_gleaning = _mg if _mg is not None else 1

            return GraphR1(
                working_dir=self._working_dir,
                llm_model_func=llm_func,
                embedding_func=emb_func,
                enable_llm_cache=True,
                chunk_token_size=chunk_token_size,
                entity_extract_max_gleaning=max_gleaning,
            )
        except Exception as e:
            logger.warning(
                f"[Client {self.client_id}] Could not initialise GraphR1: {e}. "
                "Entity extraction will be disabled."
            )
            return None

    def _init_grpo_client(self):
        """Create the FederatedGRPOClient for local LoRA training."""
        base_model = getattr(self.args, "base_model", None)
        if base_model is None:
            logger.info(
                f"[Client {self.client_id}] No base_model specified — "
                "LoRA training disabled."
            )
            return None
        try:
            from fedgraphr1.client.federated_grpo_client import FederatedGRPOClient
            target_modules = [
                m.strip()
                for m in getattr(
                    self.args, "lora_modules", "q_proj,k_proj,v_proj,o_proj"
                ).split(",")
            ]
            return FederatedGRPOClient(
                base_model_path=base_model,
                lora_rank=getattr(self.args, "lora_rank", 16),
                lora_alpha=getattr(self.args, "lora_alpha", 32),
                lora_target_modules=target_modules,
                fedprox_mu=getattr(self.args, "fedprox_mu", 0.0),
                device=str(self.device),
            )
        except Exception as e:
            logger.warning(
                f"[Client {self.client_id}] Could not init FederatedGRPOClient: {e}"
            )
            return None

    def attach_embedding_model(self, model):
        """Attach an embedding model after construction (late binding).

        Updates both the HypergraphReceiver (for FAISS index building) and the
        FederatedSearchTool (for query encoding).  Call this before the first
        FL round when the model is loaded asynchronously.
        """
        self._embedding_model = model
        self._receiver.embedding_model = model
        self._search_tool.set_embedding_model(model)
        logger.info(
            f"[Client {self.client_id}] Embedding model attached."
        )

    def get_tool_env(self):
        """Return a ToolEnv backed by this client's FederatedSearchTool.

        Used by FederatedRayPPOTrainer during the rollout phase so the LLM
        can query the local knowledge graph while generating responses.

        Returns None if agent.tool is not available.
        """
        try:
            from agent.tool.tool_env import ToolEnv
            return ToolEnv(
                tools=[self._search_tool],
                max_turns=getattr(self.args, "max_tool_turns", 10),
            )
        except ImportError:
            logger.warning(
                f"[Client {self.client_id}] agent.tool not available — "
                "ToolEnv not created."
            )
            return None

    def _run_local_grpo(self, round_id: int, num_epochs: int) -> Dict:
        """Run local GRPO training epochs.

        Builds a FederatedRayPPOTrainer on first call (lazy init) so the
        heavy model loading is deferred until actually needed.

        Plan §4.4.1 "FederatedRayPPOTrainer.train_local_epochs()"
        """
        if not hasattr(self, "_ray_trainer") or self._ray_trainer is None:
            self._ray_trainer = self._init_ray_trainer()

        if self._ray_trainer is not None:
            return self._ray_trainer.train_local_epochs(num_epochs)

        # No trainer available — return NaN metrics (entity exchange only)
        logger.debug(
            f"[Client {self.client_id}] No GRPO trainer — skipping training."
        )
        return {"policy_loss": float("nan"), "kl_divergence": float("nan")}

    def _init_ray_trainer(self):
        """Lazily construct FederatedRayPPOTrainer.

        Returns None if the GRPO client or local data is unavailable.
        """
        train_data = self.local_examples if self.local_examples else self.local_data
        if self._grpo_client is None or not train_data:
            return None
        try:
            from fedgraphr1.trainer.federated_ray_trainer import FederatedRayPPOTrainer
            trainer = FederatedRayPPOTrainer(
                grpo_client=self._grpo_client,
                local_dataset=train_data,
                args=self.args,
                search_tool=self._search_tool,
                device=self.device,
            )
            logger.info(
                f"[Client {self.client_id}] FederatedRayPPOTrainer initialised."
            )
            return trainer
        except Exception as e:
            logger.warning(
                f"[Client {self.client_id}] Could not init FederatedRayPPOTrainer: {e}"
            )
            return None
