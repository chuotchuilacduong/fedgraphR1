"""
fedgraphr1/fl/server.py
=========================
GraphR1Server — the main federated server for Federated Graph-R1.

Per-round execution sequence (§1.3 / §4.3.3):
  1. execute()        (called AFTER all clients complete)
     a. Snapshot KG fingerprint BEFORE aggregation
     b. Aggregate entity packets → update G_H^global  ← KG improves each round
     c. Snapshot KG fingerprint AFTER aggregation → compute KGDelta
     d. FedAvg LoRA weights → global_lora_state
     e. Rebuild partitioner if KG changed
     f. Persist updated KV stores to disk (so evaluator can reload FAISS)
  2. send_message()   (called BEFORE clients execute)
     a. If KG changed: broadcast LoRA + updated fragment to each client
     b. If KG unchanged: broadcast LoRA only — skip fragment (bandwidth saving)

Key contribution wiring:
  Each round clients process a new document batch (pre-determined schedule) →
  richer entity packets → server merges → G_H^global grows → clients get
  richer fragments → better retrieval context for π_θ GRPO → better QA reward.
  π_ext (TimelyGPT/GPT-4o-mini) is fixed; only the document batching schedule
  drives KG growth.  π_θ (Qwen2.5+LoRA) is the only model that trains.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Set

from fedgraphr1.fl.base import BaseServer
from fedgraphr1.shared_types import ClientExtractionResult
from fedgraphr1.utils.debug_logger import step

logger = logging.getLogger("fedgraphr1")


class GraphR1Server(BaseServer):
    """Federated Graph-R1 server.

    Args:
        args:              Parsed CLI namespace.
        message_pool:      Shared in-memory dict (FedGM message-pool pattern).
        device:            torch.device or string.
        graphr1_instance:  Optional pre-initialised GraphR1 instance for the
                           server's global KG.  If None, one is created.
    """

    def __init__(
        self,
        args,
        message_pool: dict,
        device,
        graphr1_instance=None,
    ):
        super().__init__(args, message_pool, device)

        self._working_dir = os.path.join(
            getattr(args, "working_dir", "expr/fedgraphr1"),
            "server",
        )
        os.makedirs(self._working_dir, exist_ok=True)

        # ── Global KG (G_H^global) — the graphr1 base instance ──────────
        self._graphr1 = graphr1_instance or self._init_graphr1()

        # ── Sub-modules ─────────────────────────────────────────────────
        from fedgraphr1.server.hypergraph_builder import GlobalHypergraphBuilder
        from fedgraphr1.server.hypergraph_partitioner import HypergraphPartitioner
        from fedgraphr1.server.fragment_distributor import HypergraphFragmentDistributor
        from fedgraphr1.server.lora_aggregator import FederatedLoRAServer

        self._kg_builder = GlobalHypergraphBuilder(
            graphr1_instance=self._graphr1,
            embedding_model=None,
            use_llm_summarize=False,
        )
        self._distributor = HypergraphFragmentDistributor()
        self._lora_server = FederatedLoRAServer(aggregation_strategy="fedavg")

        self._distribution_strategy = getattr(
            args, "distribution_strategy", "full_broadcast"
        )

        # ── Per-client entity contribution tracking ──────────────────────
        self._client_entity_sets: Dict[int, Set[str]] = {}

        # ── Partitioner — rebuilt each round when KG changes ─────────────
        self._partitioner = None

        # ── KG delta / efficiency protocol ──────────────────────────────
        # _kg_fingerprint: (entity_count, hyperedge_count, edge_count, name_hash)
        # Computed after each execute(); read in send_message() next round.
        self._kg_fingerprint = None   # None = no KG yet (round 0)
        self._kg_changed: bool = True  # force first broadcast
        self._last_kg_delta = None     # KGDelta exposed to trainer for W&B

    # ------------------------------------------------------------------
    # BaseServer interface
    # ------------------------------------------------------------------

    def initialize(self):
        """One-time server initialisation (called before round 0).

        Global KG starts empty; enriched incrementally each round.
        """
        logger.info("[GraphR1Server] Initialised (empty global KG).")

    def send_message(self):
        """Push global LoRA + (optionally) hypergraph fragments to clients.

        Efficiency protocol:
          - Round 0: no LoRA, no fragment (clients start from scratch).
          - KG changed: broadcast LoRA + updated fragment for each client.
          - KG unchanged: broadcast LoRA only; clients reuse cached fragment.
            This saves the full KG serialisation overhead for unchanged rounds.
        """
        round_id = self.message_pool.get("round", 0)
        sampled_clients = self.message_pool.get("sampled_clients", [])
        lora_weights = self._lora_server.global_lora_state  # None in round 0

        # ── Round 0 / no partitioner yet: send empty payload ────────────
        if self._partitioner is None:
            payload = {
                "lora_weights": lora_weights,
                "hypergraph_fragment": None,
                "kg_unchanged": False,
                "config": {"distribution_strategy": self._distribution_strategy},
            }
            self.message_pool["server"] = payload
            for cid in sampled_clients:
                self.message_pool[f"server_for_{cid}"] = dict(payload)
            return

        # ── Decide whether to broadcast the KG fragment ─────────────────
        # Skip fragment entirely if the global KG did not change this round.
        broadcast_kg = self._kg_changed

        skipped = 0
        for cid in sampled_clients:
            fragment = None
            if broadcast_kg:
                client_entities = self._client_entity_sets.get(cid, set())
                try:
                    fragment = self._partitioner.partition_for_client(
                        client_id=str(cid),
                        client_entity_names=client_entities,
                        round_number=round_id,
                    )
                    fragment = self._distributor.distribute_in_memory(fragment)
                except Exception as e:
                    logger.warning(
                        f"[GraphR1Server] Could not build fragment for client {cid}: {e}"
                    )
            else:
                skipped += 1

            self.message_pool[f"server_for_{cid}"] = {
                "lora_weights": (
                    {k: v.clone() for k, v in lora_weights.items()}
                    if lora_weights else None
                ),
                "hypergraph_fragment": fragment,   # None = reuse cached
                "kg_unchanged": not broadcast_kg,
                "config": {"distribution_strategy": self._distribution_strategy},
            }

        # Global fallback
        self.message_pool["server"] = {
            "lora_weights": lora_weights,
            "hypergraph_fragment": None,
            "kg_unchanged": not broadcast_kg,
            "config": {},
        }

        if skipped:
            logger.info(
                f"[GraphR1Server] Round {round_id}: KG unchanged — "
                f"fragment broadcast skipped for {skipped} client(s) "
                f"(bandwidth saved)."
            )
        else:
            logger.info(
                f"[GraphR1Server] Round {round_id}: KG updated — "
                f"broadcasting fragments to {len(sampled_clients)} client(s)."
            )

    def execute(self):
        """Aggregate client results and update global state.

        Reads:  message_pool["client_{cid}"] for each sampled client
        Writes: _lora_server.global_lora_state, _partitioner,
                _kg_fingerprint, _kg_changed, _last_kg_delta
        """
        from fedgraphr1.server.kg_diff import (
            compute_kg_fingerprint,
            compute_kg_delta,
        )

        round_id = self.message_pool.get("round", 0)
        sampled_clients = self.message_pool.get("sampled_clients", [])

        step(f"  Server │ execute() round={round_id}",
             {"sampled_clients": sampled_clients})

        # ── Collect client messages ─────────────────────────────────────
        extraction_results: List[ClientExtractionResult] = []
        lora_updates = []

        for cid in sampled_clients:
            client_msg = self.message_pool.get(f"client_{cid}", {})
            if not client_msg:
                logger.warning(f"[GraphR1Server] Missing message from client {cid}")
                continue

            ext_result = client_msg.get("extraction_result")
            if ext_result is not None:
                extraction_results.append(ext_result)
                self._client_entity_sets[cid] = set(ext_result.entity_names())
                step(f"  Server │ [a] Received from client_{cid}",
                     {"entities": ext_result.num_entities,
                      "hyperedges": ext_result.num_hyperedges})

            lora_sd = client_msg.get("lora_weights", {})
            num_samples = client_msg.get("num_samples", 1)
            if lora_sd:
                lora_updates.append((str(cid), lora_sd, num_samples))

        # ── a. Snapshot KG BEFORE aggregation ──────────────────────────
        old_fp = self._kg_fingerprint
        try:
            old_graph = self._kg_builder.get_global_graph()
            old_fp = compute_kg_fingerprint(old_graph)
        except Exception:
            old_fp = self._kg_fingerprint  # keep last known

        # ── b. Merge client extractions → G_H^global grows ─────────────
        # This is the core per-round KG improvement step:
        # clients re-extracted with their GRPO-improved model, so the
        # entity/hyperedge packets are richer than the previous round.
        step(f"  Server │ [b] KG aggregation",
             {"extraction_results": len(extraction_results)})
        if extraction_results:
            self._kg_builder.update(extraction_results, round_id=round_id)

        # ── c. Snapshot KG AFTER aggregation → compute delta ───────────
        try:
            new_graph = self._kg_builder.get_global_graph()
            new_fp = compute_kg_fingerprint(new_graph)
        except Exception as e:
            logger.warning(f"[GraphR1Server] Could not compute new fingerprint: {e}")
            new_fp = old_fp

        delta = compute_kg_delta(old_fp, new_fp)
        self._kg_fingerprint = new_fp
        self._kg_changed = delta.changed
        self._last_kg_delta = delta

        step(f"  Server │ [c] KG delta",
             {"changed": delta.changed,
              "new_entities": delta.new_entities,
              "new_hyperedges": delta.new_hyperedges,
              "new_edges": delta.new_edges})

        # ── d. FedAvg LoRA weights ─────────────────────────────────────
        if lora_updates:
            step(f"  Server │ [d] FedAvg LoRA", {"clients": len(lora_updates)})
            self._lora_server.aggregate(lora_updates)
        else:
            step(f"  Server │ [d] LoRA skip", {"reason": "no updates"})

        # ── e. Rebuild partitioner if KG changed ───────────────────────
        if delta.changed:
            from fedgraphr1.server.hypergraph_partitioner import HypergraphPartitioner
            try:
                self._partitioner = HypergraphPartitioner(
                    global_graph=new_graph,
                    strategy=self._distribution_strategy,
                )
                step(f"  Server │ [e] Partitioner rebuilt",
                     {"nodes": new_graph.number_of_nodes(),
                      "edges": new_graph.number_of_edges()})
            except Exception as e:
                logger.warning(f"[GraphR1Server] Partitioner rebuild failed: {e}")
        else:
            step(f"  Server │ [e] Partitioner unchanged (KG stable)")

        # ── f. Persist KV stores to disk ───────────────────────────────
        # Required so RoundEvaluator / FederatedSearchTool.reload() picks up
        # the latest entity/hyperedge data when it rebuilds FAISS indices.
        self._persist_kv_stores()

        # ── Log ─────────────────────────────────────────────────────────
        stats = self._kg_builder.kg_stats()
        step(f"  Server │ Round {round_id} complete",
             {"entity_nodes":    stats.get("num_entity_nodes", "?"),
              "hyperedge_nodes": stats.get("num_hyperedge_nodes", "?"),
              "density":         stats.get("density", "?"),
              "kg_changed":      delta.changed})
        logger.info(
            f"[GraphR1Server] Round {round_id} — "
            f"KG: {stats.get('num_entity_nodes','?')} entities, "
            f"{stats.get('num_hyperedge_nodes','?')} hyperedges | "
            f"{delta.summary()}"
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_kg_stats(self) -> Dict:
        """Return current global KG quality statistics."""
        return self._kg_builder.kg_stats()

    def get_last_kg_delta(self):
        """Return the KGDelta from the most recent execute() call."""
        return self._last_kg_delta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist_kv_stores(self):
        """Flush updated entity/hyperedge KV stores to disk.

        Called after every aggregation so the server's working_dir always
        contains the latest KV store JSONs.  The RoundEvaluator and any
        FAISS rebuild logic read from this directory.
        """
        if self._graphr1 is None:
            return
        try:
            import asyncio

            async def _flush():
                g = self._graphr1
                for store in [g.entities_vdb, g.hyperedges_vdb]:
                    if store is not None:
                        await store.index_done_callback()
                # Also flush the graph itself
                kg = g.chunk_entity_relation_graph
                if kg is not None:
                    await kg.index_done_callback()

            loop = _get_or_create_event_loop()
            loop.run_until_complete(_flush())
        except Exception as e:
            logger.debug(f"[GraphR1Server] _persist_kv_stores: {e}")

    def _init_graphr1(self):
        """Create the server-side GraphR1 (graphr1 base) instance."""
        try:
            from graphr1.graphr1 import GraphR1
            from fedgraphr1.fl.config import _load_dotenv
            import os as _os
            _load_dotenv()

            if _os.environ.get("TIMELY_API_KEY") or _os.environ.get("TGPT_API_KEY"):
                from graphr1.llm import timelygpt_complete, timelygpt_embedding
                llm_func = timelygpt_complete
                emb_func = timelygpt_embedding
            else:
                from graphr1.llm import gpt_4o_mini_complete, openai_embedding
                llm_func = gpt_4o_mini_complete
                emb_func = openai_embedding

            return GraphR1(
                working_dir=self._working_dir,
                llm_model_func=llm_func,
                embedding_func=emb_func,
                enable_llm_cache=True,
            )
        except Exception as e:
            logger.warning(
                f"[GraphR1Server] Could not init GraphR1: {e}. "
                "KG building disabled."
            )
            return None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_or_create_event_loop():
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
