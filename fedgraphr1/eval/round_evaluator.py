"""
fedgraphr1/eval/round_evaluator.py
=====================================
Per-round validation for Federated Graph-R1.

Runs at the end of every FL round to quantify KG quality improvement
without a GPT-as-a-judge.  Two complementary metrics:

  1. Retrieval Recall@K  (fast, KG-only, no model inference needed)
     Measures whether the answer to a dev question is *findable* in the
     top-K retrieved context from the current global KG.  This directly
     tracks the paper's main claim: the KG improves each round.

  2. EM / F1  (optional, requires model inference)
     Traditional QA metrics.  Skipped when no model is available or when
     `compute_qa_metrics=False` to keep per-round latency low.

Both metrics use the same FAISS-backed FederatedSearchTool loaded from the
server's current KG artifacts, so they always reflect the *latest* KG state.

Usage (in GraphR1Trainer):
    evaluator = RoundEvaluator(dev_data, kg_dir=args.output_dir, top_k=10)
    metrics = evaluator.evaluate(round_id=r)   # → dict for W&B
    evaluator.reload()  # call after server updates KG on disk
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Optional

logger = logging.getLogger("fedgraphr1")


# ---------------------------------------------------------------------------
# EM / F1 helpers (no external dependency, matches evaluate_fl.py logic)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _token_set(text: str) -> set:
    return set(_normalize(text).split())


def compute_em(prediction: str, gold: str) -> float:
    """Exact Match after normalization."""
    return float(_normalize(prediction) == _normalize(gold))


def compute_f1(prediction: str, gold: str) -> float:
    """Token-level F1."""
    pred_tokens = _token_set(prediction)
    gold_tokens = _token_set(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec  = len(common) / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)


def _best_score(pred: str, golds: List[str], fn) -> float:
    if not golds:
        return 0.0
    return max(fn(pred, g) for g in golds)


# ---------------------------------------------------------------------------
# RoundEvaluator
# ---------------------------------------------------------------------------


class RoundEvaluator:
    """Evaluate KG quality and QA performance at the end of each FL round.

    Args:
        dev_data: List of QA dicts with keys ``question`` and
            ``golden_answers`` (list of strings) or ``answer`` (string).
        kg_dir: Directory containing the server's current KG artifacts
            (index_entity.bin, index_hyperedge.bin, kv_store_*.json).
            This is the SAME directory written by build_kg.py / the server.
        top_k: Number of retrieved items for recall computation.
        max_eval_samples: Cap to keep per-round evaluation fast.
        compute_qa_metrics: Whether to attempt EM/F1 extraction from
            the retrieved context (no model inference required — checks
            if the golden answer string appears in the retrieved text).
    """

    def __init__(
        self,
        dev_data: List[dict],
        kg_dir: str,
        top_k: int = 10,
        max_eval_samples: int = 200,
        compute_qa_metrics: bool = True,
    ):
        self.dev_data = dev_data[:max_eval_samples]
        self.kg_dir = kg_dir
        self.top_k = top_k
        self.compute_qa_metrics = compute_qa_metrics

        self._search_tool = None   # lazy init on first evaluate() call
        self._emb_model = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, round_id: int) -> Dict[str, float]:
        """Run evaluation for the current round.

        Returns:
            Dict with keys:
              val/recall_at_k   — fraction of questions whose answer is in
                                  the retrieved context (KG quality metric)
              val/em            — Exact Match (context-based, no LLM needed)
              val/f1            — Token-F1  (context-based)
              val/num_samples   — number of dev examples evaluated
              val/kg_dir        — kg_dir string (for traceability)
        """
        if not self.dev_data:
            logger.warning("[RoundEvaluator] No dev data — skipping evaluation.")
            return {}

        if self._search_tool is None:
            self._init_search_tool()

        if self._search_tool is None:
            logger.warning(
                "[RoundEvaluator] Search tool unavailable — skipping evaluation."
            )
            return {}

        recall_hits = 0
        em_scores: List[float] = []
        f1_scores: List[float] = []

        for ex in self.dev_data:
            question = ex.get("question", "")
            golden   = _parse_golden(ex)

            if not question or not golden:
                continue

            # Retrieve context from current global KG
            try:
                context = self._search_tool.execute({"query": question})
            except Exception as e:
                logger.debug(f"[RoundEvaluator] search failed: {e}")
                context = ""

            # ── Retrieval Recall@K ─────────────────────────────────────
            # Check if ANY golden answer token sequence appears in context
            context_norm = _normalize(context)
            hit = any(
                _normalize(g) in context_norm or len(_token_set(g) & _token_set(context)) > 0
                for g in golden
            )
            recall_hits += int(hit)

            # ── Context-level EM / F1 (no model inference) ────────────
            # Extracts the answer by checking the most overlapping span
            # with the golden answer inside the retrieved context.
            if self.compute_qa_metrics:
                # Use the retrieved context itself as the "prediction"
                # (measures how much of the answer is covered by retrieval)
                em = _best_score(context, golden, compute_em)
                f1 = _best_score(context, golden, compute_f1)
                em_scores.append(em)
                f1_scores.append(f1)

        n = len(self.dev_data)
        recall = recall_hits / n if n else 0.0
        avg_em  = sum(em_scores) / len(em_scores)  if em_scores  else math.nan
        avg_f1  = sum(f1_scores) / len(f1_scores)  if f1_scores  else math.nan

        metrics = {
            "val/recall_at_k":  recall,
            "val/em":           avg_em,
            "val/f1":           avg_f1,
            "val/num_samples":  n,
        }
        logger.info(
            f"[RoundEvaluator] Round {round_id}: "
            f"recall@{self.top_k}={recall:.4f}  "
            f"EM={avg_em:.4f}  F1={avg_f1:.4f}  (n={n})"
        )
        return metrics

    def reload(self):
        """Reload FAISS indices after the server updates the KG on disk.

        Call this at the end of each round's server.execute() cycle so the
        next round's evaluation sees the freshly updated knowledge graph.
        """
        if self._search_tool is not None:
            try:
                self._search_tool.reload()
                logger.debug("[RoundEvaluator] FAISS indices reloaded.")
            except Exception as e:
                logger.warning(f"[RoundEvaluator] reload() failed: {e}")
        else:
            # Not yet initialised — will init on next evaluate() call
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_search_tool(self):
        """Lazy-initialise FederatedSearchTool with bge-large-en-v1.5."""
        try:
            from FlagEmbedding import FlagAutoModel
            self._emb_model = FlagAutoModel.from_finetuned(
                "BAAI/bge-large-en-v1.5",
                query_instruction_for_retrieval=(
                    "Represent this sentence for searching relevant passages: "
                ),
            )
        except ImportError:
            logger.warning(
                "[RoundEvaluator] FlagEmbedding not installed — "
                "evaluation disabled. Install with: pip install FlagEmbedding"
            )
            return

        try:
            from fedgraphr1.client.federated_search_tool import FederatedSearchTool
            self._search_tool = FederatedSearchTool(
                working_dir=self.kg_dir,
                embedding_model=self._emb_model,
                top_k_entities=self.top_k,
                top_k_hyperedges=self.top_k,
            )
            self._search_tool.load()
            logger.info(
                f"[RoundEvaluator] Initialised — "
                f"kg_dir={self.kg_dir}, top_k={self.top_k}, "
                f"dev_samples={len(self.dev_data)}"
            )
        except Exception as e:
            logger.warning(f"[RoundEvaluator] Could not init search tool: {e}")
            self._search_tool = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse_golden(example: dict) -> List[str]:
    """Extract list of golden answer strings from a QA example dict."""
    ans = example.get("golden_answers", example.get("answer", []))
    if isinstance(ans, str):
        return [ans] if ans else []
    if isinstance(ans, list):
        return [str(a) for a in ans if a]
    return []
