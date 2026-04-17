"""
fedgraphr1/data/partitioner.py
================================
Dataset partitioning strategies for simulating non-IID federated clients.

Three strategies (matching fl/config.py `--simulation_mode`):
  - "iid"          : uniform random split
  - "topic_skew"   : each client gets documents from a disjoint topic subset
  - "dirichlet"    : Dirichlet-distribution-based label skew (α controls skew)

Input: a list/dict of QA examples (HotpotQA / 2WikiMultiHopQA format).
Output: per-client list of example indices.

Plan reference: Section 1.3 (Quy trình Tổng thể — Client local data D_k)
               Phase 1 deliverable: data/partitioner.py
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger("fedgraphr1")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def partition_dataset(
    examples: Sequence,
    num_clients: int,
    strategy: str = "iid",
    seed: int = 42,
    *,
    dirichlet_alpha: float = 0.5,
    topic_key: Optional[str] = None,
) -> Dict[int, List[int]]:
    """Partition *examples* into *num_clients* client shards.

    Args:
        examples: Sequence of dataset items (dicts or any indexable).
        num_clients: Number of client partitions to produce.
        strategy: "iid" | "topic_skew" | "dirichlet".
        seed: Random seed for reproducibility.
        dirichlet_alpha: Concentration parameter α for Dirichlet strategy.
            Lower α → more heterogeneous.  Only used when strategy="dirichlet".
        topic_key: Key inside each example dict that holds the topic label.
            Required when strategy="topic_skew", optional otherwise.

    Returns:
        Dict mapping client_id (0-indexed int) → list of example indices.

    Plan §3 (simulation_mode arg in fl/config.py)
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    n = len(examples)
    indices = list(range(n))

    if strategy == "iid":
        return _iid_split(indices, num_clients, rng)
    elif strategy == "dirichlet":
        # Extract a numeric label from each example for Dirichlet skew.
        # For QA datasets we fall back to a pseudo-label derived from hash.
        labels = _extract_labels(examples, topic_key, rng)
        return _dirichlet_split(indices, labels, num_clients, dirichlet_alpha)
    elif strategy == "topic_skew":
        labels = _extract_labels(examples, topic_key, rng)
        return _topic_skew_split(indices, labels, num_clients, rng)
    else:
        raise ValueError(
            f"Unknown partition strategy '{strategy}'. "
            "Choose from: 'iid', 'topic_skew', 'dirichlet'."
        )


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _iid_split(
    indices: List[int],
    num_clients: int,
    rng: random.Random,
) -> Dict[int, List[int]]:
    """Uniform random split — equal-ish shard size, same distribution.

    Plan §simulation_mode=iid
    """
    shuffled = indices[:]
    rng.shuffle(shuffled)
    splits = {}
    chunk = len(shuffled) // num_clients
    remainder = len(shuffled) % num_clients
    start = 0
    for cid in range(num_clients):
        extra = 1 if cid < remainder else 0
        end = start + chunk + extra
        splits[cid] = shuffled[start:end]
        start = end
    logger.info(
        f"IID split: {num_clients} clients, "
        f"sizes={[len(v) for v in splits.values()]}"
    )
    return splits


def _dirichlet_split(
    indices: List[int],
    labels: List[int],
    num_clients: int,
    alpha: float,
) -> Dict[int, List[int]]:
    """Dirichlet-distribution-based non-IID split.

    Each class c is distributed across clients according to a Dirichlet
    sample Dir(α).  Lower α → more concentrated (heterogeneous).

    Plan §simulation_mode=dirichlet, §dirichlet_alpha
    """
    # Group indices by label
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, lbl in zip(indices, labels):
        label_to_indices[lbl].append(idx)

    splits: Dict[int, List[int]] = {cid: [] for cid in range(num_clients)}

    for lbl, lbl_indices in label_to_indices.items():
        np.random.shuffle(lbl_indices)
        # Sample Dirichlet proportions for this label across clients
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Convert to integer counts
        counts = (proportions * len(lbl_indices)).astype(int)
        # Assign any remainder to a random client
        counts[np.random.randint(num_clients)] += len(lbl_indices) - counts.sum()
        ptr = 0
        for cid, cnt in enumerate(counts):
            splits[cid].extend(lbl_indices[ptr: ptr + cnt])
            ptr += cnt

    sizes = [len(v) for v in splits.values()]
    logger.info(
        f"Dirichlet split (α={alpha}): {num_clients} clients, "
        f"sizes={sizes}"
    )
    return splits


def _topic_skew_split(
    indices: List[int],
    labels: List[int],
    num_clients: int,
    rng: random.Random,
) -> Dict[int, List[int]]:
    """Topic-skew split: each client receives documents from a disjoint subset
    of topics (labels).

    Topics are assigned to clients in round-robin order after shuffling.
    This simulates the realistic scenario where different hospitals / organisations
    hold data about different subject domains.

    Plan §simulation_mode=topic_skew
    """
    # Group indices by topic label
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, lbl in zip(indices, labels):
        label_to_indices[lbl].append(idx)

    unique_labels = list(label_to_indices.keys())
    rng.shuffle(unique_labels)

    splits: Dict[int, List[int]] = {cid: [] for cid in range(num_clients)}

    for i, lbl in enumerate(unique_labels):
        cid = i % num_clients
        splits[cid].extend(label_to_indices[lbl])

    sizes = [len(v) for v in splits.values()]
    logger.info(
        f"Topic-skew split: {num_clients} clients, "
        f"sizes={sizes}  ({len(unique_labels)} topics)"
    )
    return splits


# ---------------------------------------------------------------------------
# Label extraction helper
# ---------------------------------------------------------------------------


def _extract_labels(
    examples: Sequence,
    topic_key: Optional[str],
    rng: random.Random,
) -> List[int]:
    """Extract integer labels from *examples* for use in non-IID strategies.

    Priority:
    1. If *topic_key* is given and present in examples[0], use that field.
    2. Otherwise fall back to a hash-based pseudo-label with ~10 buckets.

    Returns a list of int labels, one per example.
    """
    if topic_key and isinstance(examples[0], dict) and topic_key in examples[0]:
        # Build a stable label map from unique topic strings
        topics = list({ex[topic_key] for ex in examples})
        rng.shuffle(topics)
        topic_to_int = {t: i for i, t in enumerate(topics)}
        return [topic_to_int[ex[topic_key]] for ex in examples]

    # Fallback: hash the first 50 chars of the "question" field (if present)
    # into ~10 pseudo-classes.
    labels = []
    for ex in examples:
        if isinstance(ex, dict):
            text = ex.get("question", ex.get("input", str(ex)))
        else:
            text = str(ex)
        labels.append(int(hashlib.md5(text[:50].encode()).hexdigest(), 16) % 10)
    return labels
