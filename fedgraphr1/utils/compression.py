"""
fedgraphr1/utils/compression.py
================================
zstd-based compression helpers.

Used by:
  - entity_packer.py       (client-side entity packet compression)
  - fragment_distributor.py (server-side hypergraph fragment compression)

Plan reference: Section 3.1.2 (Bandwidth Optimization — zstd compression)
               Section 3.2.2 (Fragment Compression & Streaming)

Assumption: `zstandard` package is installed (pip install zstandard).
If unavailable the module falls back gracefully to no-op identity
compression so unit tests can run without the optional dependency.
"""

from __future__ import annotations

import json
import logging
from typing import Union

logger = logging.getLogger("fedgraphr1")

try:
    import zstandard as zstd
    _ZSTD_AVAILABLE = True
except ImportError:
    _ZSTD_AVAILABLE = False
    logger.warning(
        "zstandard not installed — compression disabled. "
        "Install with: pip install zstandard"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compress(data: bytes, level: int = 3) -> bytes:
    """Compress *data* with zstd at *level*.

    Falls back to identity (no compression) if zstandard is not installed.

    Args:
        data: Raw bytes to compress.
        level: zstd compression level (1–22, default 3 is fast + decent ratio).

    Returns:
        Compressed bytes (or raw bytes if zstd unavailable).
    """
    if not _ZSTD_AVAILABLE:
        return data
    compressor = zstd.ZstdCompressor(level=level)
    return compressor.compress(data)


def decompress(data: bytes) -> bytes:
    """Decompress zstd-compressed *data*.

    Falls back to identity if zstandard is not installed.

    Args:
        data: Compressed bytes (or raw bytes if zstd was unavailable on compress).

    Returns:
        Decompressed bytes.
    """
    if not _ZSTD_AVAILABLE:
        return data
    decompressor = zstd.ZstdDecompressor()
    return decompressor.decompress(data)


def compress_json(obj: object, level: int = 3) -> bytes:
    """JSON-serialise *obj* and compress with zstd.

    Convenience wrapper used by EntityPacker / HypergraphFragmentDistributor
    for simulation-mode serialisation (no Protobuf required).

    Args:
        obj: JSON-serialisable Python object.
        level: zstd compression level.

    Returns:
        Compressed JSON bytes.
    """
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return compress(raw, level=level)


def decompress_json(data: bytes) -> object:
    """Decompress *data* and JSON-deserialise.

    Args:
        data: Bytes from compress_json().

    Returns:
        Deserialised Python object.
    """
    raw = decompress(data)
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# Size reporting (useful for bandwidth profiling)
# ---------------------------------------------------------------------------


def compression_ratio(original: bytes, compressed: bytes) -> float:
    """Return *original_size / compressed_size* (> 1 means compression helped)."""
    if len(compressed) == 0:
        return float("inf")
    return len(original) / len(compressed)
