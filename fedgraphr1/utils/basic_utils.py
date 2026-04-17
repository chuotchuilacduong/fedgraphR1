"""
fedgraphr1/utils/basic_utils.py
================================
Factory functions for instantiating GraphR1Client / GraphR1Server.

Mirrors FedGM/utils/basic_utils.py — load_client() / load_server().
The registry pattern means main.py never imports concrete classes directly,
keeping the FL algorithm swappable without editing driver code.

Plan reference: Section 6.3 (Directory structure)
               Section 6.4 (FedGM-Style FL Implementation)
"""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import Dict

logger = logging.getLogger("fedgraphr1")

# Registry maps fl_algorithm name → (ClientClass, ServerClass).
# Populated at module load time by the algorithm packages themselves.
_CLIENT_REGISTRY: Dict[str, type] = {}
_SERVER_REGISTRY: Dict[str, type] = {}


def register_algorithm(name: str, client_cls: type, server_cls: type):
    """Register a (client, server) pair under *name*.

    Called once at import time by each algorithm subpackage, e.g.::

        register_algorithm("fedavg", FedAvgClient, FedAvgServer)
    """
    _CLIENT_REGISTRY[name] = client_cls
    _SERVER_REGISTRY[name] = server_cls
    logger.debug(f"Registered FL algorithm '{name}'")


def load_client(
    args: Namespace,
    client_id: int,
    local_data: object,
    message_pool: dict,
    device,
):
    """Instantiate the Client class for *args.fl_algorithm*.

    Args:
        args: Parsed CLI namespace (must have .fl_algorithm attribute).
        client_id: Integer client index.
        local_data: Dataset partition for this client.
        message_pool: Shared in-memory dict (FedGM message-pool pattern).
        device: torch.device or string.

    Returns:
        Instantiated BaseClient subclass.
    """
    algo = getattr(args, "fl_algorithm", "graphr1")
    if algo not in _CLIENT_REGISTRY:
        # Lazy import to populate the registry on first use
        _ensure_default_registered()
    cls = _CLIENT_REGISTRY.get(algo)
    if cls is None:
        raise ValueError(
            f"Unknown fl_algorithm '{algo}'. "
            f"Available: {list(_CLIENT_REGISTRY.keys())}"
        )
    logger.debug(f"Loading client {client_id} — algorithm={algo}")
    return cls(
        client_id=client_id,
        args=args,
        local_data=local_data,
        message_pool=message_pool,
        device=device,
    )


def load_server(
    args: Namespace,
    message_pool: dict,
    device,
):
    """Instantiate the Server class for *args.fl_algorithm*.

    Args:
        args: Parsed CLI namespace.
        message_pool: Shared in-memory dict.
        device: torch.device or string.

    Returns:
        Instantiated BaseServer subclass.
    """
    algo = getattr(args, "fl_algorithm", "graphr1")
    if algo not in _SERVER_REGISTRY:
        _ensure_default_registered()
    cls = _SERVER_REGISTRY.get(algo)
    if cls is None:
        raise ValueError(
            f"Unknown fl_algorithm '{algo}'. "
            f"Available: {list(_SERVER_REGISTRY.keys())}"
        )
    logger.debug(f"Loading server — algorithm={algo}")
    return cls(args=args, message_pool=message_pool, device=device)


def _ensure_default_registered():
    """Import the default GraphR1 algorithm package to populate the registry."""
    try:
        from fedgraphr1.fl.client import GraphR1Client  # noqa: F401
        from fedgraphr1.fl.server import GraphR1Server  # noqa: F401
    except ImportError as e:
        logger.warning(f"Could not auto-register default algorithm: {e}")
