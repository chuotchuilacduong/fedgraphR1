"""fedgraphr1.fl — FL round-loop, base classes, config, client, server."""

from .base import BaseClient, BaseServer
from .client import GraphR1Client
from .server import GraphR1Server
from .trainer import GraphR1Trainer
from .config import get_args

# Auto-register the default GraphR1 algorithm so load_client / load_server work
from fedgraphr1.utils.basic_utils import register_algorithm
register_algorithm("graphr1", GraphR1Client, GraphR1Server)

__all__ = [
    "BaseClient",
    "BaseServer",
    "GraphR1Client",
    "GraphR1Server",
    "GraphR1Trainer",
    "get_args",
]
