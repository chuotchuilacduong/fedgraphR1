"""
Abstract base classes for Graph-R1 federated learning.

Mirrors FedGM/flcore/base.py — all communication happens through
the shared in-memory message_pool dict; no network I/O is needed
for simulation.  For distributed deployment, replace message_pool
with an async queue or Redis without touching any subclass logic.
"""


class BaseClient:
    """Abstract federated client.

    Subclasses must implement:
      execute()      — run one round of local work (training, extraction)
      send_message() — write results to message_pool["client_{client_id}"]
    """

    def __init__(self, client_id: int, args, message_pool: dict, device):
        self.client_id = client_id
        self.args = args
        self.message_pool = message_pool
        self.device = device

    # ------------------------------------------------------------------
    # Round interface (called by GraphR1Trainer)
    # ------------------------------------------------------------------

    def execute(self):
        """Perform local work for the current round.

        Reads:  message_pool["server"]  (or message_pool["server_for_{id}"])
        Writes: internal state (_last_metrics, _entities, …)
        """
        raise NotImplementedError

    def send_message(self):
        """Push local results into the message pool.

        Writes: message_pool["client_{self.client_id}"]
        """
        raise NotImplementedError


class BaseServer:
    """Abstract federated server.

    Subclasses must implement:
      initialize()   — round-0 cold start (create initial weights, etc.)
      send_message() — write global state to message_pool["server"]
      execute()      — aggregate client messages and update global state
    """

    def __init__(self, args, message_pool: dict, device):
        self.args = args
        self.message_pool = message_pool
        self.device = device

    # ------------------------------------------------------------------
    # Round interface (called by GraphR1Trainer)
    # ------------------------------------------------------------------

    def initialize(self):
        """One-time setup before the first round."""
        raise NotImplementedError

    def send_message(self):
        """Push global model + hypergraph fragment(s) into the message pool.

        Writes: message_pool["server"]
                message_pool["server_for_{cid}"]  (per-client, optional)
        """
        raise NotImplementedError

    def execute(self):
        """Aggregate client results and update global state.

        Reads:  message_pool["client_{cid}"] for each sampled client
        Writes: self.global_lora_weights, self.global_hypergraph
        """
        raise NotImplementedError
