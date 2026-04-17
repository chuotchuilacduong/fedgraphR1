"""
fedgraphr1 — Federated Graph-R1: Knowledge Hypergraph RAG in a Federated
Learning setting.

Architecture summary (see implement_plan.md §2):
  client/         — entity extraction, packing, local FAISS retrieval, GRPO
  server/         — entity aggregation, global KG builder, LoRA FedAvg
  fl/             — BaseClient/BaseServer, GraphR1Client/Server, trainer
  trainer/        — FederatedRayPPOTrainer (passive GRPO engine)
  data/           — non-IID dataset partitioning
  utils/          — compression, metrics, factory functions
  types.py        — shared dataclasses

Quick start (simulation mode)::

    from fedgraphr1.fl import get_args, GraphR1Client, GraphR1Server, GraphR1Trainer

    args = get_args()
    # Build clients and server, then run:
    trainer = GraphR1Trainer(args, clients, server)
    trainer.train()
"""

__version__ = "0.1.0"
