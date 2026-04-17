"""fedgraphr1.trainer — extended training engine for Federated GRPO."""

from .federated_ray_trainer import FederatedRayPPOTrainer

__all__ = ["FederatedRayPPOTrainer"]
