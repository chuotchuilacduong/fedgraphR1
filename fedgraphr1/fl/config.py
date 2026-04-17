"""
Federated Graph-R1 configuration.

Replaces fl/fl_config.py (FLConfig dataclass) with an argparse-based
namespace that mirrors the FedGM/config.py style.  Import get_args()
in main.py and pass the returned namespace through the entire stack.
"""

import argparse
import os


def _load_dotenv(path: str = None):
    """Load key=value pairs from .env file into os.environ (no overwrite)."""
    env_path = path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
    )
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def get_args() -> argparse.Namespace:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        description="Federated Graph-R1 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── FL round settings ──────────────────────────────────────────────
    parser.add_argument("--num_clients",  type=int,   default=3,
                        help="Total number of federated clients")
    parser.add_argument("--num_rounds",   type=int,   default=40,
                        help="Number of FL communication rounds")
    parser.add_argument("--client_frac",  type=float, default=1.0,
                        help="Fraction of clients sampled per round (0 < f ≤ 1)")
    parser.add_argument("--num_epochs",   type=int,   default=1,
                        help="Local GRPO epochs per round")

    # ── Model ──────────────────────────────────────────────────────────
    parser.add_argument("--base_model",   type=str,
                        default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
                        help="HuggingFace model ID or local path for the base LLM")
    parser.add_argument("--lora_rank",    type=int,   default=16,
                        help="LoRA rank r")
    parser.add_argument("--lora_alpha",   type=int,   default=32,
                        help="LoRA scaling alpha")
    parser.add_argument("--lora_modules", type=str,   default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated list of target modules for LoRA")

    # ── Optimiser ─────────────────────────────────────────────────────
    parser.add_argument("--lr",           type=float, default=5e-7,
                        help="Local GRPO learning rate")
    parser.add_argument("--train_batch_size", type=int, default=42,
                        help="Per-client batch size (~128 / num_clients)")
    parser.add_argument("--kl_loss_coef", type=float, default=0.001)
    parser.add_argument("--kl_loss_type", type=str,   default="low_var_kl")
    parser.add_argument("--fedprox_mu",   type=float, default=0.0,
                        help="FedProx proximal term coefficient (0 = FedAvg)")

    # ── Hypergraph distribution ────────────────────────────────────────
    parser.add_argument("--distribution_strategy", type=str,
                        default="full_broadcast",
                        choices=["full_broadcast", "relevance_based"],
                        help="How the server distributes the global hypergraph")

    # ── Dataset ───────────────────────────────────────────────────────
    parser.add_argument("--dataset",          type=str, default="2WikiMultiHopQA")
    parser.add_argument("--simulation_mode",  type=str, default="topic_skew",
                        choices=["topic_skew", "iid", "dirichlet"],
                        help="Non-IID data split strategy across clients")
    parser.add_argument("--dirichlet_alpha",  type=float, default=0.5,
                        help="Dirichlet concentration (lower = more skewed)")

    # ── Communication compression (P4.6) ──────────────────────────────
    parser.add_argument("--lora_top_k_ratio", type=float, default=None,
                        help="Top-k sparsification ratio for LoRA uplink "
                             "(e.g. 0.1 = keep top 10%% per tensor). "
                             "None = no sparsification (dense FedAvg).")

    # ── Misc ──────────────────────────────────────────────────────────
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--device", type=str,   default="auto",
                        help="'auto', 'cpu', 'cuda', or 'cuda:N'")
    parser.add_argument("--working_dir", type=str, default="expr/fedgraphr1",
                        help="Root directory for checkpoints and artefacts")

    return parser.parse_args()
