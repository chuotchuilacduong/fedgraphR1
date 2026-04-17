"""Smoke tests for fedgraphr1 modules — run with: python -m fedgraphr1._smoke_test"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

errors = []

# 1. Types
try:
    from fedgraphr1.shared_types import (
        ExtractedEntityRecord, ClientExtractionResult,
        HypergraphFragment, FragmentStats, EntityMetadata,
        LoRAPayload, ServerMessage, ClientMessage,
    )
    r = ClientExtractionResult(client_id="0", round_number=0)
    assert r.num_entities == 0
    print("OK fedgraphr1.types")
except Exception as e:
    errors.append("FAIL types: " + str(e))

# 2. Compression roundtrip
try:
    from fedgraphr1.utils.compression import compress_json, decompress_json
    obj = {"hello": "world", "n": 42, "list": [1, 2, 3]}
    assert decompress_json(compress_json(obj)) == obj
    print("OK compression roundtrip")
except Exception as e:
    errors.append("FAIL compression: " + str(e))

# 3. Data partitioner — IID
try:
    from fedgraphr1.data.partitioner import partition_dataset
    examples = [{"question": f"q{i}", "golden_answers": ["a"]} for i in range(30)]
    splits = partition_dataset(examples, num_clients=3, strategy="iid", seed=0)
    assert len(splits) == 3
    assert sum(len(v) for v in splits.values()) == 30
    print("OK data.partitioner iid")
except Exception as e:
    errors.append("FAIL partitioner iid: " + str(e))

# 4. Data partitioner — Dirichlet
try:
    from fedgraphr1.data.partitioner import partition_dataset
    examples = [{"question": f"q{i}", "golden_answers": ["a"]} for i in range(30)]
    splits = partition_dataset(examples, num_clients=3, strategy="dirichlet",
                               seed=0, dirichlet_alpha=0.5)
    assert sum(len(v) for v in splits.values()) == 30
    print("OK data.partitioner dirichlet")
except Exception as e:
    errors.append("FAIL partitioner dirichlet: " + str(e))

# 5. Data partitioner — topic_skew
try:
    from fedgraphr1.data.partitioner import partition_dataset
    examples = [{"question": f"q{i}", "topic": f"t{i%5}"} for i in range(30)]
    splits = partition_dataset(examples, num_clients=3, strategy="topic_skew",
                               seed=0, topic_key="topic")
    assert sum(len(v) for v in splits.values()) == 30
    print("OK data.partitioner topic_skew")
except Exception as e:
    errors.append("FAIL partitioner topic_skew: " + str(e))

# 6. EntityPacker roundtrip
try:
    from fedgraphr1.shared_types import (
        ClientExtractionResult, ExtractedEntityRecord, ExtractedHyperedgeRecord
    )
    from fedgraphr1.client.entity_packer import EntityPacker
    ents = [ExtractedEntityRecord(entity_name=f"E{i}", entity_type="ORG",
                                   description=f"desc{i}", weight=50.0+i,
                                   source_chunk_hash=f"hash{i}")
            for i in range(5)]
    hes  = [ExtractedHyperedgeRecord(hyperedge_name=f"<hyperedge>fact{i}",
                                      weight=1.0, source_chunk_hash=f"hash{i}")
            for i in range(3)]
    res = ClientExtractionResult(client_id="0", round_number=1,
                                  entities=ents, hyperedges=hes)
    packer = EntityPacker(max_description_tokens=200, weight_threshold=30.0, use_delta=False)
    payload = packer.pack(res)
    # pack() returns compressed bytes; unpack to verify roundtrip
    unpacked = packer.unpack(payload)
    assert unpacked.num_entities == 5
    assert unpacked.num_hyperedges == 3
    print("OK EntityPacker roundtrip (full)")
except Exception as e:
    errors.append("FAIL EntityPacker: " + str(e))

# 7. EntityPacker delta
try:
    from fedgraphr1.shared_types import (
        ClientExtractionResult, ExtractedEntityRecord
    )
    from fedgraphr1.client.entity_packer import EntityPacker
    ents_r1 = [ExtractedEntityRecord(entity_name=f"E{i}", entity_type="ORG",
                                      description="d", weight=50.0, source_chunk_hash="c")
               for i in range(3)]
    ents_r2 = [ExtractedEntityRecord(entity_name=f"E{i}", entity_type="ORG",
                                      description="d", weight=50.0, source_chunk_hash="c")
               for i in range(5)]
    packer = EntityPacker(use_delta=True)
    r1 = ClientExtractionResult(client_id="0", round_number=1, entities=ents_r1)
    r2 = ClientExtractionResult(client_id="0", round_number=2, entities=ents_r2)
    packer.pack(r1)
    p2_bytes = packer.pack(r2)
    p2 = packer.unpack(p2_bytes)
    assert p2.num_entities == 2, f"Expected 2 delta entities, got {p2.num_entities}"
    print("OK EntityPacker delta (round 2 sends only new)")
except Exception as e:
    errors.append("FAIL EntityPacker delta: " + str(e))

# 8. EntityDeduplicator
try:
    from fedgraphr1.shared_types import ClientExtractionResult, ExtractedEntityRecord
    from fedgraphr1.server.entity_aggregator import EntityDeduplicator
    ents = [ExtractedEntityRecord(entity_name="Alice", entity_type="PERSON",
                                   description="d", weight=60.0, source_chunk_hash="c")]
    r1 = ClientExtractionResult(client_id="0", round_number=1, entities=ents)
    r2 = ClientExtractionResult(client_id="1", round_number=1, entities=ents)
    dedup = EntityDeduplicator()
    merged = dedup.deduplicate([r1, r2])
    assert len(merged) == 1
    print("OK EntityDeduplicator exact-name dedup")
except Exception as e:
    errors.append("FAIL EntityDeduplicator: " + str(e))

# 9. FederatedLoRAServer — FedAvg
try:
    import torch
    from fedgraphr1.server.lora_aggregator import FederatedLoRAServer
    srv = FederatedLoRAServer()
    updates = [
        ("c0", {"lora_A": torch.ones(4, 4)}, 10),
        ("c1", {"lora_A": torch.zeros(4, 4)}, 10),
    ]
    agg = srv.aggregate(updates)
    expected = torch.full((4, 4), 0.5)
    assert torch.allclose(agg["lora_A"], expected), f"Expected 0.5, got {agg['lora_A']}"
    print("OK FederatedLoRAServer FedAvg (50-50 clients)")
except Exception as e:
    errors.append("FAIL FederatedLoRAServer: " + str(e))

# 10. FederatedLoRAServer — weighted FedAvg
try:
    import torch
    from fedgraphr1.server.lora_aggregator import FederatedLoRAServer
    srv2 = FederatedLoRAServer()
    updates2 = [
        ("c0", {"lora_B": torch.ones(2, 2)}, 30),
        ("c1", {"lora_B": torch.zeros(2, 2)}, 10),
    ]
    agg2 = srv2.aggregate(updates2)
    expected2 = torch.full((2, 2), 0.75)
    assert torch.allclose(agg2["lora_B"], expected2), f"Expected 0.75, got {agg2['lora_B']}"
    print("OK FederatedLoRAServer weighted FedAvg")
except Exception as e:
    errors.append("FAIL FederatedLoRAServer weighted: " + str(e))

# 11. HypergraphPartitioner — full_broadcast
try:
    import networkx as nx
    from fedgraphr1.server.hypergraph_partitioner import HypergraphPartitioner
    G = nx.Graph()
    G.add_node("Alice", role="entity"); G.add_node("Bob", role="entity")
    G.add_node("<HE>fact1", role="hyperedge")
    G.add_edge("Alice", "<HE>fact1"); G.add_edge("Bob", "<HE>fact1")
    G.add_node("Charlie", role="entity")
    part = HypergraphPartitioner(global_graph=G, strategy="full_broadcast")
    frag = part.partition_for_client("c0", client_entity_names=set(), round_number=0)
    # frag is a HypergraphFragment; graph_data holds the nx.Graph
    assert frag.graph_data.number_of_nodes() == G.number_of_nodes()
    print("OK HypergraphPartitioner full_broadcast")
except Exception as e:
    errors.append("FAIL HypergraphPartitioner full_broadcast: " + str(e))

# 12. HypergraphPartitioner — relevance_based
try:
    import networkx as nx
    from fedgraphr1.server.hypergraph_partitioner import HypergraphPartitioner
    # Build a graph where Alice's cluster is clearly separate from Dave's cluster
    G2 = nx.Graph()
    for n, r in [("Alice","entity"),("Bob","entity"),("<HE>f1","hyperedge"),
                 ("Dave","entity"),("Eve","entity"),("<HE>f2","hyperedge"),
                 ("Unrelated1","entity"),("Unrelated2","entity"),
                 ("Unrelated3","entity"),("Unrelated4","entity")]:
        G2.add_node(n, role=r)
    G2.add_edge("Alice","<HE>f1"); G2.add_edge("Bob","<HE>f1")
    G2.add_edge("Dave","<HE>f2"); G2.add_edge("Eve","<HE>f2")
    # top_k_global=1 so only 1 hub node is added globally → subset guaranteed
    part2 = HypergraphPartitioner(global_graph=G2, strategy="relevance_based",
                                   top_k_global=1)
    frag2 = part2.partition_for_client("c0", client_entity_names={"Alice"}, round_number=1)
    assert "Alice" in frag2.graph_data.nodes()
    assert frag2.graph_data.number_of_nodes() < G2.number_of_nodes()
    print("OK HypergraphPartitioner relevance_based")
except Exception as e:
    errors.append("FAIL HypergraphPartitioner relevance_based: " + str(e))

# 13. FragmentDistributor bytes roundtrip
try:
    import networkx as nx
    from fedgraphr1.server.fragment_distributor import HypergraphFragmentDistributor
    G = nx.Graph()
    G.add_node("Alice", role="entity"); G.add_node("<HE>fact", role="hyperedge")
    G.add_edge("Alice", "<HE>fact")
    dist = HypergraphFragmentDistributor()
    frag = dist.distribute_in_memory(G)
    assert frag.number_of_nodes() == G.number_of_nodes()
    print("OK FragmentDistributor bytes roundtrip")
except Exception as e:
    errors.append("FAIL FragmentDistributor: " + str(e))

# 14. compute_kg_stats
try:
    import networkx as nx
    from fedgraphr1.utils.metrics import compute_kg_stats
    G = nx.Graph()
    G.add_node("Alice", role="entity"); G.add_node("Bob", role="entity")
    G.add_node("<HE>f", role="hyperedge")
    G.add_edge("Alice", "<HE>f"); G.add_edge("Bob", "<HE>f")
    stats = compute_kg_stats(G)
    assert stats["num_entity_nodes"] == 2
    assert stats["num_hyperedge_nodes"] == 1
    print("OK compute_kg_stats")
except Exception as e:
    errors.append("FAIL compute_kg_stats: " + str(e))

# 15. fl.base abstract interface
try:
    from fedgraphr1.fl.base import BaseClient, BaseServer
    import types as _types
    assert hasattr(BaseClient, "execute")
    assert hasattr(BaseServer, "execute")
    print("OK fl.base abstract interface")
except Exception as e:
    errors.append("FAIL fl.base: " + str(e))

# 16. fl.config get_args defaults
try:
    import sys as _sys
    _argv_backup = _sys.argv
    _sys.argv = ["test"]
    from fedgraphr1.fl.config import get_args
    args = get_args()
    assert args.num_clients == 3
    assert args.num_rounds == 40
    _sys.argv = _argv_backup
    print("OK fl.config get_args defaults")
except Exception as e:
    errors.append("FAIL fl.config: " + str(e))

# 17. FederatedSearchTool Tool interface
try:
    import tempfile
    from fedgraphr1.client.federated_search_tool import FederatedSearchTool
    with tempfile.TemporaryDirectory() as tmpdir:
        tool = FederatedSearchTool(working_dir=tmpdir, embedding_model=None)
        assert tool.name == "search"
        valid, _ = tool.validate_args({"query": "test"})
        assert valid
        invalid, _ = tool.validate_args({})
        assert not invalid
        results = tool.batch_execute([{"query": "test"}])
        assert len(results) == 1
        assert "<knowledge>" in results[0]
    print("OK FederatedSearchTool Tool interface")
except Exception as e:
    errors.append("FAIL FederatedSearchTool: " + str(e))

# 18. GraphR1Client instantiation
try:
    import tempfile, sys as _sys
    _argv_backup = _sys.argv
    _sys.argv = ["test"]
    from fedgraphr1.fl.config import get_args
    args = get_args()
    _sys.argv = _argv_backup
    args.base_model = None
    with tempfile.TemporaryDirectory() as tmpdir:
        args.working_dir = tmpdir
        from fedgraphr1.fl.client import GraphR1Client
        pool = {}
        client = GraphR1Client(
            client_id=0, args=args, local_data=[],
            message_pool=pool, device="cpu",
            graphr1_instance=None, embedding_model=None,
        )
        assert client.client_id == 0
    print("OK GraphR1Client instantiation")
except Exception as e:
    errors.append("FAIL GraphR1Client: " + str(e))

# 19. FL round loop (2 clients, 2 rounds, empty data)
try:
    import tempfile, sys as _sys
    _argv_backup = _sys.argv
    _sys.argv = ["test"]
    from fedgraphr1.fl.config import get_args
    args = get_args()
    _sys.argv = _argv_backup
    args.base_model = None
    args.num_rounds = 2
    args.num_clients = 2
    args.client_frac = 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        args.working_dir = tmpdir
        pool = {}
        from fedgraphr1.fl.client import GraphR1Client
        from fedgraphr1.fl.server import GraphR1Server
        from fedgraphr1.fl.trainer import GraphR1Trainer

        clients = [
            GraphR1Client(
                client_id=i, args=args, local_data=[],
                message_pool=pool, device="cpu",
                graphr1_instance=None, embedding_model=None,
            )
            for i in range(2)
        ]
        server = GraphR1Server(args=args, message_pool=pool, device="cpu",
                               graphr1_instance=None)
        trainer = GraphR1Trainer(args=args, clients=clients, server=server)
        history = trainer.train()
        assert len(history) == 2
        assert "round" in history[0]
    print("OK FL round loop (2 clients, 2 rounds)")
except Exception as e:
    errors.append("FAIL FL round loop: " + str(e))

# 20. LoRA sparsifier (P4.6)
try:
    import torch
    from fedgraphr1.server.lora_sparsifier import TopKSparsifier
    from fedgraphr1.server.lora_aggregator import FederatedLoRAServer

    sd_a = {"lora_A": torch.randn(10, 10), "lora_B": torch.randn(10, 10)}
    sd_b = {"lora_A": torch.randn(10, 10), "lora_B": torch.randn(10, 10)}

    sparse_a = TopKSparsifier.sparsify(sd_a, top_k_ratio=0.1)
    d = TopKSparsifier.density(sparse_a)
    assert 0.05 <= d <= 0.20, f"density={d} outside expected range for top_k=0.1"

    dense_bytes  = TopKSparsifier.estimate_bytes(sd_a, sparse=False)
    sparse_bytes = TopKSparsifier.estimate_bytes(sparse_a, sparse=True)
    assert sparse_bytes < dense_bytes

    full = TopKSparsifier.sparsify(sd_a, top_k_ratio=1.0)
    assert TopKSparsifier.density(full) > 0.99

    sparse_b = TopKSparsifier.sparsify(sd_b, top_k_ratio=0.1)
    server_fedavg = FederatedLoRAServer(aggregation_strategy="fedavg")
    agg = server_fedavg.aggregate([("c0", sd_a, 50), ("c1", sparse_b, 50)])
    assert set(agg.keys()) == {"lora_A", "lora_B"}
    assert agg["lora_A"].shape == sd_a["lora_A"].shape

    server_masked = FederatedLoRAServer(aggregation_strategy="masked_fedavg")
    agg_m = server_masked.aggregate([("c0", sd_a, 50), ("c1", sparse_b, 50)])
    assert set(agg_m.keys()) == {"lora_A", "lora_B"}

    print("OK LoRA top-k sparsifier (P4.6)")
except Exception as e:
    errors.append("FAIL LoRA sparsifier: " + str(e))

# Summary
print()
if errors:
    for e in errors:
        print(e)
    print(f"\n{len(errors)} test(s) FAILED.")
    sys.exit(1)
else:
    print(f"All {20} smoke tests passed.")
