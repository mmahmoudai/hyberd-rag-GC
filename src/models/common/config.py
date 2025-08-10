"""
Configuration settings for RAG systems in network security packet analysis.
"""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = PROJECT_ROOT.parent / "data"
MODELS_DIR = PROJECT_ROOT.parent / "models"
RESULTS_DIR = PROJECT_ROOT.parent / "experiments" / "results"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Vector database settings
VECTOR_DB_CONFIG = {
    "index_type": "HNSW",  # Options: HNSW, IVF, Flat
    "dimension": 1536,     # Default embedding dimension
    "metric": "cosine",    # Options: cosine, l2, ip (inner product)
    "nprobe": 10,          # Number of clusters to visit for IVF
    "ef_search": 128,      # Exploration factor for HNSW search
    "ef_construction": 200 # Exploration factor for HNSW construction
}

# Graph database settings
GRAPH_DB_CONFIG = {
    "node_types": [
        "ip_address", "port", "protocol", "service", 
        "session", "autonomous_system", "location", "device", "user"
    ],
    "edge_types": [
        "connects_to", "uses", "belongs_to", "communicates_with", 
        "contains", "follows", "authenticates"
    ],
    "default_weight": 1.0,
    "temporal_window": 3600  # Default time window in seconds
}

# Cache settings
CACHE_CONFIG = {
    "max_size": 10000,     # Maximum number of entries
    "ttl": 3600,           # Time-to-live in seconds
    "policy": "LRU",       # Options: LRU, LFU, TLFU
    "levels": {
        "L1": {"size": 1000, "ttl": 300},    # Fast, short-lived cache
        "L2": {"size": 5000, "ttl": 3600},   # Medium cache
        "L3": {"size": 10000, "ttl": 86400}  # Larger, long-lived cache
    }
}

# Hybrid system settings
HYBRID_CONFIG = {
    "cache_weight": 0.5,   # Weight for cache component
    "graph_weight": 0.5,   # Weight for graph component
    "adaptive": True,      # Whether to use adaptive weighting
    "synergy_threshold": 0.7  # Threshold for synergy factor
}

# Embedding model settings
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",  # Default lightweight model
    "batch_size": 32,
    "max_seq_length": 256,
    "normalize_embeddings": True
}

# Evaluation settings
EVAL_CONFIG = {
    "metrics": ["precision", "recall", "f1", "latency", "memory_usage"],
    "k_values": [1, 5, 10, 20, 50],
    "test_scenarios": ["zero_day", "high_throughput", "adversarial", "long_term"],
    "significance_level": 0.05
}

# Dataset settings
DATASET_CONFIG = {
    "cic_ids2017": {
        "path": DATA_DIR / "CIC-IDS2017",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15
    },
    "unsw_nb15": {
        "path": DATA_DIR / "UNSW-NB15",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15
    },
    "custom_pcap": {
        "path": DATA_DIR / "custom_pcap",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15
    }
}
