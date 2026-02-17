"""Configuration loader for the hybrid RAG pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # LLM (LM Studio)
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "local-model"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # Retrieval
    dense_top_k: int = 30
    sparse_top_k: int = 30
    fusion_top_k: int = 10
    final_top_k: int = 5
    rrf_k: int = 60

    # Data paths
    data_path: str = "data/products.json"
    dense_index_dir: str = "indices/dense"
    bm25_index_path: str = "indices/bm25_index.pkl"
    product_lookup_path: str = "indices/product_lookup.json"

    # Dataset download
    categories: list[str] = field(default_factory=lambda: [
        "Electronics",
        "Cell_Phones_and_Accessories",
        "Toys_and_Games",
        "Arts_Crafts_and_Sewing",
    ])
    products_per_category: int = 1250


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file, falling back to defaults."""
    path = Path(config_path)
    if not path.exists():
        return Config()

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return Config(**{k: v for k, v in raw.items() if k in Config.__dataclass_fields__})
