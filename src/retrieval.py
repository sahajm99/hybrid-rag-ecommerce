"""Dense, sparse, and hybrid retrieval with Reciprocal Rank Fusion."""

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Dense Retrieval (Sentence Transformers + NumPy vector store)
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Embedding-based retrieval using sentence-transformers and numpy cosine similarity.

    At ~5,000 products, brute-force dot-product search over normalized
    embeddings is instant (<5ms) and avoids external vector DB dependencies.
    """

    def __init__(self, model_name: str, persist_dir: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._embeddings: Optional[np.ndarray] = None
        self._product_ids: list[str] = []

    def index_products(
        self,
        product_ids: list[str],
        searchable_texts: list[str],
        metadatas: list[dict],
    ) -> None:
        """Encode products and save embeddings to disk."""
        console.print("[blue]Encoding product embeddings...[/blue]")
        embeddings = self.model.encode(
            searchable_texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=64,
        )

        self._embeddings = embeddings
        self._product_ids = product_ids

        # Save to disk
        np.save(Path(self.persist_dir) / "embeddings.npy", embeddings)
        with open(Path(self.persist_dir) / "product_ids.json", "w") as f:
            json.dump(product_ids, f)

        console.print(f"[green]Indexed {len(product_ids)} products (embeddings saved)[/green]")

    def load_collection(self) -> None:
        """Load embeddings and product IDs from disk."""
        emb_path = Path(self.persist_dir) / "embeddings.npy"
        ids_path = Path(self.persist_dir) / "product_ids.json"

        if not emb_path.exists() or not ids_path.exists():
            raise FileNotFoundError(
                f"Dense index not found in {self.persist_dir}.\n"
                "Run 'python scripts/setup_data.py' first."
            )

        self._embeddings = np.load(emb_path)
        with open(ids_path, "r") as f:
            self._product_ids = json.load(f)

    def search(self, query_text: str, top_k: int = 30) -> list[tuple[str, float]]:
        """Search by cosine similarity (dot product on normalized vectors).

        No metadata filtering here — filtering happens post-RRF to keep
        dense and sparse candidate pools symmetric.
        """
        if self._embeddings is None:
            raise RuntimeError("Dense index not loaded. Run setup_data.py first.")

        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
        )

        # Cosine similarity via dot product (vectors are L2-normalized)
        scores = np.dot(self._embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self._product_ids[i], float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Sparse Retrieval (BM25)
# ---------------------------------------------------------------------------

class SparseRetriever:
    """BM25-based keyword retrieval."""

    def __init__(self):
        self._product_ids: list[str] = []
        self._index: Optional[BM25Okapi] = None

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25. Preserves hyphenated terms and model numbers."""
        text = text.lower()
        # Match words, hyphenated terms (e.g. WH-1000XM5, USB-C), and
        # alphanumeric sequences — keeps model numbers intact
        tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
        return [t for t in tokens if len(t) > 1]

    def build_index(self, product_ids: list[str], searchable_texts: list[str]) -> None:
        """Build BM25 index from product texts."""
        self._product_ids = product_ids
        tokenized = [self.tokenize(text) for text in searchable_texts]
        self._index = BM25Okapi(tokenized)
        console.print(f"[green]Built BM25 index over {len(product_ids)} products[/green]")

    def save_index(self, path: str) -> None:
        """Save BM25 index and product IDs to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"index": self._index, "product_ids": self._product_ids}, f)

    def load_index(self, path: str) -> None:
        """Load BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._index = data["index"]
        self._product_ids = data["product_ids"]

    def search(self, query_text: str, top_k: int = 30) -> list[tuple[str, float]]:
        """Search BM25 index and return (product_id, score) pairs."""
        if self._index is None:
            raise RuntimeError("BM25 index not built or loaded. Run setup_data.py first.")

        tokens = self.tokenize(query_text)
        scores = self._index.get_scores(tokens)

        # Get top_k indices by score
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((self._product_ids[idx], score))

        return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum over all lists: 1 / (k + rank(d))
    where rank is 1-based.

    Args:
        ranked_lists: Each list contains (product_id, score) sorted by score desc.
        k: RRF constant (default 60).
        top_k: Number of results to return.

    Returns:
        Fused ranked list: [(product_id, rrf_score), ...] sorted by score desc.
    """
    rrf_scores: dict[str, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def hybrid_search(
    query_text: str,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
    dense_top_k: int = 30,
    sparse_top_k: int = 30,
    fusion_top_k: int = 10,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Run hybrid search: dense + sparse retrieval, then RRF fusion."""
    dense_results = dense_retriever.search(query_text, top_k=dense_top_k)
    sparse_results = sparse_retriever.search(query_text, top_k=sparse_top_k)

    fused = reciprocal_rank_fusion(
        ranked_lists=[dense_results, sparse_results],
        k=rrf_k,
        top_k=fusion_top_k,
    )

    return fused
