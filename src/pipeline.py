"""Main orchestration pipeline: query → parse → retrieve → filter → respond."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

from src.config import Config, load_config
from src.data_loader import Product, load_products, build_searchable_text
from src.llm import (
    LLMClient,
    build_search_prompt,
    build_grounded_response,
    parse_llm_response,
)
from src.query_parser import ParsedQuery, parse_query
from src.retrieval import DenseRetriever, SparseRetriever, hybrid_search

console = Console()


@dataclass
class SearchResult:
    query: str
    parsed_query: ParsedQuery
    products: list[Product]
    explanations: list[str]
    retrieval_scores: list[float]
    formatted_output: str


class HybridRAGPipeline:
    """End-to-end hybrid RAG pipeline for product search."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self._product_lookup: dict[str, Product] = {}
        self._dense: Optional[DenseRetriever] = None
        self._sparse: Optional[SparseRetriever] = None
        self._llm: Optional[LLMClient] = None

    def _ensure_retrievers(self) -> None:
        """Lazily initialize retrievers and load indices."""
        if self._dense is not None:
            return

        # Load product lookup
        lookup_path = Path(self.config.product_lookup_path)
        if not lookup_path.exists():
            raise FileNotFoundError(
                f"Product lookup not found: {lookup_path}\n"
                "Run 'python scripts/setup_data.py' first."
            )
        with open(lookup_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._product_lookup = {pid: Product(**data) for pid, data in raw.items()}

        # Dense retriever
        self._dense = DenseRetriever(
            model_name=self.config.embedding_model,
            persist_dir=self.config.dense_index_dir,
        )
        self._dense.load_collection()

        # Sparse retriever
        self._sparse = SparseRetriever()
        bm25_path = Path(self.config.bm25_index_path)
        if not bm25_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {bm25_path}\n"
                "Run 'python scripts/setup_data.py' first."
            )
        self._sparse.load_index(str(bm25_path))

    def _ensure_llm(self) -> LLMClient:
        """Lazily initialize LLM client."""
        if self._llm is None:
            self._llm = LLMClient(
                base_url=self.config.llm_base_url,
                model=self.config.llm_model,
            )
        return self._llm

    def index(self, products_path: str) -> None:
        """Build all indices from a product JSON file."""
        products = load_products(products_path)
        console.print(f"Loaded {len(products)} products from {products_path}")

        # Build searchable texts
        searchable_texts = [build_searchable_text(p) for p in products]
        product_ids = [p.id for p in products]

        # Metadata (stored alongside embeddings for reference)
        metadatas = [
            {"price": p.price, "category": p.category}
            for p in products
        ]

        # Dense index
        self._dense = DenseRetriever(
            model_name=self.config.embedding_model,
            persist_dir=self.config.dense_index_dir,
        )
        self._dense.index_products(product_ids, searchable_texts, metadatas)

        # Sparse index
        self._sparse = SparseRetriever()
        self._sparse.build_index(product_ids, searchable_texts)
        self._sparse.save_index(self.config.bm25_index_path)

        # Product lookup
        lookup = {p.id: {
            "id": p.id,
            "title": p.title,
            "description": p.description,
            "features": p.features,
            "category": p.category,
            "price": p.price,
            "url": p.url,
            "url_verified": p.url_verified,
        } for p in products}
        lookup_path = Path(self.config.product_lookup_path)
        lookup_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lookup_path, "w", encoding="utf-8") as f:
            json.dump(lookup, f, indent=2, ensure_ascii=False)

        console.print(f"[green]All indices built and saved.[/green]")

    def _apply_filters(
        self,
        candidates: list[tuple[str, float]],
        parsed: ParsedQuery,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Apply post-RRF price and category filters."""
        filtered = []
        for pid, score in candidates:
            product = self._product_lookup.get(pid)
            if product is None:
                continue

            # Price filters
            if parsed.max_price is not None and product.price > parsed.max_price:
                continue
            if parsed.min_price is not None and product.price < parsed.min_price:
                continue

            filtered.append((pid, score))
            if len(filtered) >= top_k:
                break

        return filtered

    def search(self, query: str, top_k: Optional[int] = None) -> SearchResult:
        """Full pipeline: parse → retrieve → fuse → filter → LLM → grounded output."""
        self._ensure_retrievers()
        top_k = top_k or self.config.final_top_k

        # 1. Parse query
        parsed = parse_query(query)

        # 2. Hybrid retrieval (dense + sparse + RRF)
        fused = hybrid_search(
            query_text=parsed.search_text,
            dense_retriever=self._dense,
            sparse_retriever=self._sparse,
            dense_top_k=self.config.dense_top_k,
            sparse_top_k=self.config.sparse_top_k,
            fusion_top_k=self.config.fusion_top_k,
            rrf_k=self.config.rrf_k,
        )

        # 3. Post-RRF filtering (price, category)
        filtered = self._apply_filters(fused, parsed, top_k)

        if not filtered:
            return SearchResult(
                query=query,
                parsed_query=parsed,
                products=[],
                explanations=[],
                retrieval_scores=[],
                formatted_output=f"No products found matching \"{query}\". Try broadening your search.",
            )

        # 4. Get product objects
        products = []
        scores = []
        for pid, score in filtered:
            product = self._product_lookup.get(pid)
            if product:
                products.append(product)
                scores.append(score)

        # 5. LLM explanation
        llm = self._ensure_llm()
        explanations = [f"Matches your search for \"{query}\"." for _ in products]
        summary = "Here are the top products matching your search."

        if llm.is_available():
            prompt = build_search_prompt(query, products)
            try:
                raw_response = llm.generate(
                    prompt,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                )
                explanations, summary = parse_llm_response(raw_response, len(products))
            except Exception as e:
                console.print(f"[yellow]LLM generation failed: {e}. Using default explanations.[/yellow]")
        else:
            console.print("[yellow]LM Studio not available. Using default explanations.[/yellow]")

        # 6. Build grounded response
        formatted = build_grounded_response(query, products, explanations, summary, scores)

        return SearchResult(
            query=query,
            parsed_query=parsed,
            products=products,
            explanations=explanations,
            retrieval_scores=scores,
            formatted_output=formatted,
        )

    def search_retrieval_only(self, query: str, top_k: Optional[int] = None) -> list[Product]:
        """Retrieval only — no LLM. Useful for debugging and evaluation."""
        self._ensure_retrievers()
        top_k = top_k or self.config.final_top_k

        parsed = parse_query(query)
        fused = hybrid_search(
            query_text=parsed.search_text,
            dense_retriever=self._dense,
            sparse_retriever=self._sparse,
            dense_top_k=self.config.dense_top_k,
            sparse_top_k=self.config.sparse_top_k,
            fusion_top_k=self.config.fusion_top_k,
            rrf_k=self.config.rrf_k,
        )

        filtered = self._apply_filters(fused, parsed, top_k)

        products = []
        for pid, _score in filtered:
            product = self._product_lookup.get(pid)
            if product:
                products.append(product)

        return products
