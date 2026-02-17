"""Unit tests for retrieval components."""

import pytest

from src.query_parser import parse_query
from src.retrieval import SparseRetriever, reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Query Parser Tests
# ---------------------------------------------------------------------------

class TestQueryParser:
    def test_under_with_dollar(self):
        result = parse_query("running shoes under $100")
        assert result.max_price == 100.0
        assert result.min_price is None
        assert "under" not in result.search_text.lower()
        assert "100" not in result.search_text

    def test_under_without_dollar(self):
        result = parse_query("laptop below 1000")
        assert result.max_price == 1000.0

    def test_above_price(self):
        result = parse_query("premium headphones above $200")
        assert result.min_price == 200.0
        assert result.max_price is None

    def test_price_range(self):
        result = parse_query("tablets between $300 and $600")
        assert result.min_price == 300.0
        assert result.max_price == 600.0

    def test_no_price(self):
        result = parse_query("comfortable wireless headphones")
        assert result.min_price is None
        assert result.max_price is None
        assert result.search_text == "comfortable wireless headphones"

    def test_model_numbers_not_parsed_as_price(self):
        """iPhone 12, Size 10, 4K — these should NOT become prices."""
        result = parse_query("iPhone 12 Pro Max")
        assert result.max_price is None
        assert result.min_price is None

    def test_4k_monitor_under_500(self):
        """'4K monitor under $500' should extract max_price=500, not 4."""
        result = parse_query("4K monitor under $500")
        assert result.max_price == 500.0
        assert "4K" in result.search_text or "4k" in result.search_text.lower()

    def test_category_hint_electronics(self):
        result = parse_query("wireless headphones noise canceling")
        assert result.category_hint == "Electronics"

    def test_category_hint_phones(self):
        result = parse_query("iphone case with screen protector")
        assert result.category_hint == "Cell Phones & Accessories"

    def test_category_hint_toys(self):
        result = parse_query("lego set for kids")
        assert result.category_hint == "Toys & Games"

    def test_empty_search_text_fallback(self):
        """If price stripping leaves empty text, fall back to original."""
        result = parse_query("under $50")
        assert len(result.search_text) > 0

    def test_commas_in_price(self):
        result = parse_query("laptop under $1,500")
        assert result.max_price == 1500.0


# ---------------------------------------------------------------------------
# Sparse Retriever Tests
# ---------------------------------------------------------------------------

class TestSparseRetriever:
    def test_tokenize_basic(self):
        retriever = SparseRetriever()
        tokens = retriever.tokenize("Wireless Bluetooth Headphones")
        assert "wireless" in tokens
        assert "bluetooth" in tokens
        assert "headphones" in tokens

    def test_tokenize_preserves_hyphenated_terms(self):
        retriever = SparseRetriever()
        tokens = retriever.tokenize("Sony WH-1000XM5 USB-C headphones")
        # Should have hyphenated terms as single tokens
        assert any("wh-1000xm5" in t for t in tokens)
        assert any("usb-c" in t for t in tokens)

    def test_tokenize_removes_short_tokens(self):
        retriever = SparseRetriever()
        tokens = retriever.tokenize("a b c hello world")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "hello" in tokens

    def test_search_returns_results(self):
        retriever = SparseRetriever()
        ids = ["p1", "p2", "p3"]
        texts = [
            "Sony wireless noise canceling headphones",
            "Apple MacBook Pro laptop computer",
            "Nike running shoes for women",
        ]
        retriever.build_index(ids, texts)
        results = retriever.search("headphones", top_k=3)
        assert len(results) > 0
        # First result should be the headphones product
        assert results[0][0] == "p1"

    def test_search_empty_query(self):
        retriever = SparseRetriever()
        ids = ["p1"]
        texts = ["Some product"]
        retriever.build_index(ids, texts)
        results = retriever.search("", top_k=3)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# RRF Fusion Tests
# ---------------------------------------------------------------------------

class TestRRF:
    def test_basic_fusion(self):
        dense = [("p1", 0.9), ("p2", 0.8), ("p3", 0.7)]
        sparse = [("p2", 5.0), ("p1", 3.0), ("p4", 2.0)]
        fused = reciprocal_rank_fusion([dense, sparse], k=60, top_k=4)

        ids = [pid for pid, _ in fused]
        # p1 and p2 appear in both lists, should be top ranked
        assert "p1" in ids
        assert "p2" in ids

    def test_overlap_boosted(self):
        """Products appearing in both lists should score higher."""
        dense = [("p1", 0.9), ("p2", 0.8)]
        sparse = [("p1", 5.0), ("p3", 3.0)]
        fused = reciprocal_rank_fusion([dense, sparse], k=60, top_k=3)

        # p1 appears in both → highest RRF score
        assert fused[0][0] == "p1"

    def test_single_list(self):
        ranked = [("p1", 0.9), ("p2", 0.8)]
        fused = reciprocal_rank_fusion([ranked], k=60, top_k=2)
        assert len(fused) == 2
        assert fused[0][0] == "p1"

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([[], []], k=60, top_k=5)
        assert fused == []

    def test_top_k_limits_output(self):
        dense = [("p1", 0.9), ("p2", 0.8), ("p3", 0.7)]
        sparse = [("p4", 5.0), ("p5", 3.0), ("p6", 2.0)]
        fused = reciprocal_rank_fusion([dense, sparse], k=60, top_k=2)
        assert len(fused) == 2
