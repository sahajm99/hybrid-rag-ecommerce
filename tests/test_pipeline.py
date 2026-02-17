"""Integration tests for the pipeline and data loader."""

import pytest

from src.data_loader import clean_price, safe_join, build_searchable_text, Product
from src.llm import parse_llm_response


# ---------------------------------------------------------------------------
# Data Loader Tests
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_clean_price_basic(self):
        assert clean_price("$29.99") == 29.99

    def test_clean_price_with_commas(self):
        assert clean_price("$1,299.00") == 1299.00

    def test_clean_price_range(self):
        """Takes the lower bound of a range."""
        assert clean_price("$10.00 - $20.00") == 10.00

    def test_clean_price_none(self):
        assert clean_price(None) is None

    def test_clean_price_empty(self):
        assert clean_price("") is None

    def test_clean_price_invalid(self):
        assert clean_price("free") is None

    def test_clean_price_negative(self):
        assert clean_price("-5") is None

    def test_safe_join_list(self):
        assert safe_join(["hello", "world"]) == "hello world"

    def test_safe_join_none(self):
        assert safe_join(None) == ""

    def test_safe_join_string(self):
        assert safe_join("already a string") == "already a string"

    def test_safe_join_mixed(self):
        assert safe_join(["text", None, "more"]) == "text more"

    def test_build_searchable_text(self):
        product = Product(
            id="TEST123",
            title="Sony Headphones",
            description="Great noise canceling",
            features=["Wireless", "40hr battery"],
            category="Electronics",
            price=299.99,
            url="https://amazon.com/dp/TEST123",
        )
        text = build_searchable_text(product)
        assert "Sony Headphones" in text
        assert "noise canceling" in text
        assert "Wireless" in text
        assert "Electronics" in text


# ---------------------------------------------------------------------------
# LLM Response Parsing Tests
# ---------------------------------------------------------------------------

class TestLLMResponseParsing:
    def test_parse_valid_response(self):
        response = (
            "Product 1: Great match for running needs.\n"
            "Product 2: Affordable and lightweight.\n"
            "Summary: Both products suit your query."
        )
        explanations, summary = parse_llm_response(response, 2)
        assert explanations[0] == "Great match for running needs."
        assert explanations[1] == "Affordable and lightweight."
        assert summary == "Both products suit your query."

    def test_parse_partial_response(self):
        """If LLM only returns some products, defaults fill the gaps."""
        response = "Product 1: Good headphones.\nSummary: Solid choice."
        explanations, summary = parse_llm_response(response, 3)
        assert explanations[0] == "Good headphones."
        assert "Matches your search query." in explanations[1]
        assert "Matches your search query." in explanations[2]

    def test_parse_empty_response(self):
        explanations, summary = parse_llm_response("", 2)
        assert len(explanations) == 2
        assert all("Matches your search query." in e for e in explanations)

    def test_parse_malformed_response(self):
        """Gracefully handles nonsense from the LLM."""
        response = "Here are some random thoughts about products..."
        explanations, summary = parse_llm_response(response, 2)
        assert len(explanations) == 2  # Falls back to defaults
