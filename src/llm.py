"""LLM client for LM Studio and prompt/response building."""

from typing import Optional

from rich.console import Console

from src.data_loader import Product

console = Console()


# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------

SEARCH_PROMPT = """You are a product search assistant. The user searched for: "{query}"

Below are the top matching products retrieved from our catalog. For each product, write ONE concise sentence explaining why it matches the user's search intent. Reference ONLY the information provided below — do not invent or assume any features, specs, or details that are not explicitly listed.

RULES:
- ONLY mention features explicitly stated in the product data below.
- Do NOT infer or add features that are not listed.
- If unsure whether a product has a feature, do not claim it does.
- Keep each explanation to one sentence.

{products_context}

Respond in this exact format (one line per product, no extra text):
Product 1: [Your one-sentence explanation]
Product 2: [Your one-sentence explanation]
...
Summary: [One sentence summarizing the recommendations]"""


def format_products_for_prompt(products: list[Product]) -> str:
    """Format products into a string for the LLM prompt."""
    parts = []
    for i, p in enumerate(products, 1):
        desc = p.description[:300] if p.description else "N/A"
        feats = "; ".join(p.features[:4]) if p.features else "N/A"
        parts.append(
            f"Product {i}:\n"
            f"  Title: {p.title}\n"
            f"  Price: ${p.price:.2f}\n"
            f"  Category: {p.category}\n"
            f"  Description: {desc}\n"
            f"  Features: {feats}"
        )
    return "\n\n".join(parts)


def build_search_prompt(query: str, products: list[Product]) -> str:
    """Build the complete prompt for the LLM."""
    context = format_products_for_prompt(products)
    return SEARCH_PROMPT.format(query=query, products_context=context)


def build_grounded_response(
    query: str,
    products: list[Product],
    llm_explanations: list[str],
    summary: str,
    scores: list[float],
) -> str:
    """Assemble final output with programmatic product cards + LLM explanations.

    Product data (title, price, URL) comes from the catalog — never from the LLM.
    The LLM only provides the 'why it matches' explanation.
    """
    lines = [f"## Search Results for: \"{query}\"\n"]

    for i, (product, explanation, score) in enumerate(
        zip(products, llm_explanations, scores), 1
    ):
        url_note = product.url
        if "ASIN_NOT_AVAILABLE" in product.url:
            url_note = f"{product.url} (direct link unavailable)"

        lines.append(
            f"### {i}. {product.title} — ${product.price:.2f}\n"
            f"- **Category:** {product.category}\n"
            f"- **Why it matches:** {explanation}\n"
            f"- **Link:** {url_note}\n"
            f"- **Relevance score:** {score:.4f}\n"
        )

    lines.append(f"---\n**Summary:** {summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM Client (LM Studio via OpenAI SDK)
# ---------------------------------------------------------------------------

class LLMClient:
    """Client for a local LLM served via LM Studio's OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "local-model"):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key="not-needed")
        self._model = model

    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
        """Generate a response from the LLM."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def is_available(self) -> bool:
        """Check if LM Studio is running and reachable."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False


def parse_llm_response(response: str, num_products: int) -> tuple[list[str], str]:
    """Parse the LLM's structured response into per-product explanations and summary.

    Returns:
        (explanations, summary) where explanations[i] corresponds to product i.
    """
    explanations = ["Matches your search query." for _ in range(num_products)]
    summary = "Here are the top products matching your search."

    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match "Product N: explanation"
        if line.lower().startswith("product"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                try:
                    idx_str = parts[0].lower().replace("product", "").strip()
                    idx = int(idx_str) - 1
                    if 0 <= idx < num_products:
                        explanations[idx] = parts[1].strip()
                except (ValueError, IndexError):
                    pass

        # Match "Summary: ..."
        elif line.lower().startswith("summary"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                summary = parts[1].strip()

    return explanations, summary
