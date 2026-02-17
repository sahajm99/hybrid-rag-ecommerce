"""Parse natural language search queries to extract price constraints and category hints."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedQuery:
    search_text: str
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    category_hint: Optional[str] = None


# Category keyword mapping
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Electronics": [
        "laptop", "laptops", "headphone", "headphones", "speaker", "speakers",
        "monitor", "monitors", "keyboard", "mouse", "earbuds", "charger",
        "tv", "television", "computer", "smartwatch", "usb", "bluetooth",
        "tablet", "tablets", "camera", "cameras", "router", "adapter",
    ],
    "Cell Phones & Accessories": [
        "phone", "phones", "smartphone", "iphone", "samsung", "android",
        "case", "screen protector", "cable", "wireless charger", "sim",
        "mobile", "cell", "tempered glass", "car mount", "power bank",
    ],
    "Toys & Games": [
        "toy", "toys", "game", "games", "puzzle", "puzzles", "lego",
        "doll", "dolls", "action figure", "board game", "card game",
        "kids", "children", "play", "stuffed", "nerf", "playset",
    ],
    "Arts Crafts & Sewing": [
        "craft", "crafts", "sewing", "knitting", "yarn", "fabric",
        "paint", "painting", "brush", "canvas", "sticker", "stickers",
        "beads", "ribbon", "scrapbook", "embroidery", "thread", "needle",
    ],
}


def _parse_number(s: str) -> Optional[float]:
    """Parse a number string, handling commas."""
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_query(raw_query: str) -> ParsedQuery:
    """Parse a search query to extract price constraints and category hints.

    Extracts:
        - max_price from patterns like "under $100", "below $50", "less than 200"
        - min_price from patterns like "above $50", "over $100", "more than 30"
        - price range from "between $50 and $200", "$50 to $200"
        - category_hint from keyword matching

    The price language is stripped from search_text to avoid polluting embeddings.
    """
    text = raw_query.strip()
    min_price = None
    max_price = None

    # Pattern: "between $X and $Y" or "$X to $Y"
    range_pattern = r"(?:between\s+)?\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:to|-|and)\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)"
    range_match = re.search(range_pattern, text, re.IGNORECASE)
    if range_match:
        low = _parse_number(range_match.group(1))
        high = _parse_number(range_match.group(2))
        if low is not None and high is not None and low < high:
            min_price = low
            max_price = high
            text = text[:range_match.start()] + text[range_match.end():]

    # Pattern: "under/below/less than/up to $X" or "under/below/less than/up to X dollars"
    if max_price is None:
        max_pattern = r"(?:under|below|less\s+than|up\s+to|cheaper\s+than)\s+\$?(\d+(?:,\d{3})*(?:\.\d{2})?)(?:\s*dollars?)?"
        max_match = re.search(max_pattern, text, re.IGNORECASE)
        if max_match:
            max_price = _parse_number(max_match.group(1))
            text = text[:max_match.start()] + text[max_match.end():]

    # Pattern: "above/over/more than/at least $X"
    if min_price is None:
        min_pattern = r"(?:above|over|more\s+than|at\s+least|starting\s+at)\s+\$?(\d+(?:,\d{3})*(?:\.\d{2})?)(?:\s*dollars?)?"
        min_match = re.search(min_pattern, text, re.IGNORECASE)
        if min_match:
            min_price = _parse_number(min_match.group(1))
            text = text[:min_match.start()] + text[min_match.end():]

    # Clean up leftover whitespace
    search_text = re.sub(r"\s+", " ", text).strip()
    if not search_text:
        search_text = raw_query.strip()

    # Category hint extraction
    category_hint = None
    lower_query = raw_query.lower()
    best_match_count = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        match_count = sum(1 for kw in keywords if kw in lower_query)
        if match_count > best_match_count:
            best_match_count = match_count
            category_hint = category

    return ParsedQuery(
        search_text=search_text,
        min_price=min_price,
        max_price=max_price,
        category_hint=category_hint,
    )
