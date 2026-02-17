"""Data loading, preprocessing, and product schema for the hybrid RAG pipeline."""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@dataclass
class Product:
    id: str
    title: str
    description: str
    features: list[str]
    category: str
    price: float
    url: str
    url_verified: bool = True


def safe_join(field) -> str:
    """Safely join a field that may be a list, string, numpy array, or None."""
    if field is None:
        return ""
    # Parquet may return numpy arrays
    if hasattr(field, "tolist"):
        field = field.tolist()
    if isinstance(field, (list, tuple)):
        return " ".join(str(x) for x in field if x)
    return str(field)


def clean_price(raw_price) -> Optional[float]:
    """Parse price string into float. Handles '$29.99', '$10 - $20', etc."""
    if raw_price is None:
        return None
    price_str = str(raw_price).strip()
    if not price_str:
        return None

    # Remove dollar sign and commas
    price_str = price_str.replace("$", "").replace(",", "")

    # Handle range: take the lower bound
    if " - " in price_str:
        price_str = price_str.split(" - ")[0].strip()
    elif " to " in price_str.lower():
        price_str = price_str.lower().split(" to ")[0].strip()

    try:
        val = float(price_str)
        # Sanity check: reject unrealistic prices
        if val <= 0 or val > 50000:
            return None
        return round(val, 2)
    except (ValueError, TypeError):
        return None


def build_searchable_text(product: Product) -> str:
    """Combine product fields into a single searchable string."""
    parts = [product.title]
    if product.description:
        parts.append(product.description)
    if product.features:
        parts.append(". ".join(product.features))
    parts.append(product.category)
    return " ".join(parts)


def _download_category(cat: str) -> pd.DataFrame:
    """Download a category's data from HuggingFace Hub.

    Tries parquet first (fast), falls back to JSONL for categories
    that only have JSONL files available.
    """
    from huggingface_hub import hf_hub_download, list_repo_tree

    repo_id = "McAuley-Lab/Amazon-Reviews-2023"

    # Strategy 1: Try parquet files in raw_meta_{cat}/ folder
    prefix = f"raw_meta_{cat}"
    try:
        entries = list_repo_tree(repo_id, path_in_repo=prefix, repo_type="dataset")
        parquet_files = [
            e.path for e in entries
            if hasattr(e, "path") and e.path.endswith(".parquet")
        ]
        if parquet_files:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=parquet_files[0],
                repo_type="dataset",
            )
            return pd.read_parquet(local_path)
    except Exception:
        pass

    # Strategy 2: Try JSONL file in raw/meta_categories/
    jsonl_filename = f"raw/meta_categories/meta_{cat}.jsonl"
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=jsonl_filename,
            repo_type="dataset",
        )
        console.print(f"[dim]  Loading JSONL for {cat} (this may take a moment)...[/dim]")
        return pd.read_json(local_path, lines=True)
    except Exception:
        pass

    raise RuntimeError(
        f"No data found for category: {cat}. "
        f"Tried parquet ({prefix}/) and JSONL ({jsonl_filename})."
    )


def download_and_prepare(
    categories: list[str],
    per_category: int,
    output_path: str,
) -> list[Product]:
    """Download Amazon product metadata from HuggingFace and prepare it."""
    all_products = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for cat in categories:
            task = progress.add_task(f"Downloading {cat}...", total=None)

            try:
                df = _download_category(cat)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to download {cat}: {e}[/yellow]")
                progress.update(task, description=f"[yellow]Skipped {cat}[/yellow]")
                continue

            progress.update(task, description=f"Processing {cat} ({len(df)} raw products)...")

            category_products = []
            for _, row in df.iterrows():
                title = str(row.get("title") or "").strip()
                if not title:
                    continue

                price = clean_price(row.get("price"))
                if price is None:
                    continue

                description = safe_join(row.get("description"))
                features_raw = row.get("features")
                features = []
                # Parquet may return numpy arrays — convert to list first
                if hasattr(features_raw, "tolist"):
                    features_raw = features_raw.tolist()
                if isinstance(features_raw, (list, tuple)):
                    features = [str(f).strip() for f in features_raw if f and str(f).strip()]
                elif features_raw is not None and str(features_raw).strip():
                    features = [str(features_raw).strip()]

                # Skip if both description and features are empty
                if not description.strip() and not features:
                    continue

                asin = str(row.get("parent_asin", "")).strip()
                if not asin:
                    continue

                product = Product(
                    id=asin,
                    title=title,
                    description=description,
                    features=features,
                    category=cat.replace("_", " ").replace(" and ", " & "),
                    price=price,
                    url=f"https://www.amazon.com/dp/{asin}",
                    url_verified=True,
                )
                category_products.append(product)

            # Sample if we have more than needed
            if len(category_products) > per_category:
                random.seed(42)
                category_products = random.sample(category_products, per_category)

            all_products.extend(category_products)
            progress.update(
                task,
                description=f"[green]{cat}: {len(category_products)} products[/green]",
            )

    # Deduplicate by product ID
    seen = set()
    unique_products = []
    for p in all_products:
        if p.id not in seen:
            seen.add(p.id)
            unique_products.append(p)

    # Save to JSON
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in unique_products], f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]Saved {len(unique_products)} products to {output_path}[/green]")
    return unique_products


def load_products(path: str) -> list[Product]:
    """Load products from a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Product file not found: {path}\n"
            "Run 'python scripts/setup_data.py' first to download and index products."
        )

    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [Product(**item) for item in raw]
