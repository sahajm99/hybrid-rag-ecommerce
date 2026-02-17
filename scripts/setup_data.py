#!/usr/bin/env python3
"""Download product data from HuggingFace and build search indices."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console

from src.config import load_config
from src.data_loader import download_and_prepare
from src.pipeline import HybridRAGPipeline

app = typer.Typer(help="Download product data and build search indices.")
console = Console()


@app.command()
def setup(
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Path to config file"),
):
    """Download Amazon product data and build dense + sparse indices."""
    config = load_config(config_path)

    # Step 1: Download and prepare data
    console.print("\n[bold blue]Step 1/2: Downloading and preparing product data...[/bold blue]\n")
    data_path = Path(config.data_path)

    if data_path.exists():
        console.print(f"[yellow]Data file already exists: {data_path}[/yellow]")
        overwrite = typer.confirm("Re-download and overwrite?", default=False)
        if not overwrite:
            console.print("Skipping download, using existing data.")
        else:
            download_and_prepare(
                categories=config.categories,
                per_category=config.products_per_category,
                output_path=str(data_path),
            )
    else:
        download_and_prepare(
            categories=config.categories,
            per_category=config.products_per_category,
            output_path=str(data_path),
        )

    # Step 2: Build indices
    console.print("\n[bold blue]Step 2/2: Building search indices...[/bold blue]\n")
    pipeline = HybridRAGPipeline(config_path=config_path)
    pipeline.index(str(data_path))

    console.print("\n[bold green]Setup complete! You can now run searches:[/bold green]")
    console.print('  python scripts/search.py "affordable running shoes for women"')
    console.print('  python scripts/search.py --interactive')


if __name__ == "__main__":
    app()
