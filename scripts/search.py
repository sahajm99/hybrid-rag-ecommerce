#!/usr/bin/env python3
"""Search products using hybrid RAG pipeline."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from src.pipeline import HybridRAGPipeline

app = typer.Typer(help="Hybrid RAG product search CLI.")
console = Console()


def _display_retrieval_only(products, query: str) -> None:
    """Display raw retrieval results as a table (no LLM)."""
    if not products:
        console.print(f"[yellow]No products found for \"{query}\"[/yellow]")
        return

    table = Table(title=f"Retrieval Results: \"{query}\"", show_lines=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Title", style="cyan", max_width=50)
    table.add_column("Price", style="green", width=10)
    table.add_column("Category", style="magenta", width=25)
    table.add_column("URL", style="blue", max_width=45)

    for i, p in enumerate(products, 1):
        table.add_row(
            str(i),
            p.title[:50],
            f"${p.price:.2f}",
            p.category,
            p.url,
        )

    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(None, help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM, show raw retrieval results"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive search mode"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Path to config file"),
):
    """Search for products using hybrid RAG.

    Examples:
        python scripts/search.py "affordable running shoes for women"
        python scripts/search.py "lightweight laptop under 1000" --top-k 3
        python scripts/search.py "wireless headphones" --no-llm
        python scripts/search.py --interactive
    """
    try:
        pipeline = HybridRAGPipeline(config_path=config_path)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if interactive:
        _interactive_mode(pipeline, top_k, no_llm)
        return

    if query is None:
        console.print("[red]Please provide a search query or use --interactive mode.[/red]")
        console.print('Example: python scripts/search.py "wireless headphones under $100"')
        raise typer.Exit(1)

    _run_search(pipeline, query, top_k, no_llm)


def _run_search(pipeline: HybridRAGPipeline, query: str, top_k: int, no_llm: bool) -> None:
    """Execute a single search and display results."""
    console.print(f"\n[bold]Searching for:[/bold] \"{query}\"\n")

    if no_llm:
        products = pipeline.search_retrieval_only(query, top_k=top_k)
        _display_retrieval_only(products, query)
    else:
        result = pipeline.search(query, top_k=top_k)

        if not result.products:
            console.print(f"[yellow]{result.formatted_output}[/yellow]")
            return

        # Show parsed query info
        p = result.parsed_query
        if p.min_price or p.max_price or p.category_hint:
            parts = []
            if p.min_price:
                parts.append(f"min: ${p.min_price:.0f}")
            if p.max_price:
                parts.append(f"max: ${p.max_price:.0f}")
            if p.category_hint:
                parts.append(f"category: {p.category_hint}")
            console.print(f"[dim]Parsed constraints: {', '.join(parts)}[/dim]")
            console.print(f"[dim]Search text: \"{p.search_text}\"[/dim]\n")

        console.print(Markdown(result.formatted_output))


def _interactive_mode(pipeline: HybridRAGPipeline, top_k: int, no_llm: bool) -> None:
    """Interactive search loop."""
    console.print("\n[bold blue]Hybrid RAG Product Search — Interactive Mode[/bold blue]")
    console.print("[dim]Type your query and press Enter. Type 'quit' to exit.[/dim]\n")

    while True:
        try:
            query = console.input("[bold green]Search>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        _run_search(pipeline, query, top_k, no_llm)
        console.print()


if __name__ == "__main__":
    app()
