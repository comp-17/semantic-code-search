"""
run_part1.py
------------
Part 1: Baseline Starter Corpus Queries — Track B Code RAG

This script:
  1. Streams the first 1,000 Python functions from CodeSearchNet (HuggingFace).
  2. Embeds each function (docstring + code) using all-MiniLM-L6-v2.
  3. Populates a persistent ChromaDB collection named 'csn_python'.
  4. Runs 10 natural language queries about Python functionality.
  5. Prints and saves a results table to results_part1.md.
"""

import os
import yaml
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb

from retriever import get_collection, retrieve
from generator import get_client, generate


# ── Configuration ─────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Part 1 Queries ─────────────────────────────────────────────────────────────

QUERIES = [
    "How do I parse command line arguments in Python?",
    "How can I make an HTTP GET request and handle connection errors?",
    "How do I read and write a JSON file in Python?",
    "How can I sort a list of dictionaries by a specific key?",
    "How do I implement a retry mechanism for failed network requests?",
    "How can I flatten a nested list in Python?",
    "How do I find and count duplicate items in a list?",
    "How can I paginate through results from a REST API?",
    "How do I extract all URLs from a string using regular expressions?",
    "How can I compute the frequency of words in a text string?",
]


# ── Corpus Loading ─────────────────────────────────────────────────────────────

def load_codesearchnet(config: dict) -> list:
    """Stream the first N Python functions from CodeSearchNet."""
    print(f"\n[1/3] Loading {config['starter_corpus_size']} functions "
          f"from CodeSearchNet (Python split)...")

    ds = load_dataset(
        "code_search_net",
        "python",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    items = []
    for i, row in enumerate(tqdm(ds, total=config["starter_corpus_size"],
                                  desc="Streaming")):
        items.append(row)
        if len(items) >= config["starter_corpus_size"]:
            break

    print(f"    Loaded {len(items)} functions.")
    return items


def index_corpus(items: list, collection, model: SentenceTransformer) -> None:
    """Embed and insert CodeSearchNet functions into ChromaDB."""
    print(f"\n[2/3] Embedding and indexing {len(items)} functions...")

    # Check if already indexed
    if collection.count() >= len(items):
        print(f"    Collection already contains {collection.count()} items. "
              f"Skipping indexing.")
        return

    BATCH = 100
    for start in tqdm(range(0, len(items), BATCH), desc="Indexing batches"):
        batch = items[start: start + BATCH]

        texts = [
            (r["func_documentation_string"] or "") + "\n" + r["func_code_string"]
            for r in batch
        ]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=[f"csn_{start + i}" for i in range(len(batch))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "func_name": r["func_name"],
                    "repo":      r["repository_name"],
                    "path":      r["func_path_in_repository"],
                    "source":    "codesearchnet"
                }
                for r in batch
            ]
        )

    print(f"    Indexed {collection.count()} functions total.")


# ── Results Table ──────────────────────────────────────────────────────────────

def format_table_row(qid: int, query: str, chunks: list, answer: str) -> str:
    """Format a single results table row."""
    top = chunks[0] if chunks else {}
    citation = f"{top.get('repo','?')}/{top.get('path','?')}::{top.get('func_name','?')}"
    score    = top.get("score", 0.0)

    # First two sentences of answer
    sentences = answer.replace("\n", " ").split(". ")
    short_answer = ". ".join(sentences[:2]).strip()
    if not short_answer.endswith("."):
        short_answer += "."

    # Simple grounding check: does the answer cite at least one func_name?
    cited = any(c["func_name"] in answer for c in chunks)
    grounded = "Yes" if cited else "No"

    return (
        f"| Q{qid:02d} "
        f"| {query} "
        f"| `{citation}` "
        f"| {score:.4f} "
        f"| {short_answer} "
        f"| {grounded} |"
    )


def print_and_save_table(rows: list, output_path: str = "results_part1.md") -> None:
    header = (
        "## Part 1 — Baseline Query Results (Starter Corpus)\n\n"
        "| Q-ID | Query | Top Retrieved (`repo/path::func_name`) "
        "| Sim. Score | Generated Answer (first 2 sentences) | Grounded? |\n"
        "|---|---|---|---|---|---|"
    )
    table = header + "\n" + "\n".join(rows)

    print("\n" + table)
    with open(output_path, "w") as f:
        f.write(table + "\n")
    print(f"\n[Saved to {output_path}]")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    config     = load_config()
    model      = SentenceTransformer(config["embedding_model"])
    collection = get_collection(config)
    client     = get_client(config)

    # Step 1 & 2: Load and index corpus
    items = load_codesearchnet(config)
    index_corpus(items, collection, model)

    # Step 3: Run queries
    print(f"\n[3/3] Running {len(QUERIES)} queries...\n")
    rows = []
    for qid, query in enumerate(QUERIES, start=1):
        print(f"  Q{qid:02d}: {query}")
        chunks = retrieve(query, collection, model, k=config["top_k"])
        answer = generate(query, chunks, client, config)
        row    = format_table_row(qid, query, chunks, answer)
        rows.append(row)
        print(f"       → {chunks[0]['repo']}/{chunks[0]['path']}::{chunks[0]['func_name']}"
              f"  score={chunks[0]['score']}")

    print_and_save_table(rows)


if __name__ == "__main__":
    main()
