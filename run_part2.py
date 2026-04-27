"""
run_part2.py
------------
Part 2: Extended Corpus Queries — Track B Code RAG

This script:
  1. Parses the 5 author-written Python functions from data/new_funcs/.
  2. Embeds and inserts them into the existing 'csn_python' collection
     with source='local' metadata.
  3. Runs 5 targeted queries (designed to surface the new functions).
  4. Runs 5 cross-corpus queries (may retrieve from either set).
  5. Prints and saves a results table to results_part2.md, annotating
     the origin of each top retrieved chunk.
"""

import os
import ast
import yaml
from sentence_transformers import SentenceTransformer

from retriever import get_collection, retrieve
from generator import get_client, generate


# ── Configuration ──────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Part 2 Queries ─────────────────────────────────────────────────────────────

# Targeted: each query is designed to surface one specific new function
TARGETED_QUERIES = [
    "How do I compute cosine similarity between two numeric vectors in Python?",
    "How can I tokenize a string and count the frequency of each word?",
    "How do I compute a moving average with a sliding window over a list?",
    "How can I build a bag-of-words representation from a list of documents?",
    "How do I find the top-k most similar vectors to a query vector?",
]

# Cross-corpus: general enough to potentially retrieve from either set
CROSS_CORPUS_QUERIES = [
    "How do I measure similarity between two sequences in Python?",
    "How can I process and analyze text data to extract word statistics?",
    "How do I implement a sliding window operation over a numeric list?",
    "How can I rank items by a numeric score in descending order?",
    "How do I represent text documents as numeric vectors in Python?",
]


# ── Function Extraction ────────────────────────────────────────────────────────

def extract_functions_from_file(filepath: str) -> list:
    """
    Parse a Python source file and extract all top-level functions
    as (name, docstring, source_code) tuples using the ast module.

    Parameters
    ----------
    filepath : str
        Path to the .py file to parse.

    Returns
    -------
    list of dict
        Each dict has keys: func_name, docstring, source, path.
    """
    with open(filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    source_lines = source.splitlines()
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node) or ""
            # Extract raw source lines for this function
            start = node.lineno - 1
            end   = node.end_lineno
            func_source = "\n".join(source_lines[start:end])
            functions.append({
                "func_name": node.name,
                "docstring": docstring,
                "source":    func_source,
                "path":      os.path.relpath(filepath)
            })

    return functions


def load_new_functions(new_funcs_dir: str) -> list:
    """Load and parse all .py files from the new_funcs directory."""
    print(f"\n[1/3] Loading author-written functions from '{new_funcs_dir}'...")
    all_functions = []

    for fname in sorted(os.listdir(new_funcs_dir)):
        if not fname.endswith(".py"):
            continue
        filepath = os.path.join(new_funcs_dir, fname)
        funcs = extract_functions_from_file(filepath)
        print(f"    {fname}: {len(funcs)} function(s) extracted")
        all_functions.extend(funcs)

    print(f"    Total: {len(all_functions)} functions loaded.")
    return all_functions


# ── Corpus Extension ───────────────────────────────────────────────────────────

def index_new_functions(
    functions: list,
    collection,
    model: SentenceTransformer
) -> None:
    """Embed and insert author-written functions into the existing collection."""
    print(f"\n[2/3] Embedding and inserting {len(functions)} new functions...")

    texts = [
        f["docstring"] + "\n" + f["source"]
        for f in functions
    ]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=[f"local_{i}" for i in range(len(functions))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "func_name": f["func_name"],
                "repo":      "local",
                "path":      f["path"],
                "source":    "local"
            }
            for f in functions
        ]
    )

    print(f"    Collection now contains {collection.count()} items total.")


# ── Results Table ──────────────────────────────────────────────────────────────

def format_row(
    qid: str,
    query: str,
    chunks: list,
    answer: str,
    query_type: str
) -> str:
    """Format a single results table row with source set annotation."""
    top = chunks[0] if chunks else {}
    citation = (
        f"{top.get('repo','?')}/{top.get('path','?')}::{top.get('func_name','?')}"
    )
    score = top.get("score", 0.0)

    # Determine source set of all top-k chunks
    sources = {c.get("source", "codesearchnet") for c in chunks}
    if sources == {"local"}:
        source_set = "New Items"
    elif sources == {"codesearchnet"}:
        source_set = "Starter"
    else:
        source_set = "Mixed"

    # First two sentences
    sentences = answer.replace("\n", " ").split(". ")
    short_answer = ". ".join(sentences[:2]).strip()
    if not short_answer.endswith("."):
        short_answer += "."

    # Grounding check
    cited    = any(c["func_name"] in answer for c in chunks)
    grounded = "Yes" if cited else "No"

    return (
        f"| {qid} "
        f"| {query} "
        f"| `{citation}` "
        f"| {source_set} "
        f"| {score:.4f} "
        f"| {short_answer} "
        f"| {grounded} |"
    )


def print_and_save_table(
    targeted_rows: list,
    cross_rows: list,
    output_path: str = "results_part2.md"
) -> None:
    header = (
        "| Q-ID | Query | Top Retrieved (`repo/path::func_name`) "
        "| Source Set | Sim. Score | Generated Answer (first 2 sentences) | Grounded? |\n"
        "|---|---|---|---|---|---|---|"
    )

    targeted_section = (
        "## Part 2 — Targeted Query Results (New Functions)\n\n"
        + header + "\n"
        + "\n".join(targeted_rows)
    )

    cross_section = (
        "\n\n## Part 2 — Cross-Corpus Query Results\n\n"
        "> Source Set: **Starter** = CodeSearchNet only | "
        "**New Items** = author-written only | **Mixed** = both\n\n"
        + header + "\n"
        + "\n".join(cross_rows)
    )

    full_table = targeted_section + cross_section

    print("\n" + full_table)
    with open(output_path, "w") as f:
        f.write(full_table + "\n")
    print(f"\n[Saved to {output_path}]")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    config     = load_config()
    model      = SentenceTransformer(config["embedding_model"])
    collection = get_collection(config)
    client     = get_client(config)

    # Step 1 & 2: Load and index new functions
    functions = load_new_functions(config["new_funcs_dir"])
    index_new_functions(functions, collection, model)

    # Step 3: Targeted queries
    print(f"\n[3/3] Running {len(TARGETED_QUERIES)} targeted queries...")
    targeted_rows = []
    for i, query in enumerate(TARGETED_QUERIES, start=1):
        qid = f"T{i:02d}"
        print(f"  {qid}: {query}")
        chunks = retrieve(query, collection, model, k=config["top_k"])
        answer = generate(query, chunks, client, config)
        row    = format_row(qid, query, chunks, answer, "targeted")
        targeted_rows.append(row)
        print(f"       → {chunks[0]['func_name']}  "
              f"source={chunks[0]['source']}  score={chunks[0]['score']}")

    # Step 4: Cross-corpus queries
    print(f"\n     Running {len(CROSS_CORPUS_QUERIES)} cross-corpus queries...")
    cross_rows = []
    for i, query in enumerate(CROSS_CORPUS_QUERIES, start=1):
        qid = f"C{i:02d}"
        print(f"  {qid}: {query}")
        chunks = retrieve(query, collection, model, k=config["top_k"])
        answer = generate(query, chunks, client, config)
        row    = format_row(qid, query, chunks, answer, "cross")
        cross_rows.append(row)
        print(f"       → {chunks[0]['func_name']}  "
              f"source={chunks[0]['source']}  score={chunks[0]['score']}")

    print_and_save_table(targeted_rows, cross_rows)


if __name__ == "__main__":
    main()
