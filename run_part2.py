"""
run_part2.py
------------
Part 2: Extended Corpus Queries - Track B Code RAG
Parses author-written Python functions, indexes them into the existing
ChromaDB collection, runs 5 targeted and 5 cross-corpus queries,
and saves results_part2.md.
"""
import os
import ast
import yaml
from sentence_transformers import SentenceTransformer
import chromadb
from retriever import retrieve
from generator import get_client, generate


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


TARGETED_QUERIES = [
    "How do I compute cosine similarity between two numeric vectors in Python?",
    "How can I tokenize a string and count the frequency of each word?",
    "How do I compute a moving average with a sliding window over a list?",
    "How can I build a bag-of-words representation from a list of documents?",
    "How do I find the top-k most similar vectors to a query vector?",
]

CROSS_CORPUS_QUERIES = [
    "How do I measure similarity between two sequences in Python?",
    "How can I process and analyze text data to extract word statistics?",
    "How do I implement a sliding window operation over a numeric list?",
    "How can I rank items by a numeric score in descending order?",
    "How do I represent text documents as numeric vectors in Python?",
]


def extract_functions_from_file(filepath):
    with open(filepath, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    source_lines = source.splitlines()
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node) or ""
            func_source = "\n".join(source_lines[node.lineno - 1:node.end_lineno])
            functions.append({
                "func_name": node.name,
                "docstring": docstring,
                "source":    func_source,
                "path":      os.path.relpath(filepath)
            })
    return functions


def load_new_functions(new_funcs_dir):
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


def index_new_functions(functions, config):
    model = SentenceTransformer(config["embedding_model"])
    client = chromadb.PersistentClient(path=config["vector_db_path"])
    collection = client.get_or_create_collection(
        name=config["starter_collection"],
        metadata={"hnsw:space": "cosine"}
    )
    # Check if already indexed to avoid duplicate ID errors on re-run
    existing = collection.get(ids=[f"local_{i}" for i in range(len(functions))])
    if len(existing["ids"]) == len(functions):
        print(f"    Local functions already indexed. Skipping.")
        return
    print(f"\n[2/3] Embedding and inserting {len(functions)} new functions...")
    texts = [f["docstring"] + "\n" + f["source"] for f in functions]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    collection.add(
        ids=[f"local_{i}" for i in range(len(functions))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "func_name": f["func_name"],
            "repo":      "local",
            "path":      f["path"],
            "source":    "local"
        } for f in functions]
    )
    print(f"    Collection now contains {collection.count()} items total.")


def format_row(qid, query, chunks, answer):
    top = chunks[0] if chunks else {}
    citation = f"{top.get('repo','?')}/{top.get('path','?')}::{top.get('func_name','?')}"
    score = top.get("score", 0.0)
    sources = {c.get("source", "codesearchnet") for c in chunks}
    if sources == {"local"}:
        source_set = "New Items"
    elif sources == {"codesearchnet"}:
        source_set = "Starter"
    else:
        source_set = "Mixed"
    sentences = answer.replace("\n", " ").split(". ")
    short_answer = ". ".join(sentences[:2]).strip()
    if not short_answer.endswith("."):
        short_answer += "."
    cited = any(c["func_name"] in answer for c in chunks)
    grounded = "Yes" if cited else "No"
    return f"| {qid} | {query} | {citation} | {source_set} | {score:.4f} | {short_answer} | {grounded} |"


def print_and_save_table(targeted_rows, cross_rows, output_path="results_part2.md"):
    header = (
        "| Q-ID | Query | Top Retrieved (repo/path::func_name) "
        "| Source Set | Sim. Score | Generated Answer (first 2 sentences) | Grounded? |\n"
        "|---|---|---|---|---|---|---|"
    )
    targeted_section = (
        "## Part 2 - Targeted Query Results (New Functions)\n\n"
        + header + "\n" + "\n".join(targeted_rows)
    )
    cross_section = (
        "\n\n## Part 2 - Cross-Corpus Query Results\n\n"
        "> Source Set: Starter = CodeSearchNet only | New Items = author-written only | Mixed = both\n\n"
        + header + "\n" + "\n".join(cross_rows)
    )
    full_table = targeted_section + cross_section
    print("\n" + full_table)
    with open(output_path, "w") as f:
        f.write(full_table + "\n")
    print(f"\n[Saved to {output_path}]")


def main():
    config = load_config()
    client = get_client(config)
    functions = load_new_functions(config["new_funcs_dir"])
    index_new_functions(functions, config)
    print(f"\n[3/3] Running {len(TARGETED_QUERIES)} targeted queries...")
    targeted_rows = []
    for i, query in enumerate(TARGETED_QUERIES, start=1):
        qid = f"T{i:02d}"
        print(f"  {qid}: {query}")
        chunks = retrieve(query, k=config["top_k"])
        answer = generate(query, chunks, client, config)
        targeted_rows.append(format_row(qid, query, chunks, answer))
        print(f"       -> {chunks[0]['func_name']}  source={chunks[0]['source']}  score={chunks[0]['score']}")
    print(f"\n     Running {len(CROSS_CORPUS_QUERIES)} cross-corpus queries...")
    cross_rows = []
    for i, query in enumerate(CROSS_CORPUS_QUERIES, start=1):
        qid = f"C{i:02d}"
        print(f"  {qid}: {query}")
        chunks = retrieve(query, k=config["top_k"])
        answer = generate(query, chunks, client, config)
        cross_rows.append(format_row(qid, query, chunks, answer))
        print(f"       -> {chunks[0]['func_name']}  source={chunks[0]['source']}  score={chunks[0]['score']}")
    print_and_save_table(targeted_rows, cross_rows)


if __name__ == "__main__":
    main()
