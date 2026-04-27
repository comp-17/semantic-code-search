"""
run_part1.py
------------
Part 1: Baseline Starter Corpus Queries - Track B Code RAG
Streams 1,000 Python functions from CodeSearchNet, indexes them into
ChromaDB, runs 10 natural language queries, and saves results_part1.md.
"""
import yaml
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from retriever import retrieve
from generator import get_client, generate


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


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


def load_codesearchnet(config):
    print(f"\n[1/3] Loading {config['starter_corpus_size']} functions from CodeSearchNet...")
    ds = load_dataset("code_search_net", "python", split="train",
                      streaming=True, trust_remote_code=True)
    items = []
    for row in tqdm(ds, total=config["starter_corpus_size"], desc="Streaming"):
        items.append(row)
        if len(items) >= config["starter_corpus_size"]:
            break
    print(f"    Loaded {len(items)} functions.")
    return items


def index_corpus(items, config):
    model = SentenceTransformer(config["embedding_model"])
    client = chromadb.PersistentClient(path=config["vector_db_path"])
    collection = client.get_or_create_collection(
        name=config["starter_collection"],
        metadata={"hnsw:space": "cosine"}
    )
    print(f"\n[2/3] Embedding and indexing {len(items)} functions...")
    if collection.count() >= len(items):
        print(f"    Collection already has {collection.count()} items. Skipping.")
        return
    BATCH = 100
    for start in tqdm(range(0, len(items), BATCH), desc="Indexing batches"):
        batch = items[start: start + BATCH]
        texts = [(r["func_documentation_string"] or "") + "\n" + r["func_code_string"]
                 for r in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(
            ids=[f"csn_{start + i}" for i in range(len(batch))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "func_name": r["func_name"],
                "repo":      r["repository_name"],
                "path":      r["func_path_in_repository"],
                "source":    "codesearchnet"
            } for r in batch]
        )
    print(f"    Indexed {collection.count()} functions total.")


def format_table_row(qid, query, chunks, answer):
    top = chunks[0] if chunks else {}
    citation = f"{top.get('repo','?')}/{top.get('path','?')}::{top.get('func_name','?')}"
    score = top.get("score", 0.0)
    sentences = answer.replace("\n", " ").split(". ")
    short_answer = ". ".join(sentences[:2]).strip()
    if not short_answer.endswith("."):
        short_answer += "."
    cited = any(c["func_name"] in answer for c in chunks)
    grounded = "Yes" if cited else "No"
    return f"| Q{qid:02d} | {query} | {citation} | {score:.4f} | {short_answer} | {grounded} |"


def print_and_save_table(rows, output_path="results_part1.md"):
    header = (
        "## Part 1 - Baseline Query Results (Starter Corpus)\n\n"
        "| Q-ID | Query | Top Retrieved (repo/path::func_name) "
        "| Sim. Score | Generated Answer (first 2 sentences) | Grounded? |\n"
        "|---|---|---|---|---|---|"
    )
    table = header + "\n" + "\n".join(rows)
    print("\n" + table)
    with open(output_path, "w") as f:
        f.write(table + "\n")
    print(f"\n[Saved to {output_path}]")


def main():
    config = load_config()
    client = get_client(config)
    items = load_codesearchnet(config)
    index_corpus(items, config)
    print(f"\n[3/3] Running {len(QUERIES)} queries...\n")
    rows = []
    for qid, query in enumerate(QUERIES, start=1):
        print(f"  Q{qid:02d}: {query}")
        chunks = retrieve(query, k=config["top_k"])
        answer = generate(query, chunks, client, config)
        row = format_table_row(qid, query, chunks, answer)
        rows.append(row)
        print(f"       -> {chunks[0]['repo']}/{chunks[0]['path']}::{chunks[0]['func_name']}  score={chunks[0]['score']}")
    print_and_save_table(rows)


if __name__ == "__main__":
    main()
