"""
retriever.py
------------
Retriever module for Track B Code RAG.

Exposes a single public function `retrieve` that accepts a natural language
query, embeds it using the configured sentence-transformers model, queries
the ChromaDB collection, and returns the top-k most similar code chunks
with their metadata and similarity scores.
"""

import yaml
from sentence_transformers import SentenceTransformer
import chromadb

__all__ = ["retrieve"]


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_collection(config: dict):
    """
    Initialize and return the ChromaDB persistent client and collection.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.

    Returns
    -------
    chromadb.Collection
        The persistent Chroma collection named in config.
    """
    client = chromadb.PersistentClient(path=config["vector_db_path"])
    collection = client.get_or_create_collection(
        name=config["starter_collection"],
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def retrieve(
    query: str,
    collection,
    model: SentenceTransformer,
    k: int = 4
) -> list:
    """
    Embed a natural language query and return the top-k most similar
    code chunks from the ChromaDB collection.

    Parameters
    ----------
    query : str
        Natural language question about Python functionality.
    collection : chromadb.Collection
        The populated Chroma collection to query against.
    model : SentenceTransformer
        The embedding model. Must match the model used during indexing.
    k : int, optional
        Number of top results to return. Defaults to 4.

    Returns
    -------
    list of dict
        Each dict contains:
        - 'rank'      : int   — rank position (1-indexed)
        - 'document'  : str   — the indexed text (docstring + code)
        - 'func_name' : str   — function name
        - 'repo'      : str   — repository name or 'local'
        - 'path'      : str   — file path within repository
        - 'source'    : str   — 'codesearchnet' or 'local'
        - 'score'     : float — cosine similarity score
    """
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        # Chroma cosine distance = 1 - cosine_similarity
        similarity = 1.0 - dist
        chunks.append({
            "rank":      rank,
            "document":  doc,
            "func_name": meta.get("func_name", "unknown"),
            "repo":      meta.get("repo", "unknown"),
            "path":      meta.get("path", "unknown"),
            "source":    meta.get("source", "codesearchnet"),
            "score":     round(similarity, 4)
        })

    return chunks


if __name__ == "__main__":
    # Quick smoke test
    config = load_config()
    model = SentenceTransformer(config["embedding_model"])
    collection = get_collection(config)

    test_query = "How do I compute cosine similarity between two vectors?"
    results = retrieve(test_query, collection, model, k=config["top_k"])

    print(f"\nQuery: {test_query}\n")
    for r in results:
        print(
            f"  [{r['rank']}] {r['repo']}/{r['path']}::{r['func_name']}"
            f"  (score={r['score']}, source={r['source']})"
        )
