"""
retriever.py
------------
Retriever module for Track B Code RAG.
Exposes a single public function retrieve() that accepts a natural language
query, embeds it using the configured sentence-transformers model, queries
the ChromaDB collection, and returns the top-k most similar code chunks
with their metadata and similarity scores.
"""
import yaml
from sentence_transformers import SentenceTransformer
import chromadb

__all__ = ["retrieve"]


def retrieve(
    query: str,
    k: int = 4,
    config_path: str = "config.yaml"
) -> list:
    """
    Embed a natural language query and return the top-k most similar
    code chunks from the ChromaDB collection.

    Parameters
    ----------
    query : str
        Natural language question about Python functionality.
    k : int, optional
        Number of top results to return. Defaults to 4.
    config_path : str, optional
        Path to the YAML config file. Defaults to config.yaml.

    Returns
    -------
    list of dict
        Each dict contains:
        - rank      : int   - rank position (1-indexed)
        - document  : str   - the indexed text (docstring + code)
        - func_name : str   - function name
        - repo      : str   - repository name or local
        - path      : str   - file path within repository
        - source    : str   - codesearchnet or local
        - score     : float - cosine similarity score
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load embedding model
    model = SentenceTransformer(config["embedding_model"])

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=config["vector_db_path"])
    collection = client.get_or_create_collection(
        name=config["starter_collection"],
        metadata={"hnsw:space": "cosine"}
    )

    # Embed query and search
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    # Build result list
    chunks = []
    for rank, (doc, meta, dist) in enumerate(
        zip(results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]), start=1
    ):
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
    results = retrieve("How do I compute cosine similarity between two vectors?")
    for r in results:
        print(
            f"  [{r[chr(39)+'rank'+chr(39)]}] {r[chr(39)+'repo'+chr(39)]}/{r[chr(39)+'path'+chr(39)]}::{r[chr(39)+'func_name'+chr(39)]}"
            f"  (score={r[chr(39)+'score'+chr(39)]})"
        )
