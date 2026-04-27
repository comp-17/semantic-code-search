"""
generator.py
------------
Generator module for Track B Code RAG.

Assembles a grounded system prompt from retrieved code chunks and calls
the UTSA-hosted Llama 3.3 70B endpoint via the OpenAI-compatible API.
Citations are formatted as repo/path::func_name per the Track B spec.
"""

import yaml
from openai import OpenAI


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_client(config: dict) -> OpenAI:
    """
    Initialize and return an OpenAI-compatible client pointed at the
    UTSA Llama 3.3 70B endpoint.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.

    Returns
    -------
    OpenAI
        Configured client instance.
    """
    return OpenAI(
        api_key=config["utsa_api_key"],
        base_url=config["utsa_base_url"]
    )


def build_prompt(query: str, chunks: list) -> str:
    """
    Assemble the grounded context block from retrieved code chunks.

    Each chunk is formatted as:
        [N] repo: <repo>, path: <path>, func: <func_name>
            <docstring + code text>

    Parameters
    ----------
    query : str
        The original user query.
    chunks : list of dict
        Retrieved chunks from the retriever, each containing
        'rank', 'document', 'func_name', 'repo', 'path'.

    Returns
    -------
    str
        The assembled context string to be embedded in the prompt.
    """
    context_lines = []
    for chunk in chunks:
        header = (
            f"[{chunk['rank']}] repo: {chunk['repo']}, "
            f"path: {chunk['path']}, func: {chunk['func_name']}"
        )
        # Indent the code/doc text for readability
        body = "\n    ".join(chunk["document"].splitlines())
        context_lines.append(f"{header}\n    {body}")

    return "\n\n".join(context_lines)


def generate(
    query: str,
    chunks: list,
    client: OpenAI,
    config: dict,
    max_tokens: int = 300
) -> str:
    """
    Generate a grounded answer for a query given retrieved code chunks.

    The language model is instructed to answer solely from the provided
    code context and to cite sources as repo/path::func_name. If the
    context does not contain sufficient information, the model is directed
    to state that the answer cannot be determined from the available code.

    Parameters
    ----------
    query : str
        Natural language question about Python functionality.
    chunks : list of dict
        Retrieved code chunks from the retriever.
    client : OpenAI
        Configured OpenAI-compatible client.
    config : dict
        Loaded configuration dictionary.
    max_tokens : int, optional
        Maximum number of tokens in the generated response. Default 300.

    Returns
    -------
    str
        The model's generated answer with inline citations.
    """
    context = build_prompt(query, chunks)

    system_message = (
        "You are a helpful Python code assistant. "
        "Answer the user's question using ONLY the provided code context. "
        "Cite every source you use as repo/path::func_name. "
        "If the context does not contain enough information to answer "
        "the question, say: 'The provided code context does not contain "
        "sufficient information to answer this question.' "
        "Do not use any external knowledge beyond the provided context."
    )

    user_message = f"Context:\n\n{context}\n\nUser query: {query}"

    response = client.chat.completions.create(
        model=config["generator_model"],
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message}
        ],
        max_tokens=max_tokens,
        temperature=0.1
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Quick smoke test — requires a populated collection and active VPN
    from retriever import load_config, get_collection, retrieve
    from sentence_transformers import SentenceTransformer

    config = load_config()
    model  = SentenceTransformer(config["embedding_model"])
    collection = get_collection(config)
    client = get_client(config)

    test_query = "How do I compute cosine similarity between two vectors?"
    chunks = retrieve(test_query, collection, model, k=config["top_k"])
    answer = generate(test_query, chunks, client, config)

    print(f"\nQuery: {test_query}")
    print(f"\nAnswer:\n{answer}")
