"""
ml_nlp_utils.py
---------------
A collection of lightweight ML and NLP utility functions implemented
in pure Python. No external ML libraries are required.
"""

import math
import re
from collections import Counter


def compute_cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Compute the cosine similarity between two numeric vectors.

    Cosine similarity measures the cosine of the angle between two
    vectors in an inner product space. It ranges from -1 (opposite
    directions) to 1 (identical directions), with 0 indicating
    orthogonality. Commonly used in NLP to measure semantic similarity
    between document embeddings or word vectors.

    Parameters
    ----------
    vec_a : list of float
        First input vector.
    vec_b : list of float
        Second input vector. Must be the same length as vec_a.

    Returns
    -------
    float
        Cosine similarity score between vec_a and vec_b.
        Returns 0.0 if either vector has zero magnitude.

    Raises
    ------
    ValueError
        If vec_a and vec_b have different lengths.

    Examples
    --------
    >>> compute_cosine_similarity([1, 0, 0], [1, 0, 0])
    1.0
    >>> compute_cosine_similarity([1, 0], [0, 1])
    0.0
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vectors must have the same length: {len(vec_a)} vs {len(vec_b)}"
        )

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def tokenize_and_count(text: str) -> dict:
    """
    Tokenize a string into lowercase word tokens and count their frequencies.

    Tokenization is performed by converting the input to lowercase and
    splitting on any sequence of non-alphanumeric characters (whitespace,
    punctuation, etc.). Empty tokens are discarded. The result is a
    dictionary mapping each unique token to its frequency count in the
    input text. Useful as a preprocessing step in NLP pipelines for
    building vocabulary statistics or simple bag-of-words features.

    Parameters
    ----------
    text : str
        Input string to tokenize and count.

    Returns
    -------
    dict
        A dictionary mapping token strings to their integer frequency counts,
        sorted in descending order of frequency.

    Examples
    --------
    >>> tokenize_and_count("the cat sat on the mat")
    {'the': 2, 'cat': 1, 'sat': 1, 'on': 1, 'mat': 1}
    >>> tokenize_and_count("Hello, hello! HELLO.")
    {'hello': 3}
    """
    tokens = re.split(r'[^a-zA-Z0-9]+', text.lower())
    tokens = [t for t in tokens if t]
    counts = Counter(tokens)
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def moving_average(values: list, window: int) -> list:
    """
    Compute a simple moving average over a list of numeric values.

    The moving average smooths a sequence by replacing each element with
    the mean of a fixed-size sliding window centered on that element.
    For positions near the boundaries where a full window is unavailable,
    the average is computed over the available elements only (equivalent
    to padding-free mode). Commonly used in time-series analysis, signal
    smoothing, and tracking training loss curves in machine learning.

    Parameters
    ----------
    values : list of float or int
        Input sequence of numeric values.
    window : int
        Size of the sliding window. Must be a positive integer.

    Returns
    -------
    list of float
        A list of the same length as values containing the moving averages.

    Raises
    ------
    ValueError
        If window is less than 1.

    Examples
    --------
    >>> moving_average([1, 2, 3, 4, 5], window=3)
    [1.5, 2.0, 3.0, 4.0, 4.5]
    >>> moving_average([10, 20, 30], window=2)
    [15.0, 15.0, 25.0]
    """
    if window < 1:
        raise ValueError(f"Window size must be at least 1, got {window}")

    result = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        window_vals = values[start:end]
        result.append(sum(window_vals) / len(window_vals))
    return result


def bag_of_words(corpus: list) -> tuple:
    """
    Build a bag-of-words (BoW) representation from a list of text documents.

    The bag-of-words model represents each document as a fixed-length
    vector of token counts over a shared vocabulary derived from the entire
    corpus. Tokenization is performed by lowercasing and splitting on
    non-alphanumeric characters. The vocabulary is sorted alphabetically.
    BoW is a foundational representation in NLP used for text classification,
    information retrieval, and as a baseline for embedding-based approaches.

    Parameters
    ----------
    corpus : list of str
        A list of raw text documents.

    Returns
    -------
    tuple
        A two-element tuple (vocab, vectors) where:
        - vocab is a sorted list of unique token strings across all documents.
        - vectors is a list of lists, where each inner list is the BoW count
          vector for the corresponding document, aligned to vocab.

    Examples
    --------
    >>> vocab, vecs = bag_of_words(["the cat", "the dog"])
    >>> vocab
    ['cat', 'dog', 'the']
    >>> vecs
    [[1, 0, 1], [0, 1, 1]]
    """
    tokenized = []
    for doc in corpus:
        tokens = re.split(r'[^a-zA-Z0-9]+', doc.lower())
        tokenized.append([t for t in tokens if t])

    vocab = sorted(set(token for doc_tokens in tokenized for token in doc_tokens))
    vocab_index = {token: i for i, token in enumerate(vocab)}

    vectors = []
    for doc_tokens in tokenized:
        vec = [0] * len(vocab)
        for token in doc_tokens:
            vec[vocab_index[token]] += 1
        vectors.append(vec)

    return vocab, vectors


def top_k_similar(query_vec: list, corpus_vecs: list, k: int = 5) -> list:
    """
    Return the indices of the top-k most similar vectors to a query vector.

    Similarity is measured using cosine similarity between the query vector
    and each vector in the corpus. Results are returned in descending order
    of similarity score. This function replicates the core ranking step of
    a dense retrieval system, such as those used in RAG pipelines, without
    requiring a dedicated vector database. Useful for small in-memory
    retrieval tasks or for understanding the ranking behavior of embedding
    models.

    Parameters
    ----------
    query_vec : list of float
        The query embedding vector.
    corpus_vecs : list of list of float
        A list of embedding vectors to rank against the query.
    k : int, optional
        Number of top results to return. Defaults to 5.

    Returns
    -------
    list of tuple
        A list of (index, similarity_score) tuples sorted by similarity
        in descending order, containing at most k entries.

    Examples
    --------
    >>> query = [1.0, 0.0]
    >>> corpus = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    >>> top_k_similar(query, corpus, k=2)
    [(0, 1.0), (2, 0.7071...)]
    """
    scores = []
    for idx, vec in enumerate(corpus_vecs):
        score = compute_cosine_similarity(query_vec, vec)
        scores.append((idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]
