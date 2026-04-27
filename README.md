# Assignment 4 - Track B: Code RAG
CS 6263 - LLM and Agentic Systems
Author: fpb170 | Track: B (Code RAG, Short)

## Track Declaration
Track B - Code RAG
- Starter corpus: 1000 Python functions from CodeSearchNet (Python split)
- New items (Part 2): 5 author-written functions in data/new_funcs/ml_nlp_utils.py
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Generator model: llama-3.3-70b-instruct-awq (UTSA-hosted endpoint)
- Vector database: ChromaDB (persistent client, cosine similarity, top-k=4)

## Hugging Face Resources Used
- code_search_net (python config): Starter corpus
- sentence-transformers/all-MiniLM-L6-v2: Embedding model

## Setup
    conda activate llm
    pip install -r requirements.txt

## Reproducing Results
Part 1: python run_part1.py -> results saved to results_part1.md
Part 2: python run_part2.py -> results saved to results_part2.md
Note: UTSA Llama endpoint requires VPN access to the UTSA network.

## Vector Database
Fully reproducible by running scripts in order:
    python run_part1.py
    python run_part2.py

## Configuration
- embedding_model: sentence-transformers/all-MiniLM-L6-v2
- generator_model: llama-3.3-70b-instruct-awq
- top_k: 4
- distance_metric: cosine
- vector_db_path: ./data/chroma_code
- starter_collection: csn_python
- starter_corpus_size: 1000

## Results Summary
- Part 1: 9/10 queries grounded correctly.
- Part 2 Targeted: 5/5 queries surfaced the correct author-written function, scores 0.6967-0.8461.
- Part 2 Cross-Corpus: 4/5 retrieved local functions as top-1; all 5 had Mixed source sets.

See REPORT.md for full tables and reflection.