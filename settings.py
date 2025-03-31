import os
from pathlib import Path

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"

# Vector DB settings
VECTOR_DB_URL = "http://localhost:6333"
VECTOR_DB_NAME = "vector_db"

# Model settings
EMBEDDINGS = "NeuML/pubmedbert-base-embeddings"
LLM_PATH = "BioMistral-7B.Q4_K_M.gguf"

# Template
PROMPT_TEMPLATE = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""
