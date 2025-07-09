#!/usr/bin/env python3
"""
interactive_classifier.py
--------------------------------
Run `python interactive_classifier.py` and start asking questions.

Type 'exit' or 'quit' (case-insensitive) to leave.
"""

import sys
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Configuration – edit the list if your CSV filenames change
# ---------------------------------------------------------------------------
CSV_FILES = [
    "deployments.csv",
    "device_info.csv",
    "property.csv",
    "device_category.csv",
    "locations.csv",
]

MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Load data and model once
# ---------------------------------------------------------------------------
print("Loading CSV files...")
dfs = [pd.read_csv(path) for path in CSV_FILES]
df_all = pd.concat(dfs, ignore_index=True)

questions = df_all["question"].tolist()
labels    = df_all["label"].tolist()

print(f"Loaded {len(questions)} questions from {len(CSV_FILES)} files.")

print(f"Loading SentenceTransformer model “{MODEL_NAME}”...")
model = SentenceTransformer(MODEL_NAME)

print("Encoding questions (this happens only once)…")
embeddings = model.encode(questions, normalize_embeddings=True)
embeddings_tensor = torch.tensor(embeddings)
print("Model and embeddings ready!\n")

# ---------------------------------------------------------------------------
# Helper: classify a single query
# ---------------------------------------------------------------------------
def classify(query: str, top_k: int = 1) -> str:
    query_emb = model.encode(query, normalize_embeddings=True)
    sims = util.cos_sim(torch.tensor(query_emb), embeddings_tensor)[0]
    top_index = sims.topk(top_k).indices[0].item()
    return labels[top_index]

# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------
print("📝  Ask me something! (type 'exit' or 'quit' to leave)\n")

try:
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue  # empty line, prompt again
        label = classify(user_input)
        print(f"Predicted label: {label}\n")
except (KeyboardInterrupt, EOFError):
    print("\nGoodbye!")
    sys.exit(0)
