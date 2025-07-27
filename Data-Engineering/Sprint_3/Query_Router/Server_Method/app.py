# app.py
import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# Load the CSV files and model ONCE
file_paths = [
    "deployments.csv",
    "device_info.csv",
    "property.csv",
    "device_category.csv",
    "locations.csv",
    "deployments_2.csv",
    "device_info_2.csv",
    "property_2.csv",
    "device_category_2.csv",
    "locations_2.csv",
    "data_and_knowledge_queries.csv"
]

dfs = [pd.read_csv(path) for path in file_paths]
df_all = pd.concat(dfs, ignore_index=True)

questions = df_all["question"].tolist()
labels = df_all["label"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(questions, normalize_embeddings=True)
embeddings_tensor = torch.tensor(embeddings)

# Set up FastAPI
app = FastAPI()

# Input schema
class QueryInput(BaseModel):
    query: str
    top_k: int = 1

# Output schema
class QueryOutput(BaseModel):
    label: str

# Endpoint
@app.post("/classify", response_model=QueryOutput)
def classify_query(data: QueryInput):
    query_emb = model.encode(data.query, normalize_embeddings=True)
    similarities = util.cos_sim(torch.tensor(query_emb), embeddings_tensor)[0]
    top_index = similarities.topk(data.top_k).indices[0].item()
    label = labels[top_index]
    return QueryOutput(label=label)
