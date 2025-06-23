from transformers import BertForSequenceClassification, BertTokenizer
from huggingface_hub import hf_hub_download
import torch
import pickle

# Model repo
repo_id = "kgosal03/bert-query-classifier"

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(repo_id)
tokenizer = BertTokenizer.from_pretrained(repo_id)

# Load label encoder
label_path = hf_hub_download(repo_id, filename="label_encoder.pkl")
with open(label_path, "rb") as f:
    label_encoder = pickle.load(f)

# Set model to evaluation mode
model.eval()

# 💬 Enter a query
query = input("Enter your query: ")

# Tokenize input
inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class_id = torch.argmax(outputs.logits, dim=1).item()

# Decode the label
predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]

print(f"Predicted label: {predicted_label}")
