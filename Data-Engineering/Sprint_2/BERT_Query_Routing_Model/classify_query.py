from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert_query_classifier")
tokenizer = BertTokenizer.from_pretrained("bert_query_classifier")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Predict function
def classify_query(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = outputs.logits.argmax(dim=-1).item()
    return label_encoder.inverse_transform([pred_id])[0]

# Example
if __name__ == "__main__":
    query = input("Enter a query: ")
    label = classify_query(query)
    print("Predicted label:", label)
