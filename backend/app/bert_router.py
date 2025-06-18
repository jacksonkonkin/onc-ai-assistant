from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle

class BERTQueryRouter:
    def __init__(self, model_path="bert_query_classifier", label_encoder_path="label_encoder.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        self.model.eval()

    def classify(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        label = self.label_encoder.inverse_transform([predicted_class])[0]
        confidence = torch.softmax(logits, dim=1).max().item()
        return {"intent": label, "confidence": confidence}
