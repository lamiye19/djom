import pandas as pd
file = "analyse/analyse.csv"
df = pd.read_csv(file)

sentences = df["entree"]
labels = df["intention"]

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import joblib


model = AutoModelForSequenceClassification.from_pretrained("./bert-intentions")
tokenizer = AutoTokenizer.from_pretrained("./bert-intentions")
le = joblib.load("./bert-intentions/label_encoder.pkl")

new_questions = ["C'est quoi un master en Big Data ?", "Quels métiers puis-je faire après un bac D ?", "ok", "Bonjour je suis en licence big data. Que puis je faire après"]
def detect_intention(questions):
    encodings = tokenizer(questions, truncation=True, padding=True, max_length=64, return_tensors="pt")
    outputs = model(**encodings)
    preds = torch.argmax(outputs.logits, dim=1)
    pred_intention = le.inverse_transform(preds.numpy())
    return pred_intention[0]
"""
for q in new_questions:
    print(f"Question: {q}")
    print(f"Intention prédite: {detect_intention(q)}\n")
    """

