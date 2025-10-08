import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import joblib
from entities import intent_data


#data = pd.read_csv("analyse/analyse.csv")
data = pd.DataFrame([
    {"question": t, "intention": ann["intent"], "entities": ann["entities"]}
    for t, ann in intent_data
])

# Encoder les intentions en nombres
le = LabelEncoder()
data['label'] = le.fit_transform(data['intention'])

# Split train / test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['question'].tolist(),
    data['label'].tolist(),
    test_size=0.2,
    random_state=42
)


tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")  # support français

# Tokenizer
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64)

# Dataset PyTorch
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = IntentDataset(train_encodings, train_labels)
test_dataset = IntentDataset(test_encodings, test_labels)

# Chargement modele
num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-uncased",
    num_labels=num_labels
)


# Fonction de métriques
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)
training_args.set_evaluate(strategy="epoch", delay=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()


# Évaluer
metrics = trainer.evaluate()
print(metrics)

# Enregistrement
trainer.save_model("./bert-intentions")
tokenizer.save_pretrained("./bert-intentions")
joblib.dump(le, "./bert-intentions/label_encoder.pkl")

# Prédire de nouvelles questions
new_questions = ["C'est quoi un master en Big Data ?", "Quels métiers puis-je faire après un bac D ?", "ok", "Bonjour je suis en licence big data. Que puis je faire après"]
new_encodings = tokenizer(new_questions, truncation=True, padding=True, max_length=64, return_tensors="pt")
outputs = model(**new_encodings)
preds = torch.argmax(outputs.logits, dim=1)
predicted_intentions = le.inverse_transform(preds.numpy())

for q, intent in zip(new_questions, predicted_intentions):
    print(f"Question: {q}")
    print(f"Intention prédite: {intent}\n")
