from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import os

MODEL_PATH = "saved_model"

if os.path.exists(MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
else:
    print(f"'{MODEL_PATH}' not found. Falling back to distilbert-base-uncased with 4 labels.")
    MODEL_PATH_FALLBACK = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_FALLBACK)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_FALLBACK, num_labels=4, ignore_mismatched_sizes=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()