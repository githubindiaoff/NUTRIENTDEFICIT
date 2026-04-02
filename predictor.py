import torch
from model_loader import model, tokenizer, device
from preprocessor import clean_text

label_names = [
    "Iron Deficiency",
    "Vitamin D Deficiency",
    "Vitamin B12 Deficiency",
    "Calcium Deficiency"
]

def predict(text):
    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence = torch.max(probs).item()
    prediction = torch.argmax(probs).item()

    return label_names[prediction], round(confidence * 100, 2)