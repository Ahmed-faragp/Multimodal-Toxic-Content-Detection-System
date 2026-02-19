import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("./toxic_lora_model")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model = PeftModel.from_pretrained(model, "./toxic_lora_model")
model.eval()
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        toxicity_score = probabilities[0][1].item()
        if toxicity_score > 0.4:
            return "Toxic"
        else:
            return "Safe"
     

print(classify_text("I dont like you"))
