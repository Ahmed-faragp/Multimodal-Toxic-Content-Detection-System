from datasets import load_dataset,concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig  
import pandas as pd

dataset = load_dataset("civil_comments")
def label_toxicity(example):
      return {
        "text": example["text"],
        "label": 1 if example["toxicity"] > 0.3 else 0
    }
train_data = dataset["train"].map(label_toxicity).select(range(80000))
safe_data = train_data.filter(lambda x: x["label"] == 0)
toxic_data = train_data.filter(lambda x: x["label"] == 1)
print(pd.Series(train_data["label"]).value_counts())
safe_data = safe_data.shuffle(seed=42).select(range(7500))
toxic_data = toxic_data.shuffle(seed=42).select(range(7500))
train_data = concatenate_datasets([safe_data, toxic_data]).shuffle(seed=42)
valid_data = dataset["validation"].map(label_toxicity).select(range(1500))
  
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
train_data = train_data.map(tokenize_function, batched=True)
valid_data = valid_data.map(tokenize_function, batched=True)
columns_to_remove = [
    "text",
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit"
]

train_data = train_data.remove_columns(columns_to_remove)
valid_data = valid_data.remove_columns(columns_to_remove)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


print(pd.Series(train_data["label"]).value_counts())

# print(test_data.column_names)
# print(valid_data.column_names)
lora_config =LoraConfig(
     r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    task_type="SEQ_CLS"
)   
model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        eval_strategy="epoch", 
        save_strategy="epoch",
        learning_rate=2e-4,
        save_total_limit=2
    ),
    train_dataset=train_data,
    eval_dataset=valid_data,
    
)
trainer.train()
trainer.save_model("./toxic_lora_model")
tokenizer.save_pretrained("./toxic_lora_model")
