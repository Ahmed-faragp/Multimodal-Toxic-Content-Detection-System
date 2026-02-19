from datasets import load_dataset
dataset = load_dataset("civil_comments")
print(dataset.column_names)
train_data = dataset["train"].shuffle(seed=42).select(range(2000))
valid_data = dataset["validation"].shuffle(seed=42).select(range(500))
train_data