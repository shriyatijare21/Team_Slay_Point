# train_model_fast.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

# --------------------------
# Load dataset generated from generate_dataset.py
# --------------------------
DATA_FILE = "transactions.csv"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found. Run generate_dataset.py first.")

df = pd.read_csv(DATA_FILE)

# --------------------------
# For FAST training: use small subset
# --------------------------
train_df, test_df = train_test_split(df, test_size=0.05, stratify=df["category"], random_state=42)
train_df = train_df.sample(min(len(train_df), 5000), random_state=42)
test_df = test_df.sample(min(len(test_df), 1000), random_state=42)

# Prompt formatting
def format_example(example):
    return f"Classify the following transaction:\n{example['transaction']}\nCategory: {example['category']}"

train_df["text"] = train_df.apply(format_example, axis=1)
test_df["text"] = test_df.apply(format_example, axis=1)

train_dataset = Dataset.from_pandas(train_df[["text"]])
test_dataset = Dataset.from_pandas(test_df[["text"]])

# --------------------------
# Load small LLM
# --------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B"  # adjust to smaller variant for faster training if needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=None
).to(device)

# --------------------------
# LoRA configuration
# --------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# Tokenization
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)  # shorter max_length for speed

train_tokenized = train_dataset.map(tokenize, batched=True)
test_tokenized = test_dataset.map(tokenize, batched=True)

# --------------------------
# Training arguments (fast)
# --------------------------
training_args = TrainingArguments(
    output_dir="model/lora_llm",
    per_device_train_batch_size=1,  # small batch for fast run
    per_device_eval_batch_size=1,
    num_train_epochs=1,             # only 1 epoch
    logging_steps=10,
    save_steps=100,
    fp16=torch.cuda.is_available(),
    learning_rate=2e-4,
    report_to=None,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    data_collator=data_collator
)

# --------------------------
# Start training
# --------------------------
trainer.train()

# Save LoRA adapter + tokenizer
trainer.model.save_pretrained("model/lora_adapter")
tokenizer.save_pretrained("model/lora_adapter")
print("Saved LoRA fine-tuned model in model/lora_adapter/")

# --------------------------
# Quick inference function
# --------------------------
def predict_transaction(transaction: str):
    prompt = f"Classify the following transaction:\n{transaction}\nCategory:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    while True:
        txn = input("Enter a transaction (or 'quit' to exit): ")
        if txn.lower() == "quit":
            break
        pred = predict_transaction(txn)
        print(f"Predicted Category: {pred}")
