# evaluate.py
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# --------------------------
# Config
# --------------------------
DATA_FILE = "transactions.csv"
MODEL_PATH = "model/lora_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Load dataset
# --------------------------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found. Run generate_dataset.py first.")

df = pd.read_csv(DATA_FILE)
df = df.sample(min(len(df), 5000), random_state=42)  # optional: limit eval size for speed

# --------------------------
# Load LLM model
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    device_map="auto" if DEVICE=="cuda" else None
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.to(DEVICE)
model.eval()

# --------------------------
# Prediction function
# --------------------------
def predict_category(transaction_text, max_new_tokens=32):
    prompt = f"Classify the following transaction:\n{transaction_text}\nCategory:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    pred_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract category from LLM output
    if "Category:" in pred_text:
        category = pred_text.split("Category:")[-1].strip().split("\n")[0]
    else:
        category = pred_text.strip().split("\n")[0]
    # Approximate confidence with length of output / max_new_tokens
    confidence = min(1.0, len(pred_text)/50)  # very rough heuristic
    return category, round(confidence,2)

# --------------------------
# Evaluate dataset
# --------------------------
y_true = df["category"].tolist()
y_pred = []
confidences = []

print("Evaluating LLM model on dataset...")
for txn in tqdm(df["transaction"].tolist()):
    cat, conf = predict_category(txn)
    y_pred.append(cat)
    confidences.append(conf)

df["predicted_category"] = y_pred
df["confidence"] = confidences

# --------------------------
# Metrics
# --------------------------
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=list(set(y_true)))
print("Confusion matrix shape:", cm.shape)
print(classification_report(y_true, y_pred, zero_division=0))

# --------------------------
# Save results
# --------------------------
results_dir = "model"
os.makedirs(results_dir, exist_ok=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(results_dir, "classification_report.csv"))
df.to_csv(os.path.join(results_dir, "predictions_with_confidence.csv"), index=False)

print("Saved classification_report.csv and predictions_with_confidence.csv")
