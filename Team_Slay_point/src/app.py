# app.py
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = Flask(__name__)

# --------------------------
# Config
# --------------------------
MODEL_PATH = "model/lora_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADMIN_PASSWORD = "admin123"

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
    confidence = min(1.0, len(pred_text)/50)  # heuristic
    return category, round(confidence,2)

# --------------------------
# Routes
# --------------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    password = data.get("password")
    return jsonify({"success": password == ADMIN_PASSWORD})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    transaction = data.get("transaction", "")
    if not transaction:
        return jsonify({"error": "No transaction provided"}), 400

    category, confidence = predict_category(transaction)
    return jsonify({
        "transaction": transaction,
        "category": category,
        "confidence": confidence
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
