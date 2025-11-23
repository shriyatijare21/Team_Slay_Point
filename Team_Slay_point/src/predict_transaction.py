# predict_transaction.py
import os
import joblib

# Load the saved objects
vec = joblib.load(os.path.join("model", "vectorizer.pkl"))
le = joblib.load(os.path.join("model", "label_encoder.pkl"))
clf = joblib.load(os.path.join("model", "model.pkl"))

def predict_transaction(raw_txn):
    # Transform input
    X_vec = vec.transform([raw_txn])
    
    # Predict
    y_pred_enc = clf.predict(X_vec)
    y_pred = le.inverse_transform(y_pred_enc)[0]
    
    # Predict probability
    probs = clf.predict_proba(X_vec)[0]
    class_index = list(clf.classes_).index(y_pred_enc[0])
    confidence = probs[class_index]
    
    return y_pred, confidence

if __name__ == "__main__":
    while True:
        raw_txn = input("Enter a transaction (or 'quit' to exit): ")
        if raw_txn.lower() == "quit":
            break
        category, confidence = predict_transaction(raw_txn)
        print(f"Predicted Category: {category}, Confidence: {confidence:.2f}")
