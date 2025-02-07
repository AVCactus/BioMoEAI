# src/app.py

from flask import Flask, request, jsonify
import torch
from ensemble import TransformerEnsemble

app = Flask(__name__)

# Set device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define expert model names (must be same as training).
expert_model_names = [
    "bert-base-uncased",
    "roberta-base",
    "xlnet-base-cased"
]

# Initialize the ensemble model.
ensemble_model = TransformerEnsemble(expert_model_names, device=device)
ensemble_model.to(device)

# Optionally load a checkpoint.
checkpoint_path = "checkpoints/checkpoint-step-50.pt"
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ensemble_model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {checkpoint_path}")
except Exception as e:
    print("Checkpoint loading failed. Proceeding with random initialization.")
    print(e)

def predict(texts):
    ensemble_model.eval()
    with torch.no_grad():
        ensemble_logits, _ = ensemble_model(texts)
        preds = torch.argmax(ensemble_logits, dim=1)
    return preds.cpu().numpy().tolist()

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json(force=True)
    if "texts" not in data:
        return jsonify({"error": "Missing 'texts' field in JSON payload."}), 400
    texts = data["texts"]
    predictions = predict(texts)
    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
