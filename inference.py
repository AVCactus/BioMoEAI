# src/inference.py

import torch
from transformers import AutoTokenizer
from ensemble import TransformerEnsemble

# Set up the device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the expert model names (must match training).
expert_model_names = [
    "bert-base-uncased",
    "roberta-base",
    "xlnet-base-cased"
]

# Initialize the ensemble model.
ensemble_model = TransformerEnsemble(expert_model_names, device=device)
ensemble_model.to(device)

# Load the checkpoint (replace with your checkpoint path).
checkpoint_path = "checkpoints/checkpoint-step-50.pt"
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ensemble_model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {checkpoint_path}")
except Exception as e:
    print("Checkpoint loading failed. Proceeding with random initialization.")
    print(e)

def predict(texts):
    """
    Given a list of text strings, return the predicted class labels.
    """
    ensemble_model.eval()
    with torch.no_grad():
        ensemble_logits, _ = ensemble_model(texts)
        preds = torch.argmax(ensemble_logits, dim=1)
    return preds.cpu().numpy()

if __name__ == "__main__":
    sample_texts = [
        "This movie was fantastic! I really enjoyed it.",
        "The plot was boring and the acting was mediocre."
    ]
    predictions = predict(sample_texts)
    print("Predicted class labels:", predictions)
