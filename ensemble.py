# src/ensemble.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerEnsemble(nn.Module):
    """
    TransformerEnsemble combines multiple pretrained transformer models
    into one unified prediction using a gating network that learns to weight
    each expert's output.
    """
    def __init__(self, model_names, device):
        super(TransformerEnsemble, self).__init__()
        self.device = device
        self.num_experts = len(model_names)
        
        # Load expert models and their tokenizers.
        self.models = nn.ModuleList([
            AutoModelForSequenceClassification.from_pretrained(name).to(self.device)
            for name in model_names
        ])
        self.tokenizers = [
            AutoTokenizer.from_pretrained(name)
            for name in model_names
        ]
        
        # Assume all models share the same hidden size.
        hidden_size = self.models[0].config.hidden_size
        self.num_labels = self.models[0].config.num_labels
        
        # Define a gating network to combine the [CLS] embeddings from all experts.
        self.gating = nn.Linear(hidden_size * self.num_experts, self.num_experts)

    def forward(self, input_texts):
        """
        Processes a list of input texts and returns:
        - ensemble_logits: weighted combination of expert logits.
        - avg_logits: simple average of expert logits.
        """
        batch_size = len(input_texts)
        expert_logits = []
        cls_embeddings = []
        
        for model, tokenizer in zip(self.models, self.tokenizers):
            encoded = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            outputs = model(**encoded)
            logits = outputs.logits  # (batch_size, num_labels)
            expert_logits.append(logits)
            
            with torch.no_grad():
                base_outputs = model.base_model(**encoded)
                cls_emb = base_outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
                cls_embeddings.append(cls_emb)
        
        # Stack expert logits: (num_experts, batch_size, num_labels)
        stacked_logits = torch.stack(expert_logits, dim=0)
        avg_logits = torch.mean(stacked_logits, dim=0)
        
        # Concatenate CLS embeddings: (batch_size, num_experts * hidden_size)
        concat_cls = torch.cat(cls_embeddings, dim=1)
        gating_logits = self.gating(concat_cls)  # (batch_size, num_experts)
        gating_weights = F.softmax(gating_logits, dim=1)  # (batch_size, num_experts)
        
        # Permute logits: (batch_size, num_experts, num_labels)
        stacked_logits = stacked_logits.permute(1, 0, 2)
        gating_weights = gating_weights.unsqueeze(-1)  # (batch_size, num_experts, 1)
        weighted_logits = stacked_logits * gating_weights
        ensemble_logits = torch.sum(weighted_logits, dim=1)  # (batch_size, num_labels)
        
        return ensemble_logits, avg_logits

# For testing the ensemble module directly.
if __name__ == "__main__":
    model_names = [
        "bert-base-uncased",
        "roberta-base",
        "xlnet-base-cased"
    ]
    ensemble_model = TransformerEnsemble(model_names, device=device)
    ensemble_model.to(device)
    
    input_texts = [
        "This movie was fantastic! I really enjoyed it.",
        "The plot was boring and the acting was mediocre."
    ]
    
    ensemble_logits, avg_logits = ensemble_model(input_texts)
    preds_ensemble = torch.argmax(ensemble_logits, dim=1)
    preds_avg = torch.argmax(avg_logits, dim=1)
    
    print("Gated Ensemble Predictions:", preds_ensemble.cpu().numpy())
    print("Averaged Predictions:", preds_avg.cpu().numpy())
