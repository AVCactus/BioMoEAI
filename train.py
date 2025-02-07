# src/train.py

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import the ensemble model from our src folder.
from ensemble import TransformerEnsemble

# Set a fixed random seed for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# -------------------------
# Configuration and Hyperparameters
# -------------------------
BATCH_SIZE = 8                  # Batch size.
EPOCHS = 3                      # Number of epochs.
LEARNING_RATE = 5e-5            # Learning rate.
MAX_SEQ_LENGTH = 128            # Maximum tokens per example.
LOGGING_STEPS = 10              # Log every 10 steps.
SAVE_STEPS = 50                 # Save a checkpoint every 50 steps.
OUTPUT_DIR = "checkpoints"      # Checkpoint directory.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load Dataset and Preprocess Data
# -------------------------
# Using the SST-2 dataset from the GLUE benchmark.
dataset = load_dataset("glue", "sst2")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# Use the tokenizer from the first expert for preprocessing.
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(example):
    result = tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
    result["label"] = example["label"]
    return result

train_dataset = train_dataset.map(preprocess, batched=False)
val_dataset = val_dataset.map(preprocess, batched=False)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -------------------------
# Create DataLoaders
# -------------------------
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# Initialize the Ensemble Model
# -------------------------
expert_model_names = [
    "bert-base-uncased",
    "roberta-base",
    "xlnet-base-cased"
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ensemble_model = TransformerEnsemble(expert_model_names, device=device)
ensemble_model.to(device)

# -------------------------
# Define Optimizer, Scheduler, and Loss Function
# -------------------------
optimizer = optim.AdamW(ensemble_model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss()

# -------------------------
# Set Up TensorBoard Logging
# -------------------------
writer = SummaryWriter(log_dir="runs/ensemble_experiment_" + time.strftime("%Y%m%d-%H%M%S"))

# -------------------------
# Training Loop
# -------------------------
def train():
    global_step = 0
    ensemble_model.train()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        for step, batch in enumerate(train_loader):
            # Convert tokenized inputs back to text for our ensemble.
            input_ids = batch["input_ids"]
            input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            ensemble_logits, avg_logits = ensemble_model(input_texts)
            labels = batch["label"].to(device)
            loss = criterion(ensemble_logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % LOGGING_STEPS == 0:
                writer.add_scalar("Loss/train", loss.item(), global_step)
                print(f"Step {global_step}/{total_steps} - Loss: {loss.item():.4f}")

            if global_step % SAVE_STEPS == 0:
                checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint-step-{global_step}.pt")
                torch.save(ensemble_model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
        evaluate(epoch)

def evaluate(epoch):
    ensemble_model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"]
            input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            ensemble_logits, _ = ensemble_model(input_texts)
            labels = batch["label"].to(device)
            loss = criterion(ensemble_logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(ensemble_logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    writer.add_scalar("Loss/validation", avg_loss, epoch)
    writer.add_scalar("Accuracy/validation", accuracy, epoch)
    print(f"Validation - Epoch {epoch + 1}: Loss {avg_loss:.4f}, Accuracy {accuracy:.4f}")
    ensemble_model.train()

if __name__ == "__main__":
    print("Starting training...")
    train()
    print("Training completed. Closing TensorBoard writer.")
    writer.close()
