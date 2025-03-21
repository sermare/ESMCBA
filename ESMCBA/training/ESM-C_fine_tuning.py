import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import json
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Fine-tune ESM model with varying parameters.')
parser.add_argument('--full_pretraining', type=str, default='False', help='True/False Full pretraining')
parser.add_argument('--name_of_model', type=str, default='ESM-C_evaluation_predictions_newconfig_FP.csv', help='Name of Model')
parser.add_argument('--num_seq', type=int, default=100, help='Number of Sequences')  # Fixed type to int

args = parser.parse_args()
full_pretraining = args.full_pretraining.lower() == 'true'  # Simplified boolean conversion
name_of_model = args.name_of_model
num_seq = args.num_seq

# --- Device setup ---
if not torch.cuda.is_available():
    print("CUDA is not available. Exiting.")
    sys.exit(1)
device = torch.device("cuda")
print(f"Using device: {device}", flush = True)

# --- 1. Model Definition ---
class ESMCMasked(nn.Module):
    def __init__(self, base_model, hidden_dim=960, num_aa=33):
        super().__init__()
        self.base_model = base_model
        self.mask_head = nn.Linear(hidden_dim, num_aa)
        
        # Freeze base model initially
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, mask_positions=None):
        # Directly use base model's forward
        x = self.base_model(input_ids)["logits"]
        logits = self.mask_head(x)
        return logits[:, mask_positions, :] if mask_positions is not None else logits

# --- 2. Freezing Logic ---
def configure_parameter_groups(model, full_pretraining):
    params = []
    # Always train mask head
    params += list(model.mask_head.parameters())
    
    if full_pretraining:
        # Unfreeze entire base model
        for param in model.base_model.parameters():
            param.requires_grad = True
        params += list(model.base_model.parameters())
    else:
        # Only unfreeze last transformer block
        last_block = model.base_model.transformer.blocks[-1]
        for param in last_block.parameters():
            param.requires_grad = True
        params += list(last_block.parameters())
    
    return params

# --- 3. Data Loading & Processing ---
class ProteinDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_len=2048, mask_prob=0.15):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_id = tokenizer.vocab["<mask>"]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        header, seq = self.sequences[idx]
        tokens = self.tokenizer.encode(seq)[:self.max_len]
        tokens = torch.tensor(tokens)
        
        # Create masked version
        masked_tokens = tokens.clone()
        mask_pos = torch.rand(len(tokens)) < self.mask_prob
        mask_pos[0] = mask_pos[-1] = False  # Never mask first/last token
        masked_tokens[mask_pos] = self.mask_id
        
        return {
            "input_ids": masked_tokens,
            "labels": tokens,
            "mask_positions": mask_pos.nonzero(as_tuple=True)[0]
        }

def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    return {
        "input_ids": torch.stack([F.pad(item["input_ids"], (0, max_len - len(item["input_ids"]))) for item in batch]),
        "labels": torch.stack([F.pad(item["labels"], (0, max_len - len(item["labels"]))) for item in batch]),
        "mask_positions": [item["mask_positions"] for item in batch]
    }

# --- 4. Training Loop (Updated with Accuracy) ---
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_masked = 0
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids)
        
        batch_loss = 0
        batch_correct = 0
        batch_masked = 0
        
        for i, pos in enumerate(batch["mask_positions"]):
            if len(pos) == 0:
                continue
                
            # Calculate loss
            loss = F.cross_entropy(logits[i, pos], labels[i, pos])
            batch_loss += loss.item()
            
            # Calculate accuracy
            preds = torch.argmax(logits[i, pos], dim=-1)
            batch_correct += (preds == labels[i, pos]).sum().item()
            batch_masked += len(pos)
        
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss
        total_correct += batch_correct
        total_masked += batch_masked
    
    return (
        total_loss / len(loader),
        total_correct / total_masked if total_masked > 0 else 0
    )

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_masked = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids)
            
            for i, pos in enumerate(batch["mask_positions"]):
                if len(pos) == 0:
                    continue
                    
                # Calculate loss
                loss = F.cross_entropy(logits[i, pos], labels[i, pos])
                total_loss += loss.item()
                
                # Calculate accuracy
                preds = torch.argmax(logits[i, pos], dim=-1)
                total_correct += (preds == labels[i, pos]).sum().item()
                total_masked += len(pos)
    
    return (
        total_loss / len(loader),
        total_correct / total_masked if total_masked > 0 else 0
    )

# --- 5. Main Execution ---
if __name__ == "__main__":
    # Load model
    base_model = ESMC.from_pretrained("esmc_300m").to(device)
    model = ESMCMasked(base_model).to(device)
    
    # Configure parameters
    params = configure_parameter_groups(model, full_pretraining)
    optimizer = optim.Adam(params, lr=1e-4)
    
    # Load data
    all_data = parse_fasta("/global/scratch/users/sergiomar10/data/UP000005640_9606.fasta")
    filtered_data = [seq for seq in all_data 
                    if seq[1].startswith('M') 
                    and len(seq[1]) >= 50 
                    and 'X' not in seq[1]][:num_seq]
    
    # Split data
    train_data, val_data = train_test_split(filtered_data, test_size=0.1, random_state=42)
    
    # Create datasets
    train_dataset = ProteinDataset(train_data, base_model.tokenizer)
    val_dataset = ProteinDataset(val_data, base_model.tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)
    
    for epoch in range(10):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, device)
            
            print(f"Epoch {epoch+1}")
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")

    # Save model
    torch.save(model.state_dict(), f"/global/scratch/users/sergiomar10/models/esm_c/fine_tuned/{name_of_model}.pt")