import os
import sys
import random
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, pearsonr
import argparse
from collections import Counter

# Argument parser for command-line configuration
parser = argparse.ArgumentParser(description='Fine-tune ESM-Cambrian model for binary classification.')
parser.add_argument('--name_of_model', type=str, default='ESM-C_binary_classifier', help='Name prefix for model and CSV outputs')
parser.add_argument('--encoding', type=str, default='HLA', help='Encoding type: HLA vs epitope')
parser.add_argument('--file_path', type=str, default='here', help='Path to pre-trained model')
parser.add_argument('--HLA', type=str, default='HLA0201', help='HLA type (e.g., HLA0201)')
parser.add_argument('--train_size', type=float, default=1.0, help='Proportion of data to use for training')
parser.add_argument('--blocks_unfrozen', type=int, default=10, help='Number of transformer blocks to unfreeze')
parser.add_argument('--base_block_lr', type=float, default=1e-5, help='Learning rate for transformer blocks')
parser.add_argument('--classifier_lr', type=float, default=1e-5, help='Learning rate for classification head')

args = parser.parse_args()
name_of_model = f"{args.name_of_model}_BCE_{args.HLA}"
encoding = args.encoding
file_path = args.file_path
size_of_train = args.train_size
blocks_unfrozen = args.blocks_unfrozen
base_block_lr = args.base_block_lr
classifier_lr = args.classifier_lr
HLA = args.HLA

# Check for CUDA availability
if not torch.cuda.is_available():
    print("CUDA is not available. Exiting.")
    sys.exit(1)
device = torch.device("cuda")
print(f"Using device: {device}", flush=True)

# Display configuration
print(f"HLA: {HLA}")
print(f"Unfrozen Blocks: {blocks_unfrozen}")
print(f"Encoding: {encoding}")
print(f"Training Data Proportion: {size_of_train}")
print(f"Filepath: {file_path}")

class ESMBA(nn.Module):
    def __init__(self, base_model, dropout=0.3):
        super(ESMBA, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.classifier_head = nn.Linear(960, 1)
        nn.init.xavier_uniform_(self.classifier_head.weight, gain=0.01)
        nn.init.uniform_(self.classifier_head.bias, a=0.0, b=1.0)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model.forward(input_ids)
        hidden_states = outputs.hidden_states[-1].to(torch.float32)
        
        mask = attention_mask.unsqueeze(-1).float()
        masked_hidden_states = hidden_states * mask
        sum_embeddings = masked_hidden_states.sum(dim=1)
        sum_mask = mask.sum(dim=1)
        
        pooled_output = sum_embeddings / (sum_mask + 1e-8)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier_head(pooled_output).squeeze(-1)
        probabilities = torch.sigmoid(logits) 
        return probabilities

# Loss function
criterion = nn.BCELoss()

def load_model(model_path, device='cuda'):
    """Load a pre-trained model from a checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    base_model = ESMC.from_pretrained("esmc_300m").to(device)
    model = ESMBA(base_model).to(device)
    
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                 if not k.startswith('mask_head')}
    model.load_state_dict(state_dict, strict=False)
    return model

# Load the model
model = load_model(file_path, device=device)
base_model = model.base_model

# Amino acid to token ID mapping
amino_acids = "ARNDCEQGHILKMFPSTWYV"
aa_to_idx = {aa: base_model.tokenizer(text=aa).input_ids[1] for aa in amino_acids}
idx_to_aa = {idx: aa for aa, idx in aa_to_idx.items()}

# Optimizer setup
last_10_block_params = []
min_range = 30 - blocks_unfrozen
for block_idx in range(min_range, 30):
    last_10_block_params.extend(list(model.base_model.transformer.blocks[block_idx].parameters()))
last_10_block_params.extend(list(model.base_model.transformer.norm.parameters()))

optimizer = optim.Adam([
    {"params": last_10_block_params, "lr": base_block_lr},
    {"params": model.classifier_head.parameters(), "lr": classifier_lr},
], weight_decay=1e-5)

# FASTA parser
def parse_fasta(file_path):
    """Parse sequences from a FASTA file."""
    sequences = []
    with open(file_path, 'r') as f:
        header, seq = None, ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    sequences.append((header, seq))
                    seq = ""
                header = line[1:]
            else:
                seq += line
        if seq:
            sequences.append((header, seq))
    return sequences

# Load and preprocess data
train_fasta = f"/global/scratch/users/sergiomar10/data/IEDB_SQL/IEDB_HLA{HLA}_final.csv"
epitope_data = pd.read_csv(train_fasta, header=None)
epitope_data.columns = ['sequence', 'ref_ID', 'submissionID', 'Epitope_ID', 'protein_origin', 
                      'ID_SOURCE', "SOURCE_ORGANISM", "IC50_nM", "DESCRIPTION_BINDING", "Year_submission"]

train_fasta_path = "/global/scratch/users/sergiomar10/jupyter_notebooks/hla_protein_sequences.fasta"
all_data = parse_fasta(train_fasta_path)
for head, hla_seq in all_data:
    head = head.split('|')[1][:7].replace('*', '').replace(':', '')
    if head == HLA:
        HLA_seq = hla_seq

# Convert IC50 to binary labels (threshold at 500 nM)
epitope_data['IC50_nM'] = epitope_data['IC50_nM'].astype(str).replace('\\N', '0').astype(float)
epitope_data['label'] = (epitope_data['IC50_nM'] <= 500).astype(int)  # 1 for binding, 0 for non-binding
epitope_data['Year_submission'] = epitope_data['Year_submission'].astype(str).replace('\\N', '0').astype(int)
epitope_data = epitope_data[["label", "sequence", "Year_submission"]].values

# Filter sequences
filtered_data = []
for label, sequence, year_submission in epitope_data:
    if '+' in sequence or '(' in sequence or 'X' in sequence:
        continue
    if 'epitope' not in encoding:
        sequence = HLA_seq + sequence
    filtered_data.append((label, sequence, year_submission))

print(f"Filtered {len(filtered_data)} sequences for training.", flush=True)
aggregated = pd.DataFrame(filtered_data, columns=['label', 'sequence', 'testing'])

def split_data(aggregated, size_of_train=1.0):
    """Split data into train, validation (both <= 2021), and evaluation (> 2021) sets.
    
    Args:
        aggregated (pd.DataFrame): Dataframe with labels, sequences, and years
        size_of_train (float): Proportion of data for training
        
    Returns:
        tuple: Train, validation, and evaluation dataframes
    """
    # Data <= 2021 for training and testing
    pre_2021_data = aggregated[aggregated['testing'] <= 2021]
    total_samples = int(size_of_train * len(pre_2021_data))
    train_val_data = pre_2021_data.sample(n=total_samples)
    
    # Split into train and validation (test set within <= 2021)
    train_data, val_data = train_test_split(train_val_data, test_size=0.1, shuffle=True)
    
    # Evaluation set is > 2021
    eval_data = aggregated[aggregated['testing'] > 2021]
    
    if len(eval_data) < 10:
        print("Warning: Less than 10 samples for evaluation (> 2021). Adding remaining pre-2021 data.", flush=True)
        remaining_data = aggregated[(aggregated['testing'] <= 2021) & 
                                  (~aggregated["sequence"].isin(train_val_data["sequence"]))]
        eval_data = pd.concat([remaining_data, eval_data])
    
    return (train_data[['sequence', 'label']], 
            val_data[['sequence', 'label']], 
            eval_data[['sequence', 'label']])

# Dataset class
class EpitopeDataset(Dataset):
    def __init__(self, sequences, labels, base_model):
        self.sequences = sequences
        self.labels = labels
        self.base_model = base_model

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        protein_obj = ESMProtein(sequence=seq)
        protein_tensor = self.base_model.encode(protein_obj)
        return seq, protein_tensor, torch.tensor(self.labels[idx], dtype=torch.float)

def collate_fn(batch):
    sequences, protein_tensors, labels = zip(*batch)
    try:
        protein_tensors = torch.stack(protein_tensors)
    except Exception:
        protein_tensors = list(protein_tensors)
    labels = torch.stack(labels)
    return list(sequences), protein_tensors, labels

def prepare_dataloaders(dataframe, batch_size=10, size_of_train=1):
    """Prepare DataLoader objects for training, validation, and evaluation."""
    train_data, val_data, eval_data = split_data(dataframe, size_of_train)
    train_dataset = EpitopeDataset(train_data["sequence"].values, train_data["label"].values, base_model)
    val_dataset = EpitopeDataset(val_data["sequence"].values, val_data["label"].values, base_model)
    eval_dataset = EpitopeDataset(eval_data["sequence"].values, eval_data["label"].values, base_model)
    
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
            DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
            train_data, val_data, eval_data)

# Prepare data
train_loader, val_loader, eval_loader, train_data, val_data, eval_data = prepare_dataloaders(
    aggregated, batch_size=10, size_of_train=size_of_train)

# Training loop
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0.0
    total_train_samples = 0
    train_predictions, train_targets = [], []
    
    for sequences, protein_tensors, targets in train_loader:
        targets = targets.to(device)
        encoded_seq = base_model.tokenizer(sequences, return_tensors='pt', padding=True).input_ids.to(device)
        attention_mask = base_model.tokenizer(sequences, return_tensors='pt', padding=True).attention_mask.to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids=encoded_seq, attention_mask=attention_mask)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item() * targets.size(0)
        total_train_samples += targets.size(0)
        train_predictions.extend(torch.sigmoid(logits).detach().cpu().numpy())
        train_targets.extend(targets.cpu().numpy())
    
    avg_train_loss = total_train_loss / total_train_samples
    train_spearman, _ = spearmanr(train_targets, train_predictions)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f}, Train Spearman: {train_spearman:.4f}", flush=True)
    
    # Validation (testing) on <= 2021 data
    model.eval()
    val_sequences, val_predictions, val_targets = [], [], []
    with torch.no_grad():
        for sequences, protein_tensors, targets in val_loader:
            targets = targets.to(device)
            encoded_seq = base_model.tokenizer(sequences, return_tensors='pt', padding=True).input_ids.to(device)
            attention_mask = base_model.tokenizer(sequences, return_tensors='pt', padding=True).attention_mask.to(device)
            logits = model(input_ids=encoded_seq, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)
            
            for i, sequence in enumerate(sequences):
                val_sequences.append(sequence)
                val_predictions.append(probs[i].cpu().numpy().item())
                val_targets.append(targets[i].cpu().numpy().item())
    
    val_spearman, _ = spearmanr(val_targets, val_predictions)
    val_pearson, _ = pearsonr(val_targets, val_predictions)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Validation (<= 2021) Spearman: {val_spearman:.4f}, Pearson: {val_pearson:.4f}", flush=True)
    
    # Evaluation on > 2021 data
    eval_sequences, eval_predictions, eval_targets = [], [], []
    with torch.no_grad():
        for sequences, protein_tensors, targets in eval_loader:
            targets = targets.to(device)
            encoded_seq = base_model.tokenizer(sequences, return_tensors='pt', padding=True).input_ids.to(device)
            attention_mask = base_model.tokenizer(sequences, return_tensors='pt', padding=True).attention_mask.to(device)
            logits = model(input_ids=encoded_seq, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)
            
            for i, sequence in enumerate(sequences):
                eval_sequences.append(sequence)
                eval_predictions.append(probs[i].cpu().numpy().item())
                eval_targets.append(targets[i].cpu().numpy().item())
    
    eval_spearman, _ = spearmanr(eval_targets, eval_predictions)
    eval_pearson, _ = pearsonr(eval_targets, eval_predictions)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Evaluation (> 2021) Spearman: {eval_spearman:.4f}, Pearson: {eval_pearson:.4f}", flush=True)

# Save results
eval_df = pd.DataFrame({'sequence': eval_sequences, 'prediction': eval_predictions, 'measured': eval_targets})
val_df = pd.DataFrame({'sequence': val_sequences, 'prediction': val_predictions, 'measured': val_targets})

loss_dir = f'/global/scratch/users/sergiomar10/losses/ESMCBA_CLASSIFIER/'
os.makedirs(loss_dir, exist_ok=True)
eval_df.to_csv(os.path.join(loss_dir, f'evaluation_{name_of_model}.csv'), index=False)
val_df.to_csv(os.path.join(loss_dir, f'validation_{name_of_model}.csv'), index=False)
print(f"Saved evaluation to {os.path.join(loss_dir, f'evaluation_{name_of_model}.csv')}")
print(f"Saved validation to {os.path.join(loss_dir, f'validation_{name_of_model}.csv')}")

if eval_spearman > 0.30:
    training_dir = os.path.join(loss_dir, 'training_data')
    os.makedirs(training_dir, exist_ok=True)
    eval_df.to_csv(os.path.join(training_dir, f'training_eval_{name_of_model}.csv'), index=False)
    
    HLA_folder = HLA.replace("*", "").replace(":", "")
    model_dir = f'/global/scratch/users/sergiomar10/models/ESMCBA_CLASSIFIER/{HLA_folder}/'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, f"training_{name_of_model}_final.pth"))
    print(f"Saved model to {os.path.join(model_dir, f'training_{name_of_model}_final.pth')}")