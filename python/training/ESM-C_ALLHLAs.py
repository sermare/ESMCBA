import numpy as np
import os
import sys
import json
import random
import csv
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from esm.models.esmc import ESMC

#########################################################
# Argument Parsing
#########################################################
parser = argparse.ArgumentParser(description='Fine-tune ESM model with varying parameters.')

parser.add_argument('--name_of_model', type=str, default='ESM-C', help='Name prefix for your model & CSV')
parser.add_argument('--encoding', type=str, default='ESM-C', help='Name prefix for your model & CSV')
parser.add_argument('--blocks_unfrozen', type=int, default=5, help='Unfrozen blocks')
parser.add_argument('--base_block_lr', type=float, default=1e-5, help='LR for transformer blocks')
parser.add_argument('--regression_block_lr', type=float, default=1e-5, help='LR for mask head')
parser.add_argument('--HLA', type=str, default='HLA0201', help='HLA Type')
parser.add_argument('--num_augmentations', type=int, default=1, help='Number of Augmentations')

args = parser.parse_args()
name_of_model = args.name_of_model
encoding = args.encoding
blocks_unfrozen = args.blocks_unfrozen
base_block_lr = args.base_block_lr
regression_block_lr = args.regression_block_lr
HLA = args.HLA
num_augmentations = args.num_augmentations

#########################################################
# Device Check
#########################################################
if not torch.cuda.is_available():
    print("CUDA is not available. Exiting.", flush=True)
    sys.exit(1)

device = torch.device("cuda")
print(f"Using device: {device}", flush=True)

#########################################################
# Model Definition: ESMCMasked
#########################################################
class ESMCMasked(nn.Module):
    """
    A wrapper that takes a pre-trained ESM C model and adds
    a masking (language modeling) head on top of the final hidden states.
    This version expects batched input_ids and attention_mask in forward.
    """
    def __init__(self, base_model, hidden_dim=960, num_aa=33):
        super().__init__()
        self.base_model = base_model  # Pretrained ESM C model
        self.mask_head = nn.Linear(hidden_dim, num_aa)  # Simple linear LM head

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
          input_ids: [batch_size, seq_len] integer tokens
          attention_mask: [batch_size, seq_len], 1 for real tokens, 0 for padding
        Returns:
          out_logits: [batch_size, seq_len, num_aa]
        """
        # 1) ESM forward
        #    If your ESM model supports input_ids directly (like a HuggingFace model),
        #    do something like:
        outputs = self.base_model.forward(input_ids )
        # outputs.hidden_states[-1]: [batch_size, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states[-1].to(torch.float32) # ensure float32

        # 2) Pass through the custom LM head
        out_logits = self.mask_head(hidden_states)  # [batch_size, seq_len, num_aa]
        return out_logits

#########################################################
# Load the Base ESM C Model
#########################################################
print("Loading pretrained ESM_Cambrian model...", flush=True)
base_model = ESMC.from_pretrained("esmc_300m").to(device)

# Create our extended masked model
model_masked = ESMCMasked(base_model, hidden_dim=960, num_aa=33).to(device)

#########################################################
# Unfreeze Last N Blocks
#########################################################
last_block_params = []
total_blocks = 30  # Adjust if your ESM model has a different # of blocks
min_range = total_blocks - blocks_unfrozen
for block_idx in range(min_range, total_blocks):
    last_block_params.extend(
        list(model_masked.base_model.transformer.blocks[block_idx].parameters())
    )
# Also unfreeze final layer norm
norm_params = list(model_masked.base_model.transformer.norm.parameters())
last_block_params.extend(norm_params)

#########################################################
# Optimizer and Loss
#########################################################
optimizer = optim.Adam(
    [
        {"params": last_block_params, "lr": base_block_lr},
        {"params": model_masked.mask_head.parameters(), "lr": regression_block_lr},
    ],
    weight_decay=1e-5
)
criterion = nn.CrossEntropyLoss(ignore_index=model_masked.base_model.tokenizer.pad_token_id)  # We'll use -100 for masked positions

#########################################################
# Standard AA Mappings (for reference/logging)
#########################################################
amino_acids = "ARNDCEQGHILKMFPSTWYV"
aa_to_idx = {
    aa: base_model.tokenizer(text=aa).input_ids[1]  # index=1 to skip <cls> token
    for aa in amino_acids
}
idx_to_aa = {idx: aa for aa, idx in aa_to_idx.items()}
print(f"aa_to_idx: {aa_to_idx}", flush=True)
print(f"idx_to_aa: {idx_to_aa}", flush=True)

#########################################################
# FASTA Parser
#########################################################
def parse_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        header = None
        seq = ""
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

#########################################################
# Load + Filter Data
#########################################################
train_fasta_path = "/global/scratch/users/sergiomar10/jupyter_notebooks/hla_protein_sequences.fasta"
all_data = parse_fasta(train_fasta_path)

hla_and_epitopes = []
for head, hla_seq in all_data:
    # Parse out the HLA naming from header
    head = head.split('|')[1][:7].replace('*', '').replace(':', '')
    head = 'HLA' + head
    if head != HLA:
        continue

    # Load the CSV for that HLA
    iedb_path = f"/global/scratch/users/sergiomar10/data/IEDB_SQL/IEDB_{head}_final.csv"
    df = pd.read_csv(iedb_path, header=None)
    df.columns = [
        'sequence', 'ref_ID', 'submissionID', 'Epitope_ID', 'protein_origin',
        'ID_SOURCE', "SOURCE_ORGANISM", "IC50_nM", "DESCRIPTION_BINDING", "Year_submission"
    ]
    # Filter to "Positive" sequences
    df = df[df['DESCRIPTION_BINDING'].str.contains("Positive")][["ref_ID","sequence"]].values

    for ref_id, epitope in df:
        if any(x in epitope for x in ['+', '(', 'X']):
            continue
        # If your encoding mode = 'HLA', prepend the HLA sequence to epitope
        if encoding == 'HLA':
            epitope = hla_seq + epitope

        hla_and_epitopes.append(epitope)

hla_and_epitopes = np.unique(hla_and_epitopes)

max_length = np.max([len(x) for x in hla_and_epitopes])

random.shuffle(hla_and_epitopes)
print(f"Filtered {len(hla_and_epitopes)} sequences for training.", flush=True)

train_seqs, temp_seqs = train_test_split(hla_and_epitopes, test_size=0.2, random_state=42)
val_seqs, eval_seqs = train_test_split(temp_seqs, test_size=0.5, random_state=42)

print(f"Data split: {len(train_seqs)} train, {len(val_seqs)} val, {len(eval_seqs)} eval.", flush=True)

# Now, apply augmentation only to the training set
augmented_train_seqs = []
for seq in train_seqs:
    for _ in range(num_augmentations):
        augmented_train_seqs.append(seq)

print(f"After augmentation: {len(augmented_train_seqs)} training sequences.", flush=True)


print(
    f"Data split: {len(train_seqs)} train, "
    f"{len(val_seqs)} val, {len(eval_seqs)} eval.",
    flush=True
)

#########################################################
# Masked LM Dataset
#########################################################
class MaskedProteinDataset(Dataset):
    def __init__(self, sequences, base_model, mlm_probability=0.15, max_length=15):
        self.sequences = sequences
        self.tokenizer = base_model.tokenizer
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.pad_id = self.tokenizer.pad_token_id  # e.g. 1
        self.mask_id = self.tokenizer.mask_token_id  # e.g. 32

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # 1) Tokenize the entire sequence up to self.max_length
        encoding = self.tokenizer(
            seq,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        input_ids = encoding['input_ids'].squeeze(0)        # shape [seq_len]
        attention_mask = encoding['attention_mask'].squeeze(0)  # shape [seq_len]

        # 2) Mask only the last 11 real tokens
        masked_input_ids, labels = self.mask_tokens(input_ids)

        # Return everything, along with the raw sequence if needed
        return masked_input_ids, attention_mask, labels, seq

    def mask_tokens(self, input_ids):
        """
        Masks tokens ONLY within the last 11 real (non-pad) positions.
        All other positions remain unmasked.
        """
        # Initialize labels to pad_id (which we'll ignore in the loss)
        labels = torch.full_like(input_ids, self.pad_id)

        # Identify all non-pad token positions
        nonpad_positions = (input_ids != self.pad_id).nonzero(as_tuple=True)[0]
        if len(nonpad_positions) == 0:
            # Edge case: if there's nothing but padding, just return as-is
            return input_ids, labels

        # We'll only allow masking within the last 11 real tokens
        # e.g., if we have 15 real tokens, we choose positions [-11:].
        # if we have fewer than 11 real tokens, then it's effectively "mask up to length"
        maskable_positions = nonpad_positions[-11:]  # slice last 11 indices

        # Create a probability vector of 0 for all tokens, except for these last 11 real ones
        probs = torch.zeros_like(input_ids, dtype=torch.float)
        probs[maskable_positions] = self.mlm_probability  # mlm_probability only for last 11

        # Decide which of those positions to actually mask
        masked_indices = torch.bernoulli(probs).bool()

        # Copy the original token IDs into 'labels' only where we do mask
        labels[masked_indices] = input_ids[masked_indices]
        # Replace masked positions in input_ids with <mask> 
        input_ids[masked_indices] = self.mask_id

        return input_ids, labels


def collate_fn(batch):
    """
    batch is a list of tuples:
        (masked_input_ids, attention_mask, labels, raw_sequence)
    """
    input_ids_list, attn_masks_list, labels_list, raw_seqs_list = zip(*batch)

    input_ids = torch.stack(input_ids_list, dim=0)
    attention_mask = torch.stack(attn_masks_list, dim=0)
    labels = torch.stack(labels_list, dim=0)

    # raw_seqs_list is a tuple of strings (the raw epitopes), so just keep it as a list
    return input_ids, attention_mask, labels, list(raw_seqs_list)


def get_mlm_dataloader(sequences, base_model, batch_size=8, shuffle=True, max_length=15):
    dataset = MaskedProteinDataset(
        sequences,
        base_model,
        mlm_probability=0.15,
        max_length=max_length
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return loader

#########################################################
# Create DataLoaders
#########################################################

base_model = model_masked.base_model

batch_size = 8
train_loader = get_mlm_dataloader(train_seqs, base_model, batch_size=batch_size, shuffle=True, max_length=max_length)
val_loader   = get_mlm_dataloader(val_seqs,   base_model, batch_size=batch_size, shuffle=False, max_length=max_length)
eval_loader  = get_mlm_dataloader(eval_seqs,  base_model, batch_size=batch_size, shuffle=False, max_length=max_length)

#########################################################
# Training and Validation Loops
#########################################################
num_epochs = 10
save_dir = "/global/scratch/users/sergiomar10/logs/ESMC_Pretrain_logs"
os.makedirs(save_dir, exist_ok=True)
log_file = os.path.join(save_dir, f"training_log_{name_of_model}.csv")

# Simple function to measure MLM accuracy on masked positions
def evaluate_mlm_accuracy(loader):
    """Compute how often the model guesses the correct token for the masked tokens."""
    model_masked.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels, _ in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward
            logits = model_masked(input_ids, attention_mask)  # [batch_size, seq_len, vocab_size]
            # We only compare at positions where labels != -100
            mask_positions = (labels != 1)
            if not mask_positions.any():
                continue

            # Predictions
            preds = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
            correct += (preds[mask_positions] == labels[mask_positions]).sum().item()
            total += mask_positions.sum().item()

    model_masked.train()
    return correct / total if total > 0 else 0.0

with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Loss", "Train Acc", "Val Acc"])

    for epoch in range(num_epochs):
        model_masked.train()
        total_loss = 0.0

        for input_ids, attention_mask, labels, raw_epitopes in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward
            logits = model_masked(input_ids, attention_mask)  # [batch_size, seq_len, vocab_size]

            # CrossEntropyLoss expects shape [batch_size * seq_len, vocab_size]
            # and labels: [batch_size * seq_len]
            logits_2d = logits.view(-1, logits.size(-1))
            labels_1d = labels.view(-1)

            loss = criterion(logits_2d, labels_1d)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_acc = evaluate_mlm_accuracy(train_loader)
        val_acc   = evaluate_mlm_accuracy(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Val Acc: {val_acc*100:.2f}%",
            flush=True
        )
        writer.writerow([epoch+1, f"{avg_loss:.4f}", f"{train_acc:.4f}", f"{val_acc:.4f}"])

print(f"Training log saved at {log_file}.", flush=True)

#########################################################
# Evaluation: Save predictions on the eval set
#########################################################
model_masked.eval()
eval_results = []

model_masked.eval()
eval_results = []

with torch.no_grad():
    for input_ids, attention_mask, labels, raw_epitopes in eval_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model_masked(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        for b in range(input_ids.size(0)):
            # Grab the actual epitope string for this sample
            epitope_seq = raw_epitopes[b]

            # Identify masked positions
            masked_positions = (labels[b] != 1)  # or != pad_id

            for pos in torch.where(masked_positions)[0]:
                pos = pos.item()
                original_id = labels[b, pos].item()
                pred_id = preds[b, pos].item()
                pred_prob = probs[b, pos, pred_id].item()

                original_aa = base_model.tokenizer.decode([original_id]).strip()
                predicted_aa = base_model.tokenizer.decode([pred_id]).strip()

                eval_results.append({
                    "batch_index": b,
                    "epitope": epitope_seq,         # Store the entire raw epitope
                    "position": pos,
                    "original_aa": original_aa,
                    "predicted_aa": predicted_aa,
                    "predicted_prob": pred_prob
                })


eval_df = pd.DataFrame(eval_results)
eval_save_dir = "/global/scratch/users/sergiomar10/data/ESMC_Pretrain"
os.makedirs(eval_save_dir, exist_ok=True)
eval_csv_path = os.path.join(eval_save_dir, f"{name_of_model}.csv")
eval_df.to_csv(eval_csv_path, index=False)
print(f"Evaluation predictions saved to {eval_csv_path}", flush=True)

if val_acc > 0.20:
    
    #########################################################
    # Saving the Model
    #########################################################
    HLA_folder = HLA.replace("*", "").replace(":", "")
    model_dir = f'/global/scratch/users/sergiomar10/models/ESMC_Pretrain/HLA{HLA_folder}/'
    os.makedirs(model_dir, exist_ok=True)
    
    model_save_path = os.path.join(model_dir, f"{name_of_model}.pt")
    config_save_path = os.path.join(model_dir, f"{name_of_model}.json")
    
    model_to_save = {
        "model_state_dict": model_masked.state_dict(),
        "config": {
            "hidden_dim": 960,
            "num_aa": 33,
            "model_type": "ESMCMasked"
        }
    }
    
    torch.save(model_to_save, model_save_path)
    with open(config_save_path, "w") as f:
        json.dump(model_to_save["config"], f)
    
    print(f"Trained model saved to {model_save_path}")
    print(f"Configuration saved to {config_save_path}")
