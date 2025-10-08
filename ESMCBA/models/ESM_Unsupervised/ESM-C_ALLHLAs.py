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
parser.add_argument('--HLA', type=str, default='HLAA0201', help='HLA Type')
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
        #    If your ESM model supports input_ids and attention_mask directly:
        outputs = self.base_model(
            input_ids,
            # attention_mask=attention_mask  ### FIX: pass attention_mask if supported
        )
        
        # outputs.hidden_states[-1]: [batch_size, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states[-1].to(torch.float32)

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

# Note: We ignore pad_token_id in the loss so only masked tokens are trained
criterion = nn.CrossEntropyLoss(ignore_index=model_masked.base_model.tokenizer.pad_token_id)

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
# Load HLA Sequences
#########################################################
fasta_path = "/global/scratch/users/sergiomar10/ESMCBA/ESMCBA/jupyter_notebooks/other/hla_sequences.fasta"
hla_data = parse_fasta(fasta_path)

hla_sequences = [
    seq for header, seq in hla_data
    if header.replace(':','').replace('*','') == HLA.replace("HLA", "")
][0]

print(f"Loaded {len(hla_sequences)} HLA sequences from {fasta_path}", flush=True)

IEDB_full = pd.read_csv('/global/scratch/users/sergiomar10/ESMCBA/IEDB_subseted_HLA_w_BA_full_for_training.csv',
                        sep=',', header=0)
                        
IEDB_full_subset = IEDB_full[IEDB_full['MHC Restriction - Name'] == HLA]
IEDB_full_subset = IEDB_full_subset[IEDB_full_subset['Assay - Qualitative Measurement'].str.contains('Positive')]

peptides = IEDB_full_subset['Epitope - Name'].unique()

hla_and_epitopes = []
for epitope in peptides:
    if any(x in epitope for x in ['+', '(', 'X']):
        continue
    if encoding == 'HLA':
        combined = hla_sequences + epitope
        max_length_hla = len(hla_sequences)
    else:
        combined = epitope
        max_length_hla = 0
    hla_and_epitopes.append(combined)

max_length = np.max([len(x) for x in hla_and_epitopes])

print(f"Filtered {len(hla_and_epitopes)} sequences for training.", flush=True)

train_seqs, temp_seqs = train_test_split(hla_and_epitopes, test_size=0.2, random_state=42)
val_seqs, eval_seqs = train_test_split(temp_seqs, test_size=0.5, random_state=42)

print(f"Data split: {len(train_seqs)} train, {len(val_seqs)} val, {len(eval_seqs)} eval.", flush=True)
#########################################################
# Masked LM Dataset
#########################################################
class MaskedProteinDataset(Dataset):
    def __init__(self, sequences, base_model, mlm_probability=0.15,
                 max_length=15, max_length_hla=350):
        self.sequences = sequences
        self.tokenizer = base_model.tokenizer
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.max_length_hla = max_length_hla
        self.pad_id = self.tokenizer.pad_token_id
        self.mask_id = self.tokenizer.mask_token_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoding = self.tokenizer(
            seq,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Mask only the last `self.max_length_hla` real (non-pad) tokens
        masked_input_ids, labels = self.mask_tokens(input_ids)
        return masked_input_ids, attention_mask, labels, seq

    def mask_tokens(self, input_ids, seed=None):
        """Mask tokens in the last `self.max_length_hla` positions only."""
        labels = torch.full_like(input_ids, self.pad_id)
        
        # Identify which positions are real (non-pad)
        nonpad_positions = (input_ids != self.pad_id).nonzero(as_tuple=True)[0]
        if len(nonpad_positions) == 0:
            return input_ids, labels

        # ### CHANGED ###
        # Restrict masking to the last `self.max_length_hla` positions:
        maskable_positions = nonpad_positions[self.max_length_hla:]

        # Build a probability vector of 0 everywhere except these last positions
        probs = torch.zeros_like(input_ids, dtype=torch.float)
        probs[maskable_positions] = self.mlm_probability
        
        # Sample Bernoulli to decide which tokens to mask
        masked_indices = torch.bernoulli(probs).bool()
        
        # Record the original token in `labels`
        labels[masked_indices] = input_ids[masked_indices]
        # Replace input_ids with [MASK] token
        input_ids[masked_indices] = self.mask_id

        return input_ids, labels


def collate_fn(batch):
    input_ids_list, attn_masks_list, labels_list, raw_seqs_list = zip(*batch)
    input_ids = torch.stack(input_ids_list, dim=0)
    attention_mask = torch.stack(attn_masks_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    return input_ids, attention_mask, labels, list(raw_seqs_list)

def get_mlm_dataloader(sequences, base_model, batch_size=8, shuffle=False,
                       max_length=15, max_length_hla=11):
    dataset = MaskedProteinDataset(
        sequences,
        base_model,
        mlm_probability=0.15,
        max_length=max_length,
        max_length_hla=max_length_hla
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

# Repeat train_seqs for data augmentation
train_seqs = train_seqs * num_augmentations

batch_size = 8
train_loader = get_mlm_dataloader(
    train_seqs, base_model, batch_size=batch_size,
    shuffle=True, max_length=max_length, max_length_hla=max_length_hla
)
val_loader = get_mlm_dataloader(
    val_seqs, base_model, batch_size=batch_size,
    shuffle=False, max_length=max_length, max_length_hla=max_length_hla
)
eval_loader = get_mlm_dataloader(
    eval_seqs, base_model, batch_size=batch_size,
    shuffle=False, max_length=max_length, max_length_hla=max_length_hla
)

#########################################################
# Training and Validation Loops
#########################################################
num_epochs = 10
save_dir = "/global/scratch/users/sergiomar10/logs/ESMC_Pretrain_logs_09042025"
os.makedirs(save_dir, exist_ok=True)
log_file = os.path.join(save_dir, f"training_log_{name_of_model}.csv")


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

            logits = model_masked(input_ids, attention_mask)
            mask_positions = (labels != base_model.tokenizer.pad_token_id)
            if not mask_positions.any():
                continue

            preds = torch.argmax(logits, dim=-1)
            correct += (preds[mask_positions] == labels[mask_positions]).sum().item()
            total += mask_positions.sum().item()

    model_masked.train()
    return correct / total if total > 0 else 0.0


print("Starting training...", flush=True)
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

            logits = model_masked(input_ids, attention_mask)

            logits_2d = logits.view(-1, logits.size(-1))
            labels_1d = labels.view(-1)

            loss = criterion(logits_2d, labels_1d)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_acc = evaluate_mlm_accuracy(train_loader)
        val_acc = evaluate_mlm_accuracy(val_loader)

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
# Evaluation: Compute pseudo-perplexity on the eval set
#########################################################
eval_results = []
per_token_results = []
all_token_nlls = []

model_masked.eval()
with torch.no_grad():
    for input_ids, attention_mask, labels, raw_epitopes in eval_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model_masked(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)

        for b in range(input_ids.size(0)):
            epitope_seq = raw_epitopes[b]
            mask = (labels[b] != base_model.tokenizer.pad_token_id)
            masked_positions = torch.where(mask)[0]
            num_masked = int(mask.sum().item())

            if num_masked > 0:
                ground_truth_ids = labels[b][mask]
                masked_probs = probs[b][mask]
                true_token_probs = masked_probs.gather(1, ground_truth_ids.unsqueeze(1)).squeeze(1)
                eps = 1e-12
                true_token_probs = torch.clamp(true_token_probs, min=eps)
                nll = -torch.log(true_token_probs)
                avg_nll = nll.mean()
                pseudo_perplexity = torch.exp(avg_nll).item()
                token_nll_list = nll.cpu().tolist()
                all_token_nlls.extend(token_nll_list)
            else:
                pseudo_perplexity = float('nan')
                avg_nll = float('nan')
                token_nll_list = []

            eval_results.append({
                "epitope": epitope_seq,
                "pseudo_perplexity": pseudo_perplexity,
                "num_masked": num_masked,
                "avg_token_nll": float(avg_nll) if torch.is_tensor(avg_nll) else avg_nll
            })

            for pos in masked_positions:
                pos = pos.item()
                original_id = labels[b, pos].item()
                pred_id = torch.argmax(probs[b, pos]).item()
                pred_prob = probs[b, pos, pred_id].item()
                token_nll = -torch.log(probs[b, pos, original_id]).item()
                per_token_results.append({
                    "epitope": epitope_seq,
                    "position": pos,
                    "original_id": original_id,
                    "predicted_id": pred_id,
                    "predicted_prob": pred_prob,
                    "token_nll": token_nll
                })

eval_save_dir = "/global/scratch/users/sergiomar10/data/ESMC_Pretrain_09042025_evals"
os.makedirs(eval_save_dir, exist_ok=True)
eval_csv_path = os.path.join(eval_save_dir, f"{name_of_model}_perplexity.csv")
df_eval = pd.DataFrame(eval_results)
df_eval['epitope'] = df_eval['epitope'].apply(lambda x: x[max_length_hla:])
df_eval.to_csv(eval_csv_path, index=False)
print(f"Evaluation summary saved to {eval_csv_path}", flush=True)

eval_save_dir = "/global/scratch/users/sergiomar10/data/ESMC_Pretrain_09042025_perplexity"
os.makedirs(eval_save_dir, exist_ok=True)
per_token_csv_path = os.path.join(eval_save_dir, f"{name_of_model}_per_token_metrics.csv")
per_token_results_df = pd.DataFrame(per_token_results)
per_token_results_df['epitope'] = per_token_results_df['epitope'].apply(lambda x: x[max_length_hla:])
per_token_results_df.to_csv(per_token_csv_path, index=False)
print(f"Per-token evaluation metrics saved to {per_token_csv_path}", flush=True)

### FIX: Use len(eval_seqs) instead of eval_seqs < 10
if len(eval_seqs) > 10:
    if len(eval_seqs) > 0:
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
else:
    #########################################################
    # Saving the Model
    #########################################################
    HLA_folder = HLA.replace("*", "").replace(":", "")
    model_dir = f'/global/scratch/users/sergiomar10/models/ESMC_Pretrain/{HLA_folder}/'
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
