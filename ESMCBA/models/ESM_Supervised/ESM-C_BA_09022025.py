import os
import sys
import random
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import json
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, pearsonr
import argparse
from scipy.stats import norm
import numpy as np
from collections import Counter

parser = argparse.ArgumentParser(description='Fine-tune ESM-Cambrian model with varying parameters.')
parser.add_argument('--name_of_model', type=str, default='ESM-C_evaluation_predictions_newconfig_FP', help='Name prefix for your model & CSV')
parser.add_argument('--encoding', type=str, default='HLA', help='Encoding HLA v. epitope')
parser.add_argument('--file_path', type=str, default='here', help='Path of training')
parser.add_argument('--HLA', type=str, default='HLA0201', help='HLA Type')
parser.add_argument('--train_size', type=float, default=1, help='Total Training Data')
parser.add_argument('--blocks_unfrozen', type=int, default=10, help='Number of Unfrozen Transformer blocks')
parser.add_argument('--base_block_lr', type=float, default=1e-5, help='Loss of Transformer Model Bloc')
parser.add_argument('--regression_block_lr', type=float, default=1e-5, help='Learning rate of Regression Head')


args = parser.parse_args()
name_of_model = args.name_of_model
encoding = args.encoding
file_path = args.file_path
size_of_train = args.train_size
blocks_unfrozen = args.blocks_unfrozen
base_block_lr = args.base_block_lr
regression_block_lr = args.regression_block_lr
HLA = args.HLA

name_of_model = name_of_model + '_MSE_'

if not torch.cuda.is_available():
    print("CUDA is not available. Exiting.")
    sys.exit(1)

device = torch.device("cuda")
print(f"Using device: {device}", flush=True)

# '_Hubber' + '_NO_pretrained_' HLA-0201_epitope_only/False-Full_pretraining_20000_seq_AUG_2_HE.pt
# file_path = '/global/scratch/users/sergiomar10/models/esm_c/masking/HLA-0201_epitope_only/False-Full_pretraining_20000_seq_AUG_2_HE.pt'
# Hubber _pre_trained  USED IN THE EAALS
# /global/scratch/users/sergiomar10/data/ESMC_Masking/True-Full_pretraining_10000_seq_AUG_4_HE.csv

###USED IN MOST EVALS 
#True-Full_pretraining_30000_seq_AUG_3_ALL_HLAS


print(f"HLA: {HLA}")
print(f"UNFROZEN BLOCKS: {blocks_unfrozen}")
print(f"ENCODING: {encoding}")
print(f"PROPORTION OF TRAINING DATA: {size_of_train}")
print(f"FILEPATH: {file_path}")

import torch
import torch.nn as nn
import torch.nn.init as init

class ESMBA(nn.Module):
    def __init__(self, base_model, dropout=0.3):
        """
        Args:
            base_model: A pretrained model that returns a structure with a
                        'last_hidden_state' attribute.
            dropout (float): Dropout rate applied after pooling.
        """
        super(ESMBA, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(960, 1)

        init.xavier_uniform_(self.regression_head.weight, gain=0.01)
        nn.init.uniform_(self.regression_head.bias, a=0.0, b=1.0) # e.g. 1.5 or 2.0

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model.forward(input_ids)
        hidden_states = outputs.hidden_states[-1].to(torch.float32)

        mask = attention_mask.unsqueeze(-1).float()
        masked_hidden_states = hidden_states * mask
        sum_embeddings = masked_hidden_states.sum(dim=1)
        sum_mask = mask.sum(dim=1)

        pooled_output = sum_embeddings / (sum_mask + 1e-8)
        pooled_output = self.dropout(pooled_output)
        regression_output = self.regression_head(pooled_output).squeeze(-1)

        return regression_output

        
def load_model(model_path, device='cuda'):
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # 1. Load base ESMC model
    base_model = ESMC.from_pretrained("esmc_300m").to(device)
    
    # 3. Wrap with ESMCMasked using saved config
    model = ESMBA(
        base_model,
    ).to(device)
    
    # 4. Load trained weights
    state_dict = checkpoint['model_state_dict']

    # Remove keys corresponding to the old mask head
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('mask_head')}
    
    model.load_state_dict(state_dict, strict=False)
    
    return model

print(file_path)

model = load_model(
    file_path,  # Must match training setting!
    device=device
)

base_model = model.base_model

#########################################################
# 4. Define a mapping for the 20 standard amino acids
#########################################################
amino_acids = "ARNDCEQGHILKMFPSTWYV"

# Map each amino acid to its tokenizer-assigned ID
aa_to_idx = {
    aa: base_model.tokenizer(text=aa).input_ids[1]  # index=1 to skip <cls> or start token
    for aa in amino_acids
}

# Reverse mapping: token_id → amino acid
idx_to_aa = {idx: aa for aa, idx in aa_to_idx.items()}

####################################################
# 5. Set up training: optimizer and loss function
####################################################

# Gather last 10 blocks into one list
last_10_block_params = []

min_range = 30 - blocks_unfrozen
for block_idx in range(min_range, 30):
    last_10_block_params.extend(
        list(model.base_model.transformer.blocks[block_idx].parameters())
    )

norm_params = list(model.base_model.transformer.norm.parameters())
# (optional) You can decide whether to keep final norm in the same LR group or not
last_10_block_params.extend(norm_params)


optimizer = optim.Adam(
    [
        {"params": last_10_block_params, "lr": base_block_lr},
        {"params": model.regression_head.parameters(), "lr": regression_block_lr},
    ],
    weight_decay=1e-5
)

################################################
# 6. Define a simple FASTA parser
################################################
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
        # Append the last sequence if present
        if seq:
            sequences.append((header, seq))
    return sequences

train_fasta = f"/global/scratch/users/sergiomar10/data/IEDB_SQL/IEDB_HLA{HLA}_final.csv"

epitope_data = pd.read_csv(train_fasta, header = None)
epitope_data.columns = ['sequence', 'ref_ID', 'submissionID', 'Epitope_ID','protein_origin', 'ID_SOURCE', "SOURCE_ORGANISM", "IC50_nM", "DESCRIPTION_BINDING", "Year_submission"]

train_fasta_path = "/global/scratch/users/sergiomar10/jupyter_notebooks/hla_protein_sequences.fasta"
all_data = parse_fasta(train_fasta_path)
hla_and_epitopes = []
for head, hla_seq in all_data:
    # Parse out the HLA naming from header
    head = head.split('|')[1][:7].replace('*', '').replace(':', '')
    if head == HLA:
        HLA_seq = hla_seq

epitope_data['IC50_nM'] = epitope_data['IC50_nM'].astype(str)
epitope_data['IC50_nM'] = epitope_data['IC50_nM'].str.replace('\\N', '0', regex=False)
epitope_data['IC50_nM'] = epitope_data['IC50_nM'].astype('float')
epitope_data['IC50_nM'] = epitope_data['IC50_nM'] + 1
epitope_data['IC50_nM'] = epitope_data['IC50_nM'].apply(np.log10)

epitope_data['Year_submission'] = epitope_data['Year_submission'].astype(str)
epitope_data['Year_submission'] = epitope_data['Year_submission'].str.replace('\\N', '0', regex=False)
epitope_data['Year_submission'] = epitope_data['Year_submission'].astype(int)

epitope_data = epitope_data[["IC50_nM","sequence","Year_submission"]]

epitope_data = epitope_data.values

filtered_data = []

for header, sequence, year_submission in epitope_data:
    if '+' in sequence:
        continue
    if '(' in sequence:
        continue
    if 'X' in sequence:
        continue
    if 'epitope' not in encoding:   
        sequence = HLA_seq + sequence
        
    filtered_data.append((header, sequence, year_submission))

print(f"Filtered {len(filtered_data)} sequences for training.", flush=True)

aggregated = pd.DataFrame(filtered_data, columns = ['label','sequence','testing'])
# aggregated = aggregated.iloc[:100]

def split_data(aggregated, size_of_train=1.0):
    """
    Split data into Train (bin-sampled), Validation (10% of that train),
    and Test (20% of total). For final 'external' test, you can still use
    aggregated['Year_submission'] > 2020 if desired.
    """
    # 1) Split out the portion to treat as test_data (20%)
    #    from the portion <= 2020
    training_data = aggregated[aggregated['testing'] <= 2020]

    # 2) Bin-sample 'train_data' according to your normal approach
    print(f'Threshold used for generating the training data {size_of_train}', flush=True)
    bin_edges = [0, 1, 2, 3, 4, 5, 6, 7]
    bin_centers_normal = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    pmf = norm.pdf(bin_centers_normal, loc=3, scale=1.0)
    pmf /= pmf.sum()
    total_samples = int(size_of_train * len(training_data))
    bin_samples_normal = np.round(pmf * total_samples).astype(int)

    sampled_sequences, sampled_labels = [], []
    for i in range(len(bin_centers_normal)):
        bin_min = bin_edges[i]
        bin_max = bin_edges[i + 1]
        df_bin = training_data[(training_data["label"] > bin_min) & (training_data["label"] <= bin_max)]
        n_samples = min(bin_samples_normal[i], len(df_bin))
        if n_samples > 0:
            sample = df_bin.sample(n=n_samples) #, random_state=42)
            sampled_sequences.extend(sample["sequence"].tolist())
            sampled_labels.extend(sample["label"].tolist())

    # Construct the final "train_data" from the bin-sampled sequences
    final_train = pd.DataFrame({"sequence": sampled_sequences, "label": sampled_labels})

    # 3) Now split *final_train* into 90% train, 10% validation
    #    This ensures we keep 10% (of the final training set) for validation
    train_data_final, val_data_final = train_test_split(
        final_train, 
        test_size=0.1, 
        # random_state=42, 
        shuffle=True
    )

    # Optional histogram to check distribution
    num_bins = 7
    counts, bin_edges_ = np.histogram(train_data_final['label'], bins=num_bins)
    max_count = max(counts) if len(counts) > 0 else 1
    scale_factor = 50 / max_count

    print(f"\nTrain Histogram with {num_bins} Bins (after bin-sampling & removing val set):")
    for i in range(len(bin_edges_) - 1):
        bin_range = f"[{bin_edges_[i]:.2f}, {bin_edges_[i+1]:.2f})"
        bar = "#" * int(counts[i] * scale_factor)
        print(f"{bin_range}: {bar} ({counts[i]})")

    print('We are generating the 2021 Testing set', flush = True)

    test_data = aggregated[aggregated['testing'] > 2020]

    if len(test_data['label']) < 10:
        print('Not enough samples past the 2021 treshold', flush = True)
        output_table = aggregated[~aggregated["sequence"].isin(sampled_sequences)]
        test_data = pd.concat([output_table, test_data])

    test_data = test_data[['sequence','label']]

    return train_data_final, val_data_final, test_data

# Your dataset now computes the protein object (and encoding) during __getitem__
class EpitopeDataset(Dataset):
    def __init__(self, sequences, labels, base_model=base_model, precompute=False):
        self.labels = labels
        if precompute:
            # Precompute the protein objects (and their encodings) once and store them
            self.data = []
            for seq in sequences:
                protein_obj = ESMProtein(sequence=seq)
                # Precompute the encoding (if your base_model is frozen or you want to save time)
                protein_tensor = base_model.encode(protein_obj)
                self.data.append((protein_obj, protein_tensor))
        else:
            # Otherwise, store the raw sequences so that they are computed on the fly in __getitem__
            self.sequences = sequences

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        if hasattr(self, "data"):
            # If precomputed, return the stored protein object and its encoding.
            protein_obj, protein_tensor = self.data[idx]
            return protein_obj, protein_tensor, torch.tensor(label, dtype=torch.float)
        else:
            # Otherwise, compute the protein object (and encoding) on the fly.
            seq = self.sequences[idx]
            protein_obj = ESMProtein(sequence=seq)
            protein_tensor = base_model.encode(protein_obj)
            return seq, protein_tensor, torch.tensor(label, dtype=torch.float)

# A collate function that gathers protein objects into a list (they may have varying shapes)
def collate_fn(batch):
    # Each element of batch is (sequence, protein_tensor, label)
    sequences, protein_tensors, labels = zip(*batch)
    # Leave sequences as a list (they are custom objects)
    # If protein_tensors have the same shape you can stack them, otherwise leave as list:
    try:
        protein_tensors = torch.stack(protein_tensors)
    except Exception:
        protein_tensors = list(protein_tensors)
    labels = torch.stack(labels)
    return list(sequences), protein_tensors, labels

# Data preparation function
def prepare_dataloaders(dataframe, batch_size=10, size_of_train=1):
    """
    Split the dataframe into train, validation, and test sets,
    then return corresponding DataLoader objects.
    """
    print(f"\n----------------------------------------\nPreparing data...", flush=True)
    train_data, val_data, test_data = split_data(dataframe, size_of_train=size_of_train)
    
    print(f"Training samples: {len(train_data)}, "
          f"Validation samples: {len(val_data)}, "
          f"Test samples: {len(test_data)}\n----------------------------------------\n", flush=True)

    train_dataset = EpitopeDataset(
        sequences=train_data["sequence"].values,
        labels=train_data["label"].values,
    )
    val_dataset = EpitopeDataset(
        sequences=val_data["sequence"].values,
        labels=val_data["label"].values,
    )
    test_dataset = EpitopeDataset(
        sequences=test_data["sequence"].values,
        labels=test_data["label"].values,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, train_data, val_data, test_data

# Prepare the DataLoaders (no tokenization in the DataLoader)
train_loader, val_loader, test_loader, train_data, val_data, test_data = prepare_dataloaders(
    dataframe=aggregated,
    batch_size=10,
    size_of_train=size_of_train
)

# criterion = nn.HuberLoss()
criterion = nn.MSELoss()

###############################################
# 9. Training loop with validation checks
###############################################

NUM_EPOCHS_FIRST_ROUND = 10

train_losses_round1 = []
train_spearman_round1 = []

print("Started Traininig.", flush = True)

for epoch in range(NUM_EPOCHS_FIRST_ROUND):
    model.train()
    total_train_loss = 0.0
    total_train_samples = 0
    train_predictions = []
    train_targets = []

    for batch_num, (sequences, protein_tensors, targets) in enumerate(train_loader, start=1):
        targets = targets.to(device)
        encoded_seq = base_model.tokenizer(sequences, return_tensors='pt', padding=True).input_ids.to(device)
        attention_mask = base_model.tokenizer(sequences, return_tensors='pt', padding=True).attention_mask.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=encoded_seq, attention_mask=attention_mask)

        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_train_loss += loss.item() * batch_size
        total_train_samples += batch_size
        
        train_predictions.append(outputs.detach().cpu().numpy())
        train_targets.append(targets.cpu().numpy())

    avg_train_loss = total_train_loss / total_train_samples
    train_predictions_flat = np.concatenate(train_predictions)
    train_targets_flat = np.concatenate(train_targets)
    train_spearman_corr, _ = spearmanr(train_targets_flat, train_predictions_flat)

    train_losses_round1.append(avg_train_loss)
    train_spearman_round1.append(train_spearman_corr)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS_FIRST_ROUND}] "
          f"Train Loss: {avg_train_loss:.4f}, Train Spearman: {train_spearman_corr:.4f}", flush = True)

        # -------------------------
    # Test Set Evaluation
    # -------------------------
    sequences_list = []
    predictions_list = []
    measured_list = []

    model.eval()
    with torch.no_grad():
        for batch_num, (sequences, protein_tensors, targets) in enumerate(test_loader, start=1):
            targets = targets.to(device)
            encoded_seq = base_model.tokenizer(sequences, return_tensors='pt', padding=True).input_ids.to(device)
            attention_mask = base_model.tokenizer(sequences, return_tensors='pt', padding=True).attention_mask.to(device)

            outputs = model(input_ids=encoded_seq, attention_mask=attention_mask)

            # Process each sample in the batch:
            for i, sequence in enumerate(sequences):
                # If the protein object has an attribute (e.g., `sequence.sequence`), use it; otherwise, use the raw value.
                if hasattr(sequence, "sequence"):
                    sequences_list.append(sequence.sequence)
                else:
                    sequences_list.append(sequence)
                predictions_list.append(outputs[i].cpu().numpy().item())
                measured_list.append(targets[i].cpu().numpy().item())

    eval_spearman, _ = spearmanr(predictions_list, measured_list)
    eval_pearson, _ = pearsonr(predictions_list, measured_list)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS_FIRST_ROUND}] "
          f"Train Loss: {avg_train_loss:.4f}, Eval Spearman: {eval_spearman:.4f} , Eval Pearson: {eval_pearson:.4f}", flush = True)

df = pd.DataFrame({
    "train_loss_round1": train_losses_round1,
    "train_spearman_round1": train_spearman_round1,
})

predictions_finetuned_esm = pd.DataFrame({
    'sequence': sequences_list,
    'prediction': predictions_list,
    'measured': measured_list
})

loss_dir = f'/global/scratch/users/sergiomar10/losses/ESMCBA_02032025/'
os.makedirs(loss_dir, exist_ok=True)

df_out_path = os.path.join(loss_dir, f'predictions_{name_of_model}.csv')
predictions_finetuned_esm.to_csv(df_out_path, index=False)
print(f"Saved DataFrame to {df_out_path}")

if eval_spearman > 0.30:

    loss_dir = f'/global/scratch/users/sergiomar10/losses/ESMCBA_02032025/training_data'
    os.makedirs(loss_dir, exist_ok=True)

    df_out_path = os.path.join(loss_dir, f'training_{name_of_model}.csv')
    predictions_finetuned_esm.to_csv(df_out_path, index=False)
    print(f"Saved Training Data to {df_out_path}")

    HLA_folder = HLA.replace("*", "").replace(":", "")
    model_dir = f'/global/scratch/users/sergiomar10/models/ESMCBA_02032025/{HLA_folder}/'
    os.makedirs(model_dir, exist_ok=True)

    final_model_path = os.path.join(model_dir, f"training_{name_of_model}_final.pth")
    df.to_csv(f"{loss_dir}/{name_of_model}.csv", index=False)
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")




