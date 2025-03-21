import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel

import argparse
import glob
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import ipywidgets as widgets
from IPython.display import display

import warnings
warnings.simplefilter("ignore")  # Ignore all warnings

# Set up argument parsing
parser = argparse.ArgumentParser(description='Fine-tune ESM model with varying parameters.')
parser.add_argument('--lr_input', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--layers_to_unfreeze', type=int, default=11, help='Layers of ESM to unfreeze in second round')
parser.add_argument('--size_of_train', type=float, default=0.1, help='Size of Uniform distribution for sampling')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
parser.add_argument('--run', type=int, default=1, help='RUNTrial')
parser.add_argument('--loss', type=str, default='MSE', help='Loss Function')
parser.add_argument('--hla', type=str, default='HLA*02:01', help='HLA Type')
parser.add_argument('--path', type=str, default='None', help='Path of Model')
parser.add_argument('--filter_20knM', type=str, default='False', help='Include the filter to cut-off at 2000 nM')

args = parser.parse_args()

lr_input = args.lr_input
layers_to_unfreeze = args.layers_to_unfreeze
size_of_train = args.size_of_train
batch_size = args.batch_size
run = args.run
loss = args.loss
HLA = args.hla
file_to_train = args.path
filter_20k = args.filter_20knM

run_id = f'fESM_BA_lr_{lr_input}_LTU_{layers_to_unfreeze}_ts_{size_of_train}_b_{batch_size}_r_{run}_loss_{loss}_{HLA}_20kf_{filter_20k}'

#######################
#     CONFIGURATION    #
#######################

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
FILE_PATH = "/global/scratch/users/sergiomar10/data/mhc_ligand_table_export_1727815365.tsv"
BATCH_SIZE = batch_size
MAX_LENGTH = 15
LEARNING_RATE = lr_input
WEIGHT_DECAY = 1e-5

# Total epochs you want overall. You can split them between the two phases if desired.
NUM_EPOCHS_FIRST_ROUND = 10
NUM_EPOCHS_SECOND_ROUND = 5

if loss == "MSE":
    criterion = nn.MSELoss()
elif loss == "Huber":
    criterion = nn.HuberLoss()
else:
    criterion = nn.L1Loss()

#######################
#    SET SEEDS & DEV   #
#######################
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if not torch.cuda.is_available():
    print("CUDA is not available. Exiting.")
    sys.exit(1)
device = torch.device("cuda")
print(f"Using device: {device}")

#######################
#       FUNCTIONS      #
#######################

def aggregate_sequences(allele, filter_20k='False'):
    """
    Dynamically choose the CSV file based on the allele name.
    If the same sequence appears multiple times with different measurements,
    you could aggregate them (e.g., take the mean) to reduce data leakage.
    """
    # Pick the CSV file path based on the allele
    if 'B*57:01' in allele or 'B5701' in allele:
        csv_path = '/global/home/users/sergiomar10/HLA-B5701_training_data.csv'
    elif 'B*40:02' in allele or 'B4002' in allele:
        csv_path = '/global/home/users/sergiomar10/HLA-B4002_training_data.csv'
    elif 'A*29:02' in allele or 'A2902' in allele:
        csv_path = '/global/home/users/sergiomar10/HLA-A2902_training_data.csv'
    elif 'C*05:01' in allele or 'C0501' in allele:
        csv_path = '/global/home/users/sergiomar10/HLA-C0501_training_data.csv'
    elif 'B*15:01' in allele or 'B1501' in allele:
        csv_path = '/global/home/users/sergiomar10/HLA-B1501_training_data.csv'
    elif 'B*27:05' in allele or 'B2705' in allele:
        csv_path = '/global/home/users/sergiomar10/HLA-B2705_training_data.csv'
    elif 'A*03:01' in allele or 'A0301' in allele:
        csv_path = '/global/home/users/sergiomar10/HLA-A0301_training_data.csv'
    elif 'A*01:01' in allele or 'A0101' in allele:
        csv_path = '/global/home/users/sergiomar10/HLA-A0101_training_data.csv'
    elif 'A*02:01' in allele or 'A0201' in allele:
        csv_path = '/global/home/users/sergiomar10/HLA-A0201_training_data.csv'
    elif 'A*11:01' in allele or 'A1101' in allele:
        csv_path = '/global/home/users/sergiomar10/HLA-A1101_training_data.csv'
    elif 'B*51:01' in allele or 'B5101' in allele:
        csv_path = '/global/home/users/sergiomar10/HLA-B5101_training_data.csv'
    elif 'B*07:02' in allele or 'B0702' in allele:
        csv_path = '/global/home/users/sergiomar10/HLA-B0702_training_data.csv'
    else:
        raise ValueError(f"Unknown or unsupported allele: {allele}")
    
    # Read the chosen CSV
    aggregated = pd.read_csv(csv_path, header=0)
    print(f'Measurements with noise at 20,000 nM: {len(aggregated)}')

    # Reformat publication_year
    aggregated['publication_year'] = aggregated['publication_year'].astype(str)
    aggregated['publication_year'] = aggregated['publication_year'].str.replace('\\N', '0', regex=False)
    aggregated['publication_year'] = aggregated['publication_year'].astype(int)
    aggregated['testing'] = aggregated['publication_year'] > 2020
        
    # Filter out measurements if filter_20k is requested
    if filter_20k == 'True':
        aggregated = aggregated[aggregated['label'] < 4.3]
        print(f'Filtering measurements with noise at 20,000 nM: {len(aggregated)}')
    
    return aggregated

from scipy.stats import norm

def split_data(aggregated, size_of_train=1.0):
    """
    Split data into Train (bin-sampled), Validation (10% of that train),
    and Test (20% of total). For final 'external' test, you can still use
    aggregated['publication_year'] > 2020 if desired.
    """
    # 1) Split out the portion to treat as test_data (20%)
    #    from the portion <= 2020
    training_data = aggregated[aggregated['testing'] <= 2020]

    train_data, test_data = train_test_split(
        training_data, 
        test_size=0.2, 
        random_state=10, 
        shuffle=True
    )

    # 2) Bin-sample 'train_data' according to your normal approach
    print(f'Threshold used for generating the training data {size_of_train}', flush=True)
    bin_edges = [0, 1, 2, 3, 4, 6, 7]
    bin_centers_normal = [0.5, 1.5, 2.5, 3.5]
    pmf = norm.pdf(bin_centers_normal, loc=2.5, scale=1.0)
    pmf /= pmf.sum()
    total_samples = int(size_of_train * len(train_data))
    bin_samples_normal = np.round(pmf * total_samples).astype(int)

    sampled_sequences, sampled_labels = [], []
    for i in range(len(bin_centers_normal)):
        bin_min = bin_edges[i]
        bin_max = bin_edges[i + 1]
        df_bin = train_data[(train_data["label"] > bin_min) & (train_data["label"] <= bin_max)]
        n_samples = min(bin_samples_normal[i], len(df_bin))
        if n_samples > 0:
            sample = df_bin.sample(n=n_samples, random_state=42)
            sampled_sequences.extend(sample["sequence"].tolist())
            sampled_labels.extend(sample["label"].tolist())

    # Construct the final "train_data" from the bin-sampled sequences
    final_train = pd.DataFrame({"sequence": sampled_sequences, "label": sampled_labels})

    # 3) Now split *final_train* into 90% train, 10% validation
    #    This ensures we keep 10% (of the final training set) for validation
    train_data_final, val_data_final = train_test_split(
        final_train, 
        test_size=0.1, 
        random_state=42, 
        shuffle=True
    )

    # Optional histogram to check distribution
    num_bins = 5
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

    return train_data_final, val_data_final, test_data

class EpitopeDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.float)

def collate_fn(batch):
    input_ids_list, attention_mask_list, labels_list = zip(*batch)
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    labels = torch.stack(labels_list)
    return input_ids, attention_mask, labels

def prepare_dataloaders(hla_type, file_path, tokenizer, batch_size=10, max_length=15, size_of_train=0.1, filter_20k='False'):
    """
    Load data, filter by HLA, aggregate sequences, 
    then split into train, val, test. Return corresponding loaders.
    """
    print(f"\n----------------------------------------\nPreparing data for {hla_type}...", flush=True)
    print(f'Filtering measurements with noise at 20,000 nM: {filter_20k}')

    aggregated = aggregate_sequences(hla_type, filter_20k=filter_20k)
    train_data, val_data, test_data = split_data(aggregated, size_of_train=size_of_train)

    print(f"Training samples: {len(train_data)}, "
          f"Validation samples: {len(val_data)}, "
          f"Test samples: {len(test_data)}\n----------------------------------------\n", flush=True)

    train_dataset = EpitopeDataset(
        sequences=train_data["sequence"].values,
        labels=train_data["label"].values,
        tokenizer=tokenizer,
        max_length=max_length
    )

    val_dataset = EpitopeDataset(
        sequences=val_data["sequence"].values,
        labels=val_data["label"].values,
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_dataset = EpitopeDataset(
        sequences=test_data["sequence"].values,
        labels=test_data["label"].values,
        tokenizer=tokenizer,
        max_length=max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, train_data, val_data, test_data

#######################
#      MODEL CLASS     #
#######################
class ESMFineTune(nn.Module):
    def __init__(self, esm_model):
        super(ESMFineTune, self).__init__()
        self.esm_model = esm_model
        self.config = esm_model.config
        self.dropout = nn.Dropout(0.3)
        self.regression_head = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        
        attention_mask = attention_mask.unsqueeze(-1).float()
        masked_hidden_states = hidden_states * attention_mask
        sum_embeddings = masked_hidden_states.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1)
        
        pooled_output = sum_embeddings / sum_mask
        pooled_output = self.dropout(pooled_output)
        regression_output = self.regression_head(pooled_output).squeeze(-1)

        return regression_output

def freeze_entire_esm_except_regression(model: ESMFineTune):
    """
    Freeze all layers in the ESM model except the regression head.
    """
    for param in model.esm_model.parameters():
        param.requires_grad = False
    # Ensure regression head remains trainable
    for param in model.regression_head.parameters():
        param.requires_grad = True

def unfreeze_last_n_layers(model: ESMFineTune, n: int):
    """
    Unfreeze the last n layers of the ESM model.
    """
    # Layers are typically in model.esm_model.encoder.layer
    # Check how your ESM model is structured
    # For ESM2, it might be model.esm_model.encoder.layers or similar.
    # This example will assume model.esm_model.encoder.layers is valid:

    # Make sure the embedding layer stays frozen if you want it that way
    # We'll unfreeze just the last 'n' blocks
    total_blocks = len(model.esm_model.encoder.layer)
    for i, layer_module in enumerate(model.esm_model.encoder.layer):
        # If it's in the last n layers, unfreeze
        if i >= (total_blocks - n):
            for param in layer_module.parameters():
                param.requires_grad = True

#######################
#      MAIN CODE       #
#######################

# 1) Load tokenizer
print("Loading Dataset and Tokenizer.", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2) Prepare Data
train_loader, val_loader, test_loader, train_data, val_data, test_data = prepare_dataloaders(
    hla_type=HLA,
    file_path=FILE_PATH,
    tokenizer=tokenizer,
    batch_size=batch_size,
    max_length=MAX_LENGTH,
    size_of_train=size_of_train,
    filter_20k=filter_20k
)

# 3) Load Base ESM
base_model = EsmModel.from_pretrained(MODEL_NAME)

# 4) Initialize our model
finetuned_model = ESMFineTune(base_model).to(device)

# 5) Optionally, load the checkpoint (regression head alignment, etc.)
pretrained_path = file_to_train
if pretrained_path.lower() != "none":
    checkpoint = torch.load(pretrained_path, map_location=device)
    finetuned_model.load_state_dict(checkpoint, strict=False)
    print("Loaded weights from:", pretrained_path, flush=True)

#######################
#   ROUND 1 TRAINING  #
#  (freeze all ESM except regression head)
#######################

print("\n===== ROUND 1: Fine-tuned ESM Model =====")

unfreeze_last_n_layers(finetuned_model, layers_to_unfreeze)

# Create optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, finetuned_model.parameters()), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY
)

train_losses_round1 = []
val_losses_round1 = []
train_spearman_round1 = []
val_spearman_round1 = []

for epoch in range(NUM_EPOCHS_FIRST_ROUND):
    finetuned_model.train()
    train_loss_sum = 0.0
    total_train_samples = 0
    train_predictions = []
    train_targets = []

    for input_ids, attention_mask, targets in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = finetuned_model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size_ = input_ids.size(0)
        train_loss_sum += loss.item() * batch_size_
        total_train_samples += batch_size_

        train_predictions.append(outputs.detach().cpu().numpy())
        train_targets.append(targets.cpu().numpy())

    avg_train_loss = train_loss_sum / total_train_samples
    train_predictions_flat = np.concatenate(train_predictions)
    train_targets_flat = np.concatenate(train_targets)
    train_spearman_corr, _ = spearmanr(train_targets_flat, train_predictions_flat)

    # Validation
    finetuned_model.eval()
    val_loss_sum = 0.0
    total_val_samples = 0
    val_predictions = []
    val_targets_list = []

    with torch.no_grad():
        for input_ids, attention_mask, targets in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            val_out = finetuned_model(input_ids, attention_mask=attention_mask)
            val_loss = criterion(val_out, targets)

            batch_size_ = input_ids.size(0)
            val_loss_sum += val_loss.item() * batch_size_
            total_val_samples += batch_size_

            val_predictions.append(val_out.cpu().numpy())
            val_targets_list.append(targets.cpu().numpy())

    avg_val_loss = val_loss_sum / total_val_samples
    val_predictions_flat = np.concatenate(val_predictions)
    val_targets_flat = np.concatenate(val_targets_list)
    val_spearman_corr, _ = spearmanr(val_targets_flat, val_predictions_flat)

    train_losses_round1.append(avg_train_loss)
    val_losses_round1.append(avg_val_loss)
    train_spearman_round1.append(train_spearman_corr)
    val_spearman_round1.append(val_spearman_corr)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS_FIRST_ROUND}] "
        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
        f"Train Spearman: {train_spearman_corr:.4f}, Val Spearman: {val_spearman_corr:.4f}"
    )

#######################
#   ROUND 2 TRAINING  #
#  (unfreeze last `layers_to_unfreeze` layers)
#######################

print("\n===== ROUND 2: Unfreeze last {} ESM layers and continue training =====".format(2))

freeze_entire_esm_except_regression(finetuned_model)

# Rebuild the optimizer (now that more params are trainable)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, finetuned_model.parameters()), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY
)

train_losses_round2 = []
val_losses_round2 = []
train_spearman_round2 = []
val_spearman_round2 = []

for epoch in range(NUM_EPOCHS_SECOND_ROUND):
    finetuned_model.train()
    train_loss_sum = 0.0
    total_train_samples = 0
    train_predictions = []
    train_targets = []

    for input_ids, attention_mask, targets in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = finetuned_model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size_ = input_ids.size(0)
        train_loss_sum += loss.item() * batch_size_
        total_train_samples += batch_size_

        train_predictions.append(outputs.detach().cpu().numpy())
        train_targets.append(targets.cpu().numpy())

    avg_train_loss = train_loss_sum / total_train_samples
    train_predictions_flat = np.concatenate(train_predictions)
    train_targets_flat = np.concatenate(train_targets)
    train_spearman_corr, _ = spearmanr(train_targets_flat, train_predictions_flat)

    # Validation
    finetuned_model.eval()
    val_loss_sum = 0.0
    total_val_samples = 0
    val_predictions = []
    val_targets_list = []

    with torch.no_grad():
        for input_ids, attention_mask, targets in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            val_out = finetuned_model(input_ids, attention_mask=attention_mask)
            val_loss = criterion(val_out, targets)

            batch_size_ = input_ids.size(0)
            val_loss_sum += val_loss.item() * batch_size_
            total_val_samples += batch_size_

            val_predictions.append(val_out.cpu().numpy())
            val_targets_list.append(targets.cpu().numpy())

    avg_val_loss = val_loss_sum / total_val_samples
    val_predictions_flat = np.concatenate(val_predictions)
    val_targets_flat = np.concatenate(val_targets_list)
    val_spearman_corr, _ = spearmanr(val_targets_flat, val_predictions_flat)

    train_losses_round2.append(avg_train_loss)
    val_losses_round2.append(avg_val_loss)
    train_spearman_round2.append(train_spearman_corr)
    val_spearman_round2.append(val_spearman_corr)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS_SECOND_ROUND}] "
        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
        f"Train Spearman: {train_spearman_corr:.4f}, Val Spearman: {val_spearman_corr:.4f}"
    )

###############################
#   SAVE LOSSES & CHECKPOINTS
###############################
HLA_folder = HLA.replace("*", "").replace(":", "")
model_dir = f'/global/scratch/users/sergiomar10/models/BA_05022025_Post2021Test/{HLA_folder}/'
os.makedirs(model_dir, exist_ok=True)

loss_dir = f'/global/scratch/users/sergiomar10/losses/BA_05022025_Post2021Test/'
os.makedirs(loss_dir, exist_ok=True)

# Save final model after Round 2
final_model_path = os.path.join(model_dir, f"{run_id}_final.pth")
torch.save(finetuned_model.state_dict(), final_model_path)
print(f"Saved final model to {final_model_path}")

# Example of saving intermediate losses
np.save(os.path.join(loss_dir, f'train_losses_round1_{run_id}.npy'), np.array(train_losses_round1))
np.save(os.path.join(loss_dir, f'val_losses_round1_{run_id}.npy'), np.array(val_losses_round1))
np.save(os.path.join(loss_dir, f'train_spearman_round1_{run_id}.npy'), np.array(train_spearman_round1))
np.save(os.path.join(loss_dir, f'val_spearman_round1_{run_id}.npy'), np.array(val_spearman_round1))

np.save(os.path.join(loss_dir, f'train_losses_{run_id}.npy'), np.array(train_losses_round2))
np.save(os.path.join(loss_dir, f'val_losses_{run_id}.npy'), np.array(val_losses_round2))
np.save(os.path.join(loss_dir, f'train_spearman_{run_id}.npy'), np.array(train_spearman_round2))
np.save(os.path.join(loss_dir, f'val_spearman_round2_{run_id}.npy'), np.array(val_spearman_round2))

print("\n===== Training completed. Now evaluating on Test set. =====\n")

#######################
#      EVALUATION     #
#######################

sequences_list = []
predictions_list = []
measured_list = []

finetuned_model.eval()

with torch.no_grad():
    for batch_num, (input_ids, attention_mask, targets) in enumerate(test_loader, start=1):
        # Move tensors to the proper device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        # Forward pass to get predictions
        outputs = finetuned_model(input_ids, attention_mask=attention_mask)
        batch_size = input_ids.size(0)

        for i in range(batch_size):
            # Convert input_ids to tokens
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu())
            # Reconstruct the sequence string and remove special tokens (adjust replacements as needed)
            sequence = ''.join(tokens).replace('<pad>', '').replace('<cls>', '').replace('<eos>', '')
            
            # Append data from this sample to the lists
            sequences_list.append(sequence)
            predictions_list.append(outputs[i].cpu().numpy().item())
            measured_list.append(targets[i].cpu().numpy().item())

print("Done with predictions from train_loader_full", flush=True)

# Create a DataFrame with the collected data
predictions_finetuned_esm = pd.DataFrame({
    'sequence': sequences_list,
    'prediction': predictions_list,
    'measured': measured_list
})

# Save the lists as .npy files (if needed)
np.save(os.path.join(loss_dir, f'predictions_list_{run_id}.npy'), np.array(predictions_list))
np.save(os.path.join(loss_dir, f'sequences_list_{run_id}.npy'), np.array(sequences_list))
np.save(os.path.join(loss_dir, f'measured_list_{run_id}.npy'), np.array(measured_list))

# Optionally, save the DataFrame as a CSV (or pickle) file as your “ultimate” result
df_out_path = os.path.join(loss_dir, f'predictions_finetuned_esm_{run_id}.csv')
predictions_finetuned_esm.to_csv(df_out_path, index=False)
print(f"Saved DataFrame to {df_out_path}")
