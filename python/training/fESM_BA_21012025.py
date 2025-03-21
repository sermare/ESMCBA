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

# Set up argument parsing

import warnings
warnings.simplefilter("ignore")  # Ignore all warnings

parser = argparse.ArgumentParser(description='Fine-tune ESM model with varying parameters.')

parser.add_argument('--lr_input', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--layers_to_unfreeze', type=int, default=11, help='Layers of ESM')
parser.add_argument('--size_of_train', type=float, default=0.1, help='Size of Uniform distribution')
parser.add_argument('--batch_size', type=int, default=10, help='Size of Uniform distribution')
parser.add_argument('--run', type=int, default=1, help='RUNTrial')
parser.add_argument('--loss', type=str, default='MSE', help='Loss Function')
parser.add_argument('--hla', type=str, default='HLA*02:01', help='HLA Type')
parser.add_argument('--path', type=str, default='None', help='Path of Model')
parser.add_argument('--filter_20knM', type=str, default='False', help='Include the filter to cut-off at 2000 nM')

# Parse the arguments
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

# Update the model_dir and loss_dir to include the run_id
strategy = 'fine-tune' 

#######################
#     CONFIGURATION    #
#######################

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
FILE_PATH = "/global/scratch/users/sergiomar10/data/mhc_ligand_table_export_1727815365.tsv"
BATCH_SIZE = batch_size
MAX_LENGTH = 15
LEARNING_RATE = lr_input
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 10

if loss == "MSE":
    # Define loss and optimizer
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

torch.cuda.empty_cache()

#######################
#       FUNCTIONS      #
#######################

def aggregate_sequences(allele, filter_20k='False'):
    """
    Dynamically choose the CSV file based on the allele name.
    If the same sequence appears multiple times with different measurements, 
    aggregate them (e.g., take the mean) to reduce data leakage.
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
    elif 'A*29:02' in allele or 'A2902' in allele:
        csv_path = '/global/home/users/sergiomar10/HLA_A2902_training_data.csv'
    elif 'B*07:02' in allele or 'B0702' in allele:
        csv_path = '/global/home/users/sergiomar10/HLA-B0702_training_data.csv'
    else:
        raise ValueError(f"Unknown or unsupported allele: {allele}")
    
    # Read the chosen CSV
    aggregated = pd.read_csv(csv_path, header=0)
    print(f'Measurements with noise at 20,000 nM: {len(aggregated)}')

    aggregated['publication_year'] = aggregated['publication_year'].astype(str)
    aggregated['publication_year'] = aggregated['publication_year'].str.replace('\\N', '0', regex=False)
    aggregated['publication_year'] = aggregated['publication_year'].astype(int)
    aggregated['testing'] = aggregated['publication_year'] > 2020
        
    # (Optional) If the same peptide sequence appears multiple times, you could aggregate like this:
    # aggregated = aggregated.groupby(['sequence'], as_index=False).mean()

    # Filter out measurements if filter_20k is requested
    if filter_20k == 'True':
        # The 4.3 threshold presumably corresponds to 20,000 nM on a log-scale
        aggregated = aggregated[aggregated['label'] < 4.3]
        print(f'Filtering measurements with noise at 20,000 nM: {len(aggregated)}')
    
    return aggregated

from scipy.stats import norm

def split_data(aggregated, size_of_train=1.0, TESTING=False):

    print(f'Treshold used for generating the training data {size_of_train}',flush=True)

    train_data, test_data = train_test_split(
        aggregated[aggregated['testing'] <= 2020], test_size=0.2, random_state=10, shuffle=True
    )
    
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
        df_bin = train_data[
            (train_data["label"] > bin_min) & (train_data["label"] <= bin_max)
        ]
        n_samples = min(bin_samples_normal[i], len(df_bin))
        if n_samples > 0:
            sample = df_bin.sample(n=n_samples, random_state=42)
            sampled_sequences.extend(sample["sequence"].tolist())
            sampled_labels.extend(sample["label"].tolist())

    # for i in [4, 5]:
    #     bin_min = bin_edges[i]
    #     bin_max = bin_edges[i + 1]
    #     df_bin = train_data[
    #         (train_data["label"] > bin_min) & (train_data["label"] <= bin_max)
    #     ]
    #     n_samples = min(2, len(df_bin))
    #     if n_samples > 0:
    #         sample = df_bin.sample(n=n_samples, random_state=42)
    #         sampled_sequences.extend(sample["sequence"].tolist())
    #         sampled_labels.extend(sample["label"].tolist())

    output_table = pd.DataFrame({"sequence": sampled_sequences, "label": sampled_labels})

    if TESTING == True:
        print('We are generating the 2021 Testing set', flush = True)
        unused_portion = aggregated[aggregated['testing'] > 2020]

        if len(unused_portion['label']) < 1:
            print('Not enough samples past the 2021 treshold', flush = True)
            output_table = aggregated[~aggregated["sequence"].isin(sampled_sequences)]
            output_table = pd.concat([output_table, test_data])
            return output_table, test_data

        unused_portion = unused_portion[~unused_portion["sequence"].isin(output_table['sequence'])]
        output_table = unused_portion

    else:
                # Define number of bins
        num_bins = 5  # Change this to adjust bins

        # Compute histogram with specified bins
        counts, bin_edges = np.histogram(output_table['label'], bins=num_bins)

        # Normalize counts for ASCII representation
        max_count = max(counts)
        scale_factor = 50 / max_count if max_count > 0 else 1  # Scale to fit within 50 chars width

        # Print histogram with custom bins
        print(f"Label Histogram with {num_bins} Bins:")
        for i in range(len(bin_edges) - 1):
            bin_range = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
            bar = "#" * int(counts[i] * scale_factor)
            print(f"{bin_range}: {bar} ({counts[i]})")

    return output_table, test_data

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


def prepare_dataloaders(hla_type, file_path, tokenizer, batch_size=10, max_length=15, size_of_train=0.1, TESTING = False, filter_20k = 'False'):
    """
    Load data, filter by HLA, aggregate sequences, and create train and evaluation dataloaders.
    Ensures no data leakage by using unique sequences for train and test.
    """
    print(f"\n ---------------------------------------- \n Preparing data for {hla_type}...", flush=True)

    print(f'Filtering measurements with noise at 20,000 nM: {filter_20k}')
    aggregated = aggregate_sequences(hla_type, filter_20k = filter_20k)

    train_data, test_data = split_data(aggregated, size_of_train=size_of_train, TESTING=TESTING)

    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)} \n ---------------------------------------- \n", flush=True)

    train_dataset = EpitopeDataset(
        sequences=train_data["sequence"].values,
        labels=train_data["label"].values,
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_dataset = EpitopeDataset(
        sequences=train_data["sequence"].values,
        labels=train_data["label"].values,
        tokenizer=tokenizer,
        max_length=max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, eval_loader, train_data, test_data

#######################
#      MAIN CODE       #
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



print("Loading Dataset and Tokenizer.", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_loader, eval_loader, train_data, test_data = prepare_dataloaders(
    hla_type=HLA,
    file_path=FILE_PATH,
    tokenizer=tokenizer,
    batch_size=batch_size,
    max_length=MAX_LENGTH,
    size_of_train=size_of_train,
    filter_20k = filter_20k)

train_loader_full, eval_loader_full, train_data_full, test_data_full = prepare_dataloaders(
    hla_type=HLA,
    file_path=FILE_PATH,
    tokenizer=tokenizer,
    batch_size=batch_size,
    max_length=MAX_LENGTH,
    size_of_train=size_of_train,
    TESTING = True
)

base_model = EsmModel.from_pretrained(MODEL_NAME)
# Initialize the fine-tuned model
finetuned_model = ESMFineTune(base_model).to(device)

layers = list(finetuned_model.esm_model.named_children())

# Unfreeze the last n layers
for layer_name, layer in layers[-layers_to_unfreeze:]:
    for param in layer.parameters():
        param.requires_grad = True
    print(f"Unfrozen layer: {layer_name}")

# Keep parameters in the regression head trainable
for param in finetuned_model.regression_head.parameters():
    param.requires_grad = True

# Load the pre-trained model checkpoint into finetuned_model
pretrained_path = file_to_train
checkpoint = torch.load(pretrained_path, map_location=device)
finetuned_model.load_state_dict(checkpoint, strict=False)
print("Model Loaded.", flush = True)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, finetuned_model.parameters()), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY
)

HLA_folder = HLA.replace("*", "").replace(":", "")

model_dir = f'/global/scratch/users/sergiomar10/models/BA_04022025/{HLA_folder}/'
os.makedirs(model_dir, exist_ok=True)

train_losses = []
eval_losses = []

print("Starting training...", flush=True)
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}", flush=True)

    # Training phase
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

        batch_size = input_ids.size(0)
        train_loss_sum += loss.item() * batch_size
        total_train_samples += batch_size

        train_predictions.append(outputs.detach().cpu().numpy())
        train_targets.append(targets.cpu().numpy())

    avg_train_loss = train_loss_sum / total_train_samples
    train_losses.append(avg_train_loss)
    train_predictions_flat = np.concatenate(train_predictions)
    train_targets_flat = np.concatenate(train_targets)
    train_spearman, _ = spearmanr(train_targets_flat, train_predictions_flat)

    print(torch.cuda.memory_summary())
    
    # Evaluation phase
    finetuned_model.eval()
    eval_loss_sum = 0.0
    total_eval_samples = 0
    eval_predictions = []
    eval_targets_list = []

    with torch.no_grad():
        for input_ids, attention_mask, targets in eval_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            outputs = finetuned_model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets)

            batch_size = input_ids.size(0)
            eval_loss_sum += loss.item() * batch_size
            total_eval_samples += batch_size

            eval_predictions.append(outputs.cpu().numpy())
            eval_targets_list.append(targets.cpu().numpy())

    avg_eval_loss = eval_loss_sum / total_eval_samples
    eval_losses.append(avg_eval_loss)
    eval_predictions_flat = np.concatenate(eval_predictions)
    eval_targets_flat = np.concatenate(eval_targets_list)
    eval_spearman, _ = spearmanr(eval_targets_flat, eval_predictions_flat)

    print(
        f'Epoch {epoch+1}/{NUM_EPOCHS}, '
        f'Train Loss: {avg_train_loss:.6f}, Eval Loss: {avg_eval_loss:.6f}, '
        f'Train Spearman: {train_spearman:.4f}, Eval Spearman: {eval_spearman:.4f}',
        flush=True
    )

loss_dir = f'/global/scratch/users/sergiomar10/losses/BA_04022025/'
os.makedirs(loss_dir, exist_ok=True)

np.save(os.path.join(loss_dir, f'train_losses_{run_id}.npy'), np.array(train_losses))
np.save(os.path.join(loss_dir, f'eval_losses_{run_id}.npy'), np.array(eval_losses))
np.save(os.path.join(loss_dir, f'train_spearman_{run_id}.npy'), np.array(train_spearman))
np.save(os.path.join(loss_dir, f'eval_spearman_{run_id}.npy'), np.array(eval_spearman))

# Save model checkpoint after each epoch
model_path = os.path.join(model_dir, f"{run_id}_{epoch+1}.pth")
torch.save({
    'epoch': epoch,
    'model_state_dict': finetuned_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'eval_loss': avg_eval_loss,
    'train_spearman': train_spearman,
    'eval_spearman': eval_spearman
}, model_path)

print("Training completed.", flush=True)

#######################
#      EVALUATION     #
#######################

sequences_list = []
predictions_list = []
measured_list = []

finetuned_model.eval()

print("Starting Predictions of External Dataset.", flush = True)

# Disable gradient computation for evaluation
with torch.no_grad():
    for batch_num, (input_ids, attention_mask, targets) in enumerate(train_loader_full, start=1):
        
        # Move tensors to the appropriate device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        
        # Forward pass to get outputs
        outputs = finetuned_model(input_ids, attention_mask=attention_mask)
        
        # Update evaluation loss
        batch_size = input_ids.size(0)
        
        # Iterate over each sequence in the batch
        for i in range(batch_size):
            # Reconstruct the sequence from input_ids
            # Convert input_ids to tokens
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu())
            
            # Convert tokens to string and remove special tokens
            # Adjust the replacements based on the actual special tokens used by ESM
            sequence = ''.join(tokens).replace('<pad>', '').replace('<cls>', '').replace('<eos>', '')
            
            # Append the sequence, prediction, and measured value to respective lists
            sequences_list.append(sequence)
            predictions_list.append(outputs[i].cpu().numpy().item())
            measured_list.append(targets[i].cpu().numpy().item())

print("Done with predictions", flush = True)

# Create a DataFrame with the collected data
predictions_finetuned_esm = pd.DataFrame({
    'sequence': sequences_list,
    'prediction': predictions_list,
    'measured': measured_list
})

np.save(os.path.join(loss_dir, f'predictions_list{run_id}.npy'), np.array(predictions_list))
np.save(os.path.join(loss_dir, f'sequences_list{run_id}.npy'), np.array(sequences_list))
np.save(os.path.join(loss_dir, f'measured_list{run_id}.npy'), np.array(measured_list))