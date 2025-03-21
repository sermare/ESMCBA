import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from transformers import AutoTokenizer, EsmModel

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device Used:", device)
print("Some output", flush=True)

#######################
#     CONFIGURATION    #
#######################
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
FILE_PATH = "/global/scratch/users/sergiomar10/data/mhc_ligand_table_export_1727815365.tsv"
BATCH_SIZE = 10
MAX_LENGTH = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 20
HLA = "HLA-A*02:01"

#######################
#       FUNCTIONS      #
#######################

def load_data(file_path):
    """
    Load the raw data file and return it as a DataFrame.
    """
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip().split('\t')
            rows.append(line)
    df = pd.DataFrame(rows)
    # The last column seems to be empty based on original code, so we drop it
    df = df.iloc[:, :-1]

    # Rename columns
    df.columns = [x.replace('"', '') for x in df.iloc[0].values]
    df = df.iloc[1:, :]  # Remove header row
    return df


def filter_data(df, hla_type):
    """
    Filter the dataframe for a given HLA type and ensure correct units.
    Convert measurements to float and log-transform them.
    """
    required_columns = ["MHC Restriction - Name", "Assay - Units", "Epitope - Name", "Assay - Quantitative measurement"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in the data.")

    # Filter rows by HLA type and units
    filtered = df[(df["MHC Restriction - Name"] == hla_type) &
                  (df["Assay - Units"] == "nM")]

    # Convert to float and log-transform measurements
    filtered["Assay - Quantitative measurement"] = filtered["Assay - Quantitative measurement"].astype(float)
    filtered["log_measurement"] = np.log10(filtered["Assay - Quantitative measurement"])

    return filtered


def aggregate_sequences(filtered_df):
    """
    If the same sequence appears multiple times with different measurements, aggregate them.
    Here we take the mean. This reduces the chance of data leakage.
    """
    # Group by sequence and compute mean of log_measurement
    aggregated = filtered_df.groupby("Epitope - Name")["log_measurement"].mean().reset_index()
    aggregated = aggregated.rename(columns={"Epitope - Name": "sequence", "log_measurement": "label"})
    return aggregated


def split_data(aggregated):
    """
    Split into training and test sets ensuring no overlap of sequences.
    Since each sequence is now unique, a standard train_test_split suffices.
    """
    train_data, test_data = train_test_split(
        aggregated,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    # train_data and test_data contain unique sequences now.
    return train_data, test_data


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


def prepare_dataloaders(hla_type, file_path, tokenizer, batch_size=10, max_length=15):
    """
    Load data, filter by HLA, aggregate sequences, and create train and evaluation dataloaders.
    Ensures no data leakage by using unique sequences for train and test.
    """
    print(f"Preparing data for {hla_type}...")
    df = load_data(file_path)
    filtered = filter_data(df, hla_type)
    aggregated = aggregate_sequences(filtered)

    train_data, test_data = split_data(aggregated)
    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")

    train_dataset = EpitopeDataset(
        sequences=train_data["sequence"].values,
        labels=train_data["label"].values,
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
    eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, eval_loader, train_data, test_data


class ESMFineTune(nn.Module):
    def __init__(self, esm_model):
        super(ESMFineTune, self).__init__()
        self.esm_model = esm_model
        self.config = esm_model.config
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Regression head (single layer)
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


#######################
#      MAIN CODE       #
#######################

print("Loading tokenizer and model...")
print("Some output", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = EsmModel.from_pretrained(MODEL_NAME)

finetuned_model = ESMFineTune(base_model).to(device)

# Freeze all parameters in the base model
for param in finetuned_model.esm_model.parameters():
    param.requires_grad = False

# Keep parameters in the regression head trainable
for param in finetuned_model.regression_head.parameters():
    param.requires_grad = True

# Prepare data
train_loader, eval_loader, train_data, test_data = prepare_dataloaders(
    hla_type=HLA,
    file_path=FILE_PATH,
    tokenizer=tokenizer,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH
)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, finetuned_model.parameters()), 
                       lr=LEARNING_RATE, 
                       weight_decay=WEIGHT_DECAY)

# Create directories for model saving
model_dir = f'/global/scratch/users/sergiomar10/models/BA/'
os.makedirs(model_dir, exist_ok=True)

train_losses = []
eval_losses = []

print("Starting training...")
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print("Some output", flush=True)


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

    # Evaluation
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

    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.6f}, Eval Loss: {avg_eval_loss:.6f}, '
          f'Train Spearman: {train_spearman:.4f}, Eval Spearman: {eval_spearman:.4f}')
    print("Some output", flush=True)


# Save model checkpoint
model_path = os.path.join(model_dir, f"ESM_frozen_to_BA_{epoch+1}.pth")
torch.save({
    'epoch': epoch,
    'model_state_dict': finetuned_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'eval_loss': avg_eval_loss,
    'train_spearman': train_spearman,
    'eval_spearman': eval_spearman
}, model_path)

print("Training completed.")
