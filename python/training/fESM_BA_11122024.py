### MODEL FOR L1 LOSS
breokn 

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

# Set up argument parsing
parser = argparse.ArgumentParser(description='Fine-tune ESM model with varying parameters.')

parser.add_argument('--lr_input', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--layers_to_unfreeze', type=int, default=11, help='Layers of ESM')
parser.add_argument('--size_of_train', type=float, default=0.1, help='Size of Uniform distribution')
parser.add_argument('--batch_size', type=int, default=10, help='Size of Uniform distribution')
parser.add_argument('--run', type=int, default=1, help='RUNTrial')
# parser.add_argument('--loss', type=str, default=nn.MSELoss(), help='Loss Function')

# Parse the arguments
args = parser.parse_args()

lr_input = args.lr_input
layers_to_unfreeze = args.layers_to_unfreeze
size_of_train = args.size_of_train
batch_size = args.batch_size
run = args.run

run_id = f'fESM_BA_lr_{lr_input}_LTU_{layers_to_unfreeze}_ts_{size_of_train}_b_{batch_size}_r_{run}_L1'

# Update the model_dir and loss_dir to include the run_id
strategy = 'fine-tune'  # or 'other strategy'

#######################
#     CONFIGURATION    #
#######################
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
FILE_PATH = "/global/scratch/users/sergiomar10/data/mhc_ligand_table_export_1727815365.tsv"
BATCH_SIZE = batch_size
MAX_LENGTH = 15
LEARNING_RATE = lr_input
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 20
HLA = "HLA-A*02:01"

#######################
#    SET SEEDS & DEV   #
#######################
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device Used:", device, flush=True)

torch.cuda.empty_cache()

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
    # Drop the last empty column if it exists
    df = df.iloc[:, :-1]

    # Rename columns
    df.columns = [x.replace('"', '') for x in df.iloc[0].values]
    df = df.iloc[1:, :]  # Remove header row
    return df


def filter_data(df, hla_type):
    
    filtered = df[
        (df["MHC Restriction - Name"] == hla_type) &
        (df["Assay - Units"] == "nM")
    ].copy()
    
    # Step 2: Extract the first word from the sequence column
    # Replace 'Epitope - Name' with the correct sequence column if different
    filtered['first_word'] = filtered['Epitope - Name'].str.replace('+', '')
    filtered['first_word'] = filtered['first_word'].str.split(' ', expand = True)[0]
    
    # filtered['first_word'] = filtered['first_word'].str.replace('"', '')
    
    # # Step 3: Define the set of valid amino acid letters (uppercase)
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    
    # # Step 4: Function to validate sequences
    def is_valid_sequence(seq):
        return set(seq.upper()).issubset(valid_aas)
    
    # Apply the validation function to the 'first_word' column
    filtered['valid_sequence'] = filtered['first_word'].apply(is_valid_sequence)
    
    # # # Optional: Identify invalid sequences
    # invalid_sequences = filtered[~filtered['valid_sequence']]['first_word'].unique()
    
    # if len(invalid_sequences) > 4:
    #     print(f"Found {len(invalid_sequences)} invalid sequences. These will be excluded.")
    #     print("Invalid sequences:", invalid_sequences)
    
    # # Step 5: Exclude invalid sequences
    valid_filtered = filtered[filtered['valid_sequence']].copy()
    
    # Step 6: Convert 'Assay - Quantitative measurement' to float and log-transform
    valid_filtered["Assay - Quantitative measurement"] = valid_filtered["Assay - Quantitative measurement"].astype(float)
    valid_filtered["log_measurement"] = np.log10(valid_filtered["Assay - Quantitative measurement"])

    # (Optional) Drop helper columns if not needed
    valid_filtered.drop(columns=['first_word', 'valid_sequence'], inplace=True)
    
    return valid_filtered


def aggregate_sequences(filtered_df):
    """
    If the same sequence appears multiple times with different measurements, aggregate them.
    Here we take the mean. This reduces the chance of data leakage.
    """
    aggregated = filtered_df.groupby("Epitope - Name")["log_measurement"].mean().reset_index()
    aggregated = aggregated.rename(columns={"Epitope - Name": "sequence", "log_measurement": "label"})
    return aggregated


def split_data(aggregated):
    """
    Split into training and test sets ensuring no overlap of sequences.
    """
    train_data, test_data = train_test_split(
        aggregated,
        test_size=0.2,
        random_state=10,
        shuffle=True
    )

    uniform_train = []
    sequences_uniform = []
    
    for n in range(6):
        values = 0
        for y, x in train_data[ (train_data['label'] > n) & ( train_data['label'] < n + 1)] .values:
            sequences_uniform.append(y)
            uniform_train.append(x)
            values += 1
            if values > size_of_train * len(train_data['label']) :
                break

    train_data = pd.DataFrame( np.stack([sequences_uniform, uniform_train]).T , columns = ['sequence', 'label'])
    train_data['label'] =  train_data['label'].astype('float')

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
    print(f"Preparing data for {hla_type}...", flush=True)
    df = load_data(file_path)
    filtered = filter_data(df, hla_type)
    aggregated = aggregate_sequences(filtered)

    train_data, test_data = split_data(aggregated)
    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}", flush=True)

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

print("Loading tokenizer and base model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
pretrained_path = '/global/scratch/users/sergiomar10/models/masking_models_ha/numaug_3_maskprob_0.15_lr_0.0001_LTU_27_EL_alpha_0.01/model_epoch_10.pt'
checkpoint = torch.load(pretrained_path, map_location=device)
finetuned_model.load_state_dict(checkpoint, strict=False)

# Prepare data loaders
train_loader, eval_loader, train_data, test_data = prepare_dataloaders(
    hla_type=HLA,
    file_path=FILE_PATH,
    tokenizer=tokenizer,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH
)

# Define loss and optimizer
# criterion = nn.MSELoss()
criterion = nn.L1Loss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, finetuned_model.parameters()), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY
)

# Directory for saving models
model_dir = f'/global/scratch/users/sergiomar10/models/BA/'
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

loss_dir = f'/global/scratch/users/sergiomar10/losses/BA/'
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

model_name = "facebook/esm2_t33_650M_UR50D"

tokenizer_esm = AutoTokenizer.from_pretrained(model_name)

def prepare_data_for_hla(hla_type, batch_size=10, max_length=15,
                         file_path='/global/scratch/users/sergiomar10/data/mhc_ligand_table_export_1727815365.tsv',
                         tokenizer_esm = tokenizer_esm,
                        sequences_and_BA = None):


    if sequences_and_BA is None:
        # Load MHC ligand table
        mhc_ligand_table = []
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    tokens = line.strip().split('\t')
                    mhc_ligand_table.append(tokens)
                except Exception as e:
                    print(f"Skipping line due to error: {e}")
    
        mhc_ligand_table = pd.DataFrame(mhc_ligand_table)
        mhc_ligand_table = mhc_ligand_table.iloc[:, :-1]
        mhc_ligand_table.columns = [x.replace('"', '') for x in mhc_ligand_table.iloc[0].values]
        mhc_ligand_table = mhc_ligand_table.iloc[1:, :]
    
        # Filter for relevant data based on HLA type
        filtered_data = mhc_ligand_table[
            (mhc_ligand_table['MHC Restriction - Name'] == hla_type) &
            (mhc_ligand_table['Assay - Units'] == 'nM')
        ]
        filtered_data['Assay - Quantitative measurement'] = filtered_data['Assay - Quantitative measurement'].astype('float')
    
        # Prepare sequences and measurements
        sequences = filtered_data['Epitope - Name'].values
        measurements = filtered_data['Assay - Quantitative measurement'].values

        # Log-transform measurements
        measurements = np.log10(measurements)
    
        # Split the data into training and test sets
        train_sequences, test_sequences, train_labels, test_labels = train_test_split(
            sequences, measurements, test_size=0.2, random_state=42)

    else:
                # Load MHC ligand table
        mhc_ligand_table = []
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    tokens = line.strip().split('\t')
                    mhc_ligand_table.append(tokens)
                except Exception as e:
                    print(f"Skipping line due to error: {e}")
    
        mhc_ligand_table = pd.DataFrame(mhc_ligand_table)
        mhc_ligand_table = mhc_ligand_table.iloc[:, :-1]
        mhc_ligand_table.columns = [x.replace('"', '') for x in mhc_ligand_table.iloc[0].values]
        mhc_ligand_table = mhc_ligand_table.iloc[1:, :]
    
        # Filter for relevant data based on HLA type
        filtered_data = mhc_ligand_table[
            (mhc_ligand_table['MHC Restriction - Name'] == hla_type) &
            (mhc_ligand_table['Assay - Units'] == 'nM')
        ]
        filtered_data['Assay - Quantitative measurement'] = filtered_data['Assay - Quantitative measurement'].astype('float')

        training = filtered_data['Epitope - Name'].values
        
        external_set = sequences_and_BA[~sequences_and_BA['sequence'].isin(training)]
        
        sequences = external_set['sequence'].values
        measurements = external_set['measured'].values
    
        # Split the data into training and test sets
        train_sequences, test_sequences, train_labels, test_labels = train_test_split(
            sequences, measurements, test_size=0.99, random_state=42)
    
    tokenizer_esm = tokenizer_esm

    # Dataset class
    class EpitopeDataset(Dataset):
        def __init__(self, sequences, targets, tokenizer, max_length):
            self.sequences = sequences
            self.targets = targets
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            sequence = self.sequences[idx]
            target = self.targets[idx]
            encoding = self.tokenizer(
                sequence,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            return input_ids, attention_mask, torch.tensor(target, dtype=torch.float)

    # DataLoader collation function
    def collate_fn(batch):
        input_ids_list, attention_mask_list, targets_list = zip(*batch)
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        targets = torch.stack(targets_list)
        return input_ids, attention_mask, targets

    # Create datasets
    train_dataset = EpitopeDataset(
        sequences=train_sequences,
        targets=train_labels,
        tokenizer=tokenizer_esm,
        max_length=max_length
    )

    eval_dataset = EpitopeDataset(
        sequences=test_sequences,
        targets=test_labels,
        tokenizer=tokenizer_esm,
        max_length=max_length
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f'Done processing for evaluation with {len(test_sequences)} sequences')

    return train_sequences, test_sequences, eval_loader, sequences, measurements

print("Setting model now for Evaluation of External Dataset", flush = True)

mhc_ligand_table = []

with open('/global/scratch/users/sergiomar10/data/mhc_ligand_table_export_after_2015.tsv', 'r') as file:

    for line in file:
        try:
            tokens = line.strip().split('\t')
            mhc_ligand_table.append(tokens)
        except Exception as e:
            print(f"Skipping line due to error: {e}")

mhc_ligand_table = pd.DataFrame(mhc_ligand_table)
mhc_ligand_table = mhc_ligand_table.iloc[:, :-1]
mhc_ligand_table.columns = [x.replace('"', '') for x in mhc_ligand_table.iloc[0].values]
mhc_ligand_table = mhc_ligand_table.iloc[1:, :]
mhc_ligand_table.columns = mhc_ligand_table.iloc[0]
mhc_ligand_table = mhc_ligand_table.iloc[1:, :]
mhc_ligand_table.columns = [x.replace('"','') for x in mhc_ligand_table.columns]

external_set = mhc_ligand_table[
    (mhc_ligand_table['Date'].notna()) & 
    (mhc_ligand_table['Quantitative measurement'].notna()) &
    (mhc_ligand_table['Quantitative measurement'] != '')
]

# Convert 'Assay - Quantitative measurement' to float, catching any errors
external_set['Quantitative measurement'] = pd.to_numeric(
    external_set['Quantitative measurement'], errors='coerce'
)

# Drop rows where conversion to float resulted in NaN
external_set = external_set.dropna(subset=['Quantitative measurement'])

# Sort values by 'Reference - Date'
external_set = external_set.sort_values(by='Date')
external_set = external_set.iloc[:, [11, -15]]

external_set['sequence'] = external_set['Name'].astype('str')
external_set['measured'] = external_set['Quantitative measurement'].apply(np.log10)

train_sequences, eval_sequences, eval_loader, sequences, test_labels = prepare_data_for_hla('HLA-A*02:01',
                                                                sequences_and_BA = external_set)

sequences_list = []
predictions_list = []
measured_list = []

finetuned_model.eval()

print("Starting Predictions of External Dataset.", flush = True)

# Disable gradient computation for evaluation
with torch.no_grad():
    for batch_num, (input_ids, attention_mask, targets) in enumerate(eval_loader, start=1):
        
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
            tokens = tokenizer_esm.convert_ids_to_tokens(input_ids[i].cpu())
            
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

loss_dir = f'/global/scratch/users/sergiomar10/losses/BA/evals/'

os.makedirs(loss_dir, exist_ok=True)

np.save(os.path.join(loss_dir, f'predictions_list{run_id}.npy'), np.array(predictions_list))
np.save(os.path.join(loss_dir, f'sequences_list{run_id}.npy'), np.array(sequences_list))
np.save(os.path.join(loss_dir, f'measured_list{run_id}.npy'), np.array(measured_list))