import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import random
import numpy as np
from transformers import AutoTokenizer, EsmForMaskedLM
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("Loading Model")

# Define model name
model_name = "facebook/esm2_t33_650M_UR50D"

# Load the tokenizer and model
tokenizer_esm = AutoTokenizer.from_pretrained(model_name)

# Determine the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device Used:')
print(device)

# Define the amino acid vocabulary
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

# Load and preprocess the data
mhc_ligand_table = []
with open('/global/scratch/users/sergiomar10/data/mhc_ligand_table_export_1727815365.tsv', 'r') as file:
    for line in file:
        try:
            tokens = line.strip().split('\t')  # Tokenize using tab delimiter
            mhc_ligand_table.append(tokens)
        except Exception as e:
            print(f"Skipping line due to error: {e}")

mhc_ligand_table = pd.DataFrame(mhc_ligand_table)  # Convert the list to a DataFrame after reading

mhc_ligand_table = mhc_ligand_table.iloc[:, :-1]
mhc_ligand_table.columns = [x.replace('"', '') for x in mhc_ligand_table.iloc[0].values]

mhc_ligand_table = mhc_ligand_table.iloc[1:, :]

filtered_data = mhc_ligand_table[
    (mhc_ligand_table['MHC Restriction - Name'] == 'HLA-A*02:01') &
    (mhc_ligand_table['Assay - Units'] == 'nM')
]

# Convert 'Assay - Quantitative measurement' to float
filtered_data['Assay - Quantitative measurement'] = filtered_data['Assay - Quantitative measurement'].astype('float')

# Extract sequences and measurements
sequences = filtered_data['Epitope - Name'].values
measurements = filtered_data['Assay - Quantitative measurement'].astype(float).values

# Apply log transformation to the measurements
measurements = np.log10(measurements)

# Filter sequences to include only standard amino acids and length < 15
def filter_sequences(sequences, targets):
    filtered_sequences = []
    filtered_targets = []
    for seq, tgt in zip(sequences, targets):
        if all(c in amino_acids for c in seq) and len(seq) < 15:
            filtered_sequences.append(seq)
            filtered_targets.append(tgt)
    return filtered_sequences, filtered_targets

sequences, measurements = filter_sequences(sequences, measurements)

# Convert sequences and measurements to numpy arrays
sequences = np.array(sequences)
measurements = np.array(measurements)

# Define the dataset class
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

# Define the collate function
def collate_fn(batch):
    input_ids_list, attention_mask_list, targets_list = zip(*batch)
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    targets = torch.stack(targets_list)
    return input_ids, attention_mask, targets

import torch
import torch.nn as nn

class ESMFineTune(nn.Module):
    def __init__(self, esm_model):
        super(ESMFineTune, self).__init__()
        self.esm_model = esm_model
        self.config = esm_model.config

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Regression head
        self.regression_head = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.esm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]

        # Mean pooling over the sequence length
        attention_mask = attention_mask.unsqueeze(-1).float()
        masked_hidden_states = hidden_states * attention_mask
        sum_embeddings = masked_hidden_states.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1)

        pooled_output = sum_embeddings / sum_mask

        # Apply dropout for regularization
        pooled_output = self.dropout(pooled_output)

        regression_output = self.regression_head(pooled_output).squeeze(-1)

        return regression_output

from sklearn.model_selection import KFold

num_epochs = 10
batch_size = 10
learning_rate = 1e-4

model_dir = "/global/scratch/users/sergiomar10/models/BA_models/finetuned_ESM_BA/CV/"
os.makedirs(model_dir, exist_ok=True)

# Path to your pre-trained model checkpoint
pretrained_path = '/global/scratch/users/sergiomar10/models/hla0201/mask_models/finetuned_ESM_fine-tune/masking_model_epoch_10.pth'

# Set up KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_indices, test_indices) in enumerate(kfold.split(sequences)):
    print(f"Starting Fold {fold + 2}")

    # Split the data
    train_sequences = sequences[train_indices]
    train_labels = measurements[train_indices]
    test_sequences = sequences[test_indices]
    test_labels = measurements[test_indices]

    # Create dataset instances
    train_dataset = EpitopeDataset(
        sequences=train_sequences,
        targets=train_labels,
        tokenizer=tokenizer_esm,
        max_length=15
    )

    eval_dataset = EpitopeDataset(
        sequences=test_sequences,
        targets=test_labels,
        tokenizer=tokenizer_esm,
        max_length=15
    )

    # Initialize DataLoaders with the dataset instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Load the tokenizer and model
    esm_model = EsmForMaskedLM.from_pretrained(model_name)
    esm_model.to(device)

    # Initialize the regression model
    regression_model = ESMFineTune(esm_model).to(device)

    # Load the pre-trained model checkpoint
    pretrained_state_dict = torch.load(pretrained_path, map_location=device)

    # Extract weights that belong to the esm_model
    esm_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k.startswith('esm_model.'):
            esm_state_dict[k.replace('esm_model.', '')] = v

    # Load the ESM model weights into the regression model's esm_model
    regression_model.esm_model.load_state_dict(esm_state_dict, strict=False)

    # Freeze all layers in esm_model
    for param in regression_model.esm_model.parameters():
        param.requires_grad = False

    # Unfreeze the last 11 layers
    for layer in regression_model.esm_model.esm.encoder.layer[-11:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Ensure the regression head's parameters are trainable
    for param in regression_model.regression_head.parameters():
        param.requires_grad = True

    # Set up the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, regression_model.parameters()), lr=learning_rate, weight_decay=1e-5)

    # Define the loss function
    criterion = nn.MSELoss()

    # Create a directory for this fold
    fold_dir = os.path.join(model_dir, f"fold_{fold + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    for epoch in range(num_epochs):

        regression_model.train()
        train_loss = 0.0
        train_total = 0

        for input_ids, attention_mask, targets in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = regression_model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * input_ids.size(0)
            train_total += input_ids.size(0)

        avg_train_loss = train_loss / train_total

        # Evaluation Phase
        regression_model.eval()
        eval_loss = 0.0
        eval_total = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for input_ids, attention_mask, targets in eval_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                targets = targets.to(device)

                outputs = regression_model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, targets)

                eval_loss += loss.item() * input_ids.size(0)
                eval_total += input_ids.size(0)

                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

        avg_eval_loss = eval_loss / eval_total

        # Concatenate all outputs and targets
        all_outputs_tensor = torch.cat(all_outputs)
        all_targets_tensor = torch.cat(all_targets)

        # Compute Pearson correlation coefficient
        pearson_correlation = np.corrcoef(all_outputs_tensor.numpy(), all_targets_tensor.numpy())[0, 1]

        print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Eval Loss: {avg_eval_loss:.4f}")
        print(f"Eval Pearson Correlation: {pearson_correlation:.4f}\n")

        # Save the model after the 10th epoch
        if epoch == num_epochs - 1:
            model_path = os.path.join(fold_dir, f"regression_model_epoch_{epoch + 1}.pth")
            torch.save(regression_model.state_dict(), model_path)

    # After training, collect predictions on the test set
    regression_model.eval()
    all_outputs = []
    all_targets = []
    all_sequences = []

    with torch.no_grad():
        for input_ids, attention_mask, targets in eval_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            outputs = regression_model(input_ids, attention_mask=attention_mask)

            all_outputs.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Save predictions
    predictions = pd.DataFrame({
        'Sequence': test_sequences,
        'Target': all_targets,
        'Prediction': all_outputs
    })
    predictions_path = os.path.join(fold_dir, f"predictions_fold_{fold + 2}.csv")
    predictions.to_csv(predictions_path, index=False)

print("Cross-validation complete.")
