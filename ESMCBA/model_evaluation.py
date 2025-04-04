import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, pearsonr
import argparse

# Argument parser for command-line configuration
parser = argparse.ArgumentParser(description='Evaluate pre-trained ESM-Cambrian model on new data.')
parser.add_argument('--model_path', type=str, required=True, help='Path to pre-trained model .pth file')
parser.add_argument('--hla', type=str, default='HLA-A0201', help='HLA type (e.g., HLA-A0201)')
parser.add_argument('--output_dir', type=str, default='/global/scratch/users/sergiomar10/ESMCBA/evaluations', 
                    help='Directory to save evaluation results')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size for evaluation')
parser.add_argument('--peptides', nargs='+', default=['MASK'], help='Peptides to evaluate')

args = parser.parse_args()
model_path = args.model_path
hla = args.hla
output_dir = args.output_dir
batch_size = args.batch_size
peptides = args.peptides

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(0)
else:
    device = torch.device("cpu")
    
print(f"Using device: {device}", flush=True)

device = torch.device("cpu")
print(f"Using device: {device}", flush=True)

# Display configuration
print(f"Model Path: {model_path}")
print(f"HLA: {hla}")
print(f"Output Directory: {output_dir}")
print(f"Batch Size: {batch_size}")

# Define the regression model
class ESMBA(nn.Module):
    def __init__(self, base_model, dropout=0.3):
        """Initialize the ESM-based regression model."""
        super(ESMBA, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(960, 1)  # Single output for regression

    def forward(self, input_ids, attention_mask=None):
        """Forward pass of the model."""
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
    """Load a pre-trained model from a checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    base_model = ESMC.from_pretrained("esmc_300m").to(device)
    model = ESMBA(base_model).to(device)
    
    state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('mask_head')}
    model.load_state_dict(state_dict, strict=False)
    return model

# Load the model
model = load_model(model_path, device=device)
base_model = model.base_model
model.eval()  # Set to evaluation mode

# Dataset class for evaluation
class PeptideDataset(Dataset):
    def __init__(self, sequences, base_model):
        self.sequences = sequences
        self.base_model = base_model

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        protein_obj = ESMProtein(sequence=seq)
        protein_tensor = self.base_model.encode(protein_obj)
        return seq, protein_tensor

def collate_fn(batch):
    sequences, protein_tensors = zip(*batch)
    try:
        protein_tensors = torch.stack(protein_tensors)
    except Exception:
        protein_tensors = list(protein_tensors)
    return list(sequences), protein_tensors

# mhc_flurry_eval_data = pd.read_csv(mhcflurry_data_path)
# unique_peptides = mhc_flurry_eval_data['peptide'].unique()

unique_peptides = peptides

print(f"Loaded {len(unique_peptides)} unique peptides from MHCflurry data.", flush=True)
# Prepare DataLoader
eval_dataset = PeptideDataset(unique_peptides, base_model)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Evaluate on the unique peptides
predictions = []
sequences = []
# Create output directory
os.makedirs(output_dir, exist_ok=True)

chunk_idx = 0
total_count = 0

output_file = os.path.join(output_dir, f'evaluation_{hla}_mhcflurry_unique.csv')
write_header = True  # Only write header the first time

with torch.no_grad():
    for batch_sequences, protein_tensors in eval_loader:
        encoded = base_model.tokenizer(batch_sequences, return_tensors='pt', padding=True)
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        batch_df = pd.DataFrame({
            'sequence': batch_sequences,
            'prediction': outputs.cpu().numpy().squeeze()
        })

        # Append to CSV
        batch_df.to_csv(output_file, mode='a', index=False, header=write_header)
        if write_header:
            print(f"Started writing results to {output_file}")
            write_header = False

        total_count += len(batch_sequences)
        if total_count % 500 == 0:
            print(f"Processed {total_count} peptides...", flush=True)

print(f"✅ Done! Final output saved to {output_file}")    