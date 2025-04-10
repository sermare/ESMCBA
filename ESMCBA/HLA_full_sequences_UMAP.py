import numpy as np
import os
import sys
import argparse
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from esm.models.esmc import ESMC

#########################################################
# Argument Parsing
#########################################################
parser = argparse.ArgumentParser(description='Extract ESM-C embeddings for HLA sequences.')
parser.add_argument('--name_of_model', type=str, default='ESM-C', help='Name prefix for output files')
parser.add_argument('--output_dir', type=str, default='/global/scratch/users/sergiomar10/ESMC_embeddings', help='Directory to save embeddings')
args = parser.parse_args()

name_of_model = args.name_of_model
output_dir = args.output_dir

#########################################################
# Device Check
#########################################################
device = torch.device("cpu")
print(f"Using device: {device}", flush=True)

#########################################################
# Load the Base ESM-C Model
#########################################################
print("Loading pretrained ESM_Cambrian model...", flush=True)
base_model = ESMC.from_pretrained("esmc_300m").to(device)

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
fasta_path = "/global/scratch/users/sergiomar10/ESMCBA/ESMCBA/jupyter_notebooks/hla_sequences.fasta"
hla_data = parse_fasta(fasta_path)

hla_sequences = [(header, seq) for header, seq in hla_data]
max_length = max([len(seq) for _, seq in hla_sequences])

print(f"Loaded {len(hla_sequences)} HLA sequences from {fasta_path}", flush=True)
print(f"Max sequence length: {max_length}", flush=True)

#########################################################
# Dataset for HLA Embeddings
#########################################################
class HLADataset(Dataset):
    def __init__(self, sequences, base_model, max_length):
        self.sequences = sequences  # List of (header, seq) tuples
        self.tokenizer = base_model.tokenizer
        self.max_length = max_length
        self.pad_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        header, seq = self.sequences[idx]
        encoding = self.tokenizer(
            seq,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        input_ids = encoding['input_ids'].squeeze(0)  # [seq_len]
        return input_ids, header

def collate_fn(batch):
    input_ids_list, headers_list = zip(*batch)
    input_ids = torch.stack(input_ids_list, dim=0)
    return input_ids, list(headers_list)

def get_dataloader(sequences, base_model, batch_size=8, max_length=400):
    dataset = HLADataset(sequences, base_model, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Preserve order from FASTA file
        collate_fn=collate_fn
    )
    return loader

#########################################################
# Extract Embeddings
#########################################################
batch_size = 8
hla_loader = get_dataloader(hla_sequences, base_model, batch_size=batch_size, max_length=max_length)

os.makedirs(output_dir, exist_ok=True)
embeddings = {}
headers_order = []  # To maintain original order

base_model.eval()
with torch.no_grad():
    for input_ids, headers in hla_loader:  # Removed attention_mask from unpacking
        input_ids = input_ids.to(device)

        # Forward pass through base model without attention_mask
        outputs = base_model.forward(input_ids)  # Assumes ESMC can handle input_ids alone
        last_hidden_state = outputs.hidden_states[-1].to(torch.float32)  # [batch_size, seq_len, hidden_dim]

        # Mean pool over sequence length to get [batch_size, hidden_dim]
        pooled_embeddings = last_hidden_state.mean(dim=1)  # [batch_size, 960]

        # Store embeddings
        for i, header in enumerate(headers):
            embeddings[header] = pooled_embeddings[i].cpu().numpy()
            headers_order.append(header)

# Convert embeddings to array
embedding_array = np.stack([embeddings[header] for header in headers_order])  # [n_sequences, hidden_dim]

# Save embeddings
embedding_df = pd.DataFrame(embedding_array, index=headers_order)
embedding_df.to_csv(os.path.join(output_dir, f"{name_of_model}_hla_embeddings.csv"), index_label="Allele")
print(f"Embeddings saved to {os.path.join(output_dir, f'{name_of_model}_hla_embeddings.csv')}", flush=True)

# # Perform UMAP reduction
# print("Performing UMAP reduction...", flush=True)
# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
# umap_embeddings = reducer.fit_transform(embedding_array)  # [n_sequences, 2]

# # Create UMAP DataFrame with HLA alleles in original order
# umap_df = pd.DataFrame(umap_embeddings, columns=["UMAP1", "UMAP2"], index=headers_order)
# umap_df.index.name = "Allele"

# # Save UMAP coordinates
# umap_csv_path = os.path.join(output_dir, f"{name_of_model}_hla_umap.csv")
# umap_df.to_csv(umap_csv_path)
# print(f"UMAP coordinates saved to {umap_csv_path}", flush=True)

# # Optional: Visualize UMAP
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 6))
# plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=50, c='blue', alpha=0.5)
# for i, header in enumerate(headers_order):
#     plt.annotate(header, (umap_embeddings[i, 0], umap_embeddings[i, 1]), fontsize=8, alpha=0.7)
# plt.title("UMAP of HLA Protein Embeddings (ESMC)")
# plt.xlabel("UMAP1")
# plt.ylabel("UMAP2")
# plt.savefig(os.path.join(output_dir, f"{name_of_model}_hla_umap.png"), dpi=300, bbox_inches='tight')
# plt.close()
# print(f"UMAP plot saved to {os.path.join(output_dir, f'{name_of_model}_hla_umap.png')}", flush=True)