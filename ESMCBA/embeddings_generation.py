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
from umap import UMAP  # umap‑learn >=0.5

###############################################################################
#                       ─── ARGUMENT PARSER ───                               #
###############################################################################

parser = argparse.ArgumentParser(description="Evaluate ESM‑Cambrian model, save embeddings, and compute UMAP reduction.")
parser.add_argument('--model_path', type=str, required=True, help='Path to pre‑trained model .pth file')
parser.add_argument('--name', type=str, default='hla-A0201', help='Name of output file')
parser.add_argument('--hla', type=str, default='hla-A0201', help='hla type (e.g., hla-A0201)')
parser.add_argument('--encoding', type=str, default='epitope', help='Encoding of the model (epitope or hla)')
parser.add_argument('--output_dir', type=str, default='/global/scratch/users/sergiomar10/ESMCBA/evaluations', help='Where to save results')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
parser.add_argument('--peptides', nargs='+', default=['MASK'], help='Peptides to evaluate')
parser.add_argument('--umap_dims', type=int, default=2, choices=[2, 3], help='UMAP output dimensionality')
parser.add_argument('--umap_neighbors', type=int, default=15, help='UMAP n_neighbors (local structure)')
args = parser.parse_args()

###############################################################################
#                         ─── DEVICE SETUP ───                                #
###############################################################################
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(0)
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

###############################################################################
#                        ─── MODEL DEFINITION ───                             #
###############################################################################
class ESMBA(nn.Module):
    """ESM‑based regressor returning both predictions and pooled embeddings."""
    def __init__(self, base_model, dropout=0.3):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(960, 1)

    def forward(self, input_ids, attention_mask=None, return_embedding=False):
        outputs = self.base_model.forward(input_ids)
        hidden_states = outputs.hidden_states[-1].to(torch.float32)  # (B,L,960)

        mask = attention_mask.unsqueeze(-1).float()  # (B,L,1)
        masked_hidden = hidden_states * mask
        sum_emb = masked_hidden.sum(dim=1)           # (B,960)
        sum_mask = mask.sum(dim=1)                   # (B,1)
        pooled = sum_emb / (sum_mask + 1e-8)         # mean pooling
        dropped = self.dropout(pooled)
        pred = self.regression_head(dropped).squeeze(-1)
        if return_embedding:
            return pred, pooled.detach()
        return pred

###############################################################################
#                         ─── LOAD CHECKPOINT ───                             #
###############################################################################

def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    base = ESMC.from_pretrained("esmc_300m").to(device)
    model = ESMBA(base).to(device)
    # ignore mask language‑model head weights
    filtered = {k: v for k, v in ckpt.items() if not k.startswith('mask_head')}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model

model = load_model(args.model_path, device)
base_model = model.base_model  # quick reference

###############################################################################
#                         ─── DATA HANDLING ───                               #
###############################################################################

# FASTA parser
def parse_fasta(file_path):
    """Parse sequences from a FASTA file."""
    sequences = []
    with open(file_path, 'r') as f:
        header, seq = None, ""
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

# Load hla sequences
fasta_path = "../jupyter_notebooks/other/hla_sequences.fasta"
hla_data = parse_fasta(fasta_path)

hla_sequence = [
    seq for header, seq in hla_data
    if header.replace(':', '').replace('*', '') == args.hla.replace("hla", "")
][0]

# Updated Dataset
class PeptideDataset(Dataset):
    def __init__(self, sequences, hla_sequence=None, encoding=None):
        if encoding == 'hla' and hla_sequence is not None:
            self.sequences = [hla_sequence + seq for seq in sequences]
        else:
            self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# Collate function
def collate_fn(batch):
    """Tokenize and pad inside DataLoader."""
    enc = base_model.tokenizer(batch, return_tensors='pt', padding=True)
    return batch, enc

# Prepare dataset
peptides = args.peptides
loader = DataLoader(
    PeptideDataset(peptides, hla_sequence=hla_sequence, encoding=args.encoding),
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

###############################################################################
#                         ─── EVALUATION LOOP ───                             #
###############################################################################
all_seqs, all_preds, all_embeds = [], [], []

with torch.no_grad():
    for seqs, enc in loader:
        input_ids = enc.input_ids.to(device)
        attn_mask = enc.attention_mask.to(device)
        preds, embeds = model(input_ids, attention_mask=attn_mask, return_embedding=True)

        all_seqs.extend(seqs)
        all_preds.append(preds.cpu())
        all_embeds.append(embeds.cpu())

all_preds = torch.cat(all_preds).numpy()
all_embeds = torch.cat(all_embeds).numpy()  # shape (N,960)


###############################################################################
#                        ─── SAVE RAW EMBEDDINGS ───                          #
###############################################################################
os.makedirs(args.output_dir, exist_ok=True)
emb_path = os.path.join(args.output_dir, f"{args.name}_embeddings.npy")
np.save(emb_path, all_embeds)
print(f"Saved raw embeddings → {emb_path}")

###############################################################################
#                      ─── UMAP DIMENSIONALITY REDUCTION ───                  #
###############################################################################

if args.umap_neighbors > len(peptides):
    umap_neighbors = len(peptides)
else:
    umap_neighbors = args.umap_neighbors
    


reducer = UMAP(n_components=args.umap_dims, n_neighbors=umap_neighbors) #, metric='cosine', random_state=42)
umap_coords = reducer.fit_transform(all_embeds)  # shape (N,umap_dims)

umap_cols = [f"UMAP_{i+1}" for i in range(args.umap_dims)]
result_df = pd.DataFrame({
    'sequence': all_seqs,
    'prediction': all_preds,
    **{c: umap_coords[:, i] for i, c in enumerate(umap_cols)}
})

csv_path = os.path.join(args.output_dir, f"{args.name}_umap.csv")
result_df.to_csv(csv_path, index=False)
print(f"Saved UMAP results → {csv_path}")

###############################################################################
print("✅ Finished evaluation + embedding extraction + UMAP")
