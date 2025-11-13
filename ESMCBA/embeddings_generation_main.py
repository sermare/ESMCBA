import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, pearsonr
import argparse
from umap import UMAP  # umap-learn >=0.5


###############################################################################
#                        ─── MODEL DEFINITION ───                             #
###############################################################################
class ESMBA(nn.Module):
    """ESM-based regressor returning both predictions and pooled embeddings."""
    def __init__(self, base_model, dropout=0.3):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(960, 1)

    def forward(self, input_ids, attention_mask=None, return_embedding=False):
        outputs = self.base_model.forward(input_ids)
        hidden_states = outputs.hidden_states[-1].to(torch.float32)  # (B,L,960)

        mask = attention_mask.unsqueeze(-1).float()   # (B,L,1)
        masked_hidden = hidden_states * mask
        sum_emb = masked_hidden.sum(dim=1)            # (B,960)
        sum_mask = mask.sum(dim=1)                    # (B,1)
        pooled = sum_emb / (sum_mask + 1e-8)          # mean pooling
        dropped = self.dropout(pooled)
        pred = self.regression_head(dropped).squeeze(-1)
        if return_embedding:
            return pred, pooled.detach()
        return pred


def load_model(path, device):
    """Load ESMCBA checkpoint from a .pth file."""
    ckpt = torch.load(path, map_location=device)
    base = ESMC.from_pretrained("esmc_300m").to(device)
    model = ESMBA(base).to(device)
    # ignore mask language-model head weights
    filtered = {k: v for k, v in ckpt.items() if not k.startswith('mask_head')}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


###############################################################################
#                         ─── FASTA / HLA HELPERS ───                         #
###############################################################################

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


def get_hla_sequence(hla, fasta_path):
    """Get HLA sequence from FASTA given an HLA like 'hla-A0201' or 'HLA-A*02:01'."""
    hla_data = parse_fasta(fasta_path)
    # normalize user HLA
    target = (
        hla.upper()
        .replace("HLA-", "")
        .replace("HLA", "")
        .replace("HLA_", "")
        .replace("HLA", "")
        .replace("HLA", "")
        .replace("HLA", "")
    )
    # more robust normalization:
    target = hla.replace("hla", "").replace("HLA", "")
    target = target.replace("HLA-", "").replace("HLA_", "")
    target = target.replace("*", "").replace(":", "")

    for header, seq in hla_data:
        header_norm = header.replace("HLA-", "").replace("HLA", "")
        header_norm = header_norm.replace("*", "").replace(":", "")
        if header_norm == target:
            return seq

    raise ValueError(f"Could not find HLA sequence for '{hla}' in FASTA '{fasta_path}'")


###############################################################################
#                         ─── DATASET / DATALOADER ───                        #
###############################################################################

class PeptideDataset(Dataset):
    def __init__(self, sequences, hla_sequence=None, encoding=None):
        """
        If encoding == 'hla', concatenate HLA + peptide.
        Otherwise, just use peptide.
        """
        if encoding == 'hla' and hla_sequence is not None:
            self.sequences = [hla_sequence + seq for seq in sequences]
        else:
            self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def make_collate_fn(base_model):
    def collate_fn(batch):
        """Tokenize and pad inside DataLoader."""
        enc = base_model.tokenizer(batch, return_tensors='pt', padding=True)
        return batch, enc
    return collate_fn


###############################################################################
#                         ─── CORE FUNCTION ───                               #
###############################################################################

def run_embeddings(
    model_path: str,
    name: str,
    hla: str,
    encoding: str = "epitope",
    output_dir: str = "/global/scratch/users/sergiomar10/ESMCBA/evaluations",
    batch_size: int = 10,
    peptides=None,
    umap_dims: int = 2,
    umap_neighbors: int = 15,
    fasta_path: str = "/global/scratch/users/sergiomar10/ESMCBA/ESMCBA/jupyter_notebooks/other/hla_sequences.fasta",
    device=None,
):
    """
    Core function that:
      - loads the model
      - builds dataset/dataloader
      - runs prediction + embedding extraction
      - runs UMAP
      - saves .npy embeddings and .csv UMAP
    Returns:
      (emb_path, csv_path)
    """
    if peptides is None:
        peptides = ["MASK"]

    # DEVICE SETUP
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.set_device(0)
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # LOAD MODEL
    model = load_model(model_path, device)
    base_model = model.base_model  # quick reference

    # HLA SEQUENCE
    hla_seq = get_hla_sequence(hla, fasta_path=fasta_path)

    # DATA
    dataset = PeptideDataset(peptides, hla_sequence=hla_seq, encoding=encoding)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(base_model),
    )

    # EVAL LOOP
    all_seqs, all_preds, all_embeds = [], [], []
    model.eval()
    with torch.no_grad():
        for seqs, enc in loader:
            input_ids = enc.input_ids.to(device)
            attn_mask = enc.attention_mask.to(device)
            preds, embeds = model(input_ids, attention_mask=attn_mask, return_embedding=True)

            all_seqs.extend(seqs)
            all_preds.append(preds.cpu())
            all_embeds.append(embeds.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_embeds = torch.cat(all_embeds).numpy()  # (N, 960)

    # SAVE RAW EMBEDDINGS
    os.makedirs(output_dir, exist_ok=True)
    emb_path = os.path.join(output_dir, f"{name}_embeddings.npy")
    np.save(emb_path, all_embeds)
    print(f"Saved raw embeddings → {emb_path}")

    # UMAP
    if umap_neighbors > len(peptides):
        umap_neighbors = len(peptides)

    reducer = UMAP(n_components=umap_dims, n_neighbors=umap_neighbors)
    umap_coords = reducer.fit_transform(all_embeds)  # (N, umap_dims)

    umap_cols = [f"UMAP_{i+1}" for i in range(umap_dims)]
    result_df = pd.DataFrame({
        'sequence': all_seqs,
        'prediction': all_preds,
        **{c: umap_coords[:, i] for i, c in enumerate(umap_cols)}
    })

    csv_path = os.path.join(output_dir, f"{name}_umap.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"Saved UMAP results → {csv_path}")
    print("✅ Finished evaluation + embedding extraction + UMAP")

    return emb_path, csv_path


###############################################################################
#                         ─── CLI ENTRY (OLD STYLE) ───                       #
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ESM-Cambrian model, save embeddings, and compute UMAP reduction."
    )
    parser.add_argument('--model_path', type=str, required=True, help='Path to pre-trained model .pth file')
    parser.add_argument('--name', type=str, default='hla-A0201', help='Name of output file')
    parser.add_argument('--hla', type=str, default='hla-A0201', help='hla type (e.g., hla-A0201)')
    parser.add_argument('--encoding', type=str, default='epitope', help='Encoding of the model (epitope or hla)')
    parser.add_argument('--output_dir', type=str,
                        default='/global/scratch/users/sergiomar10/ESMCBA/evaluations',
                        help='Where to save results')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--peptides', nargs='+', default=['MASK'], help='Peptides to evaluate')
    parser.add_argument('--umap_dims', type=int, default=2, choices=[2, 3], help='UMAP output dimensionality')
    parser.add_argument('--umap_neighbors', type=int, default=15, help='UMAP n_neighbors (local structure)')
    return parser.parse_args()


def main():
    args = parse_args()
    run_embeddings(
        model_path=args.model_path,
        name=args.name,
        hla=args.hla,
        encoding=args.encoding,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        peptides=args.peptides,
        umap_dims=args.umap_dims,
        umap_neighbors=args.umap_neighbors,
    )


if __name__ == "__main__":
    main()

