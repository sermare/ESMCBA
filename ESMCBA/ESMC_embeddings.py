import os
import torch
import pandas as pd
import numpy as np

from esm.models.esmc import ESMC
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition   import PCA
from sklearn.manifold       import TSNE
from umap                   import UMAP

# ------------------------
# 1) Config & data prep
# ------------------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
MAX_LEN    = 400
PCA_DIMS   = None    # set to None to skip PCA
UMAP_K     = [400, 500, 700, 1000]
UMAP_DIST  = 0.1

# Base TSNE args
TSNE_BASE_KW = dict(
    n_components=2,
    metric='cosine',
    method='exact',
    random_state=42,
    verbose=2
)

TSNE_SWEEP = [
    # Small datasets or local structure focus
    {'perplexity': 5,   'learning_rate': 100,  'early_exaggeration': 12.0},
    {'perplexity': 15,  'learning_rate': 200,  'early_exaggeration': 12.0},
    
    # Medium datasets (1000-5000 sequences)  
    {'perplexity': 30,  'learning_rate': 200,  'early_exaggeration': 12.0},
    {'perplexity': 50,  'learning_rate': 300,  'early_exaggeration': 12.0},
    
    # Larger datasets
    {'perplexity': 100, 'learning_rate': 500,  'early_exaggeration': 12.0},
]

table_mt = pd.read_csv('/global/scratch/users/sergiomar10/table_mt.csv')

# ------------------------
# 2) Model wrapper + load
# ------------------------
class ESMCMasked(nn.Module):
    def __init__(self, base_model, hidden_dim=960, num_aa=33):
        super().__init__()
        self.base_model = base_model
        self.mask_head  = nn.Linear(hidden_dim, num_aa)

    def forward(self, input_ids, attention_mask=None):
        out = self.base_model.forward(input_ids)
        hs  = out.hidden_states[-1].to(torch.float32)
        logits = self.mask_head(hs)
        return logits

base = ESMC.from_pretrained("esmc_300m").to(DEVICE)
model = ESMCMasked(base, hidden_dim=960, num_aa=33).to(DEVICE)

ckpt = torch.load(
    "/global/scratch/users/sergiomar10/models/esm_c/masking/ALLHLAs_epitope_only/False-Full_pretraining_100000_seq_AUG_3_ALL_HLAS.pt",
    map_location=DEVICE
)
model.load_state_dict(ckpt, strict=False)
model.eval()

# ------------------------
# 3) Dataset carries pos + seq
# ------------------------
class ProteinDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LEN):
        self.seqs      = df['full_sequence'].tolist()
        self.positions = (df['position'] - 1).astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length= max_length

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        pos = self.positions[idx]
        enc = self.tokenizer(
            seq,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        return (
            enc.input_ids.squeeze(0),
            enc.attention_mask.squeeze(0),
            pos,
            seq
        )

# ------------------------
# 4) Embed-at-site + DR + save
# ------------------------
def run_embedding_and_dr(df):
    ds     = ProteinDataset(df, model.base_model.tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    embeds, seqs = [], []
    with torch.no_grad():
        for input_ids, attn_mask, pos_idx, batch_seqs in loader:
            input_ids = input_ids.to(DEVICE)
            _ = model(input_ids)  # forward mask head

            out = model.base_model.forward(input_ids)
            hs  = out.hidden_states[-1].float()

            batch_ix   = torch.arange(hs.size(0), device=DEVICE)
            site_embed = hs[batch_ix, pos_idx.to(DEVICE), :].cpu().numpy()

            embeds.append(site_embed)
            seqs.extend(batch_seqs)

    # stack + standardize
    X  = np.vstack(embeds)
    Xn = StandardScaler().fit_transform(X)

    # # PCA
    # if PCA_DIMS:
    #     Xn = PCA(n_components=PCA_DIMS, random_state=42).fit_transform(Xn)
    #     print(f"PCA → {PCA_DIMS}")

    # # # TSNE sweep
    # print("Running TSNE sweep…")
    # for params in TSNE_SWEEP:
    #     kw = {**TSNE_BASE_KW, **params}
    #     tsne = TSNE(**kw)
    #     coords = tsne.fit_transform(Xn)

    #     fname = (f"tsne_perp{params['perplexity']}"
    #              f"_lr{params['learning_rate']}"
    #              f"_ee{int(params['early_exaggeration'])}.csv")

    #     pd.DataFrame(coords, index=seqs, columns=['TSNE1','TSNE2']) \
    #       .to_csv(fname)

    #     print(f"→ wrote {fname}")

    # (Optional) UMAP block left commented
    for k in UMAP_K:
        for md in [1]:
            umap = UMAP(n_components=2, n_neighbors=k, min_dist=md,
                        random_state=7)
            coords = umap.fit_transform(Xn)
            pd.DataFrame(coords, index=seqs, columns=['UMAP1','UMAP2']) \
            .to_csv(f"umap_n{k}_md{int(md*100)}.csv")

# ------------------------
# 5) Execute
# ------------------------
run_embedding_and_dr(table_mt)
