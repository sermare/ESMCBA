import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from esm.models.esmc import ESMC

# ------------------------
# Setup
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load base and wrapped model
class ESMCMasked(nn.Module):
    def __init__(self, base_model, hidden_dim=960, num_aa=33):
        super().__init__()
        self.base_model = base_model
        self.mask_head = nn.Linear(hidden_dim, num_aa)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model.forward(input_ids)
        hidden_states = outputs.hidden_states[-1].to(torch.float32)
        logits = self.mask_head(hidden_states)
        return logits

print("Loading base model...")
base_model = ESMC.from_pretrained("esmc_300m").to(device)
model = ESMCMasked(base_model, hidden_dim=960, num_aa=33).to(device)

ckpt_path = "/global/scratch/users/sergiomar10/ESMCBA/models/weights/ESMCBA_epitope_0.8_30_ESMMASK_epitope_FT_5_0.001_1e-06_AUG_6_HLAA0201_2_0.001_1e-06__2_A0201_Hubber_A0201_final.pth"
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# ------------------------
# Input epitope
# ------------------------
epitope = "YLQPRTFLL"
max_length = 15
tokenizer = base_model.tokenizer
pad_id = tokenizer.pad_token_id
mask_id = tokenizer.mask_token_id

# Tokenize original
encoding = tokenizer(
    epitope,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=max_length
)
original_input_ids = encoding['input_ids'].squeeze(0)
attention_mask = encoding['attention_mask'].squeeze(0)
true_ids = original_input_ids.clone()

# ------------------------
# Vocab mapping
# ------------------------
vocab = base_model.tokenizer.vocab
idx_to_token = {v: k for k, v in vocab.items()}
vocab_size = len(idx_to_token)

# ------------------------
# Run per-position masking
# ------------------------
logit_matrix = []     # Each row = softmax probs at masked position
nlls = []             # Each row = (position, true_aa, NLL)

with torch.no_grad():
    for pos in range(len(epitope)):
        # Clone original input
        input_ids = original_input_ids.clone()
        labels = torch.full_like(input_ids, -100)
        attention = attention_mask.clone()

        # Mask one position
        input_ids[pos] = mask_id
        labels[pos] = true_ids[pos]

        # Prepare batch
        input_ids = input_ids.unsqueeze(0).to(device)
        attention = attention.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)

        # Forward pass
        logits = model(input_ids, attention)         # [1, seq_len, vocab]
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = logits

        # Get NLL for correct amino acid at this position
        true_idx = labels[0, pos].item()
        token_nll = -log_probs[0, pos, true_idx].item()
        true_aa = tokenizer.decode([true_idx])

        nlls.append((pos + 1, true_aa, token_nll))

        # Save softmax probs at that position
        probs_row = probs[0, pos, :].cpu().numpy()
        logit_matrix.append(probs_row)

# ------------------------
# Save results
# ------------------------

# Save matrix of predictions
logit_matrix = np.array(logit_matrix)  # shape: [seq_len, vocab_size]
column_names = [idx_to_token.get(i, f"unk_{i}") for i in range(vocab_size)]
df_probs = pd.DataFrame(logit_matrix, columns=column_names)
df_probs.index = [f"Position_{i+1}" for i in range(len(epitope))]

prob_csv = f"softmax_matrix_{epitope}.csv"
df_probs.to_csv(prob_csv)
print(f"Saved softmax matrix to: {prob_csv}")

# Save NLLs per position
df_nll = pd.DataFrame(nlls, columns=["Position", "TrueAA", "NLL"])
nll_csv = f"nll_per_position_{epitope}.csv"
df_nll.to_csv(nll_csv, index=False)
print(f"Saved NLL per position to: {nll_csv}")

# Pretty print NLLs
print("\nPer-position NLLs:")
print(df_nll)
