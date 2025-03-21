import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import json
from collections import Counter

####################################################
# 1. Extend the model with a masking head (LM Head)
####################################################
class ESMCMasked(nn.Module):
   
    # A simple wrapper that takes a pre-trained ESM C model and adds
    # a masking (language modeling) head on top of the final hidden states.

    def __init__(self, base_model, hidden_dim=64, num_aa=33):
        super().__init__()
        self.base_model = base_model  # Pretrained ESM C model
        self.mask_head = nn.Linear(hidden_dim, num_aa)  # Simple linear LM head

    def forward(self, protein_obj, mask_positions=None):
        # Encode the protein to get initial embeddings
        encoded_seq = self.base_model.encode(protein_obj)
        # Obtain the hidden representations (logits call ensures forward pass)
        logits_out = self.base_model.logits(
            encoded_seq,
            LogitsConfig(sequence=True, return_embeddings=False)
        )
        # logits_out.logits.sequence is a list of length batch_size; here presumably 1
        # hidden has shape [L, hidden_dim] if single-sequence
        hidden = logits_out.logits.sequence[0]

        # Convert hidden from bfloat16 to float32 (match linear layer weights)
        hidden = hidden.to(self.mask_head.weight.dtype)

        # Pass through the custom LM head
        out_logits = self.mask_head(hidden)  # shape: [L, num_aa]

        if mask_positions is not None:
            # Return just the masked positions
            masked_logits = out_logits[mask_positions]
            return masked_logits, hidden
        else:
            # Or return logits for every position
            return out_logits

def load_finetuned_model(model_path, full_pretraining=False, device='cuda'):
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # 1. Load base ESMC model
    base_model = ESMC.from_pretrained("esmc_300m").to(device)
    
    # 3. Wrap with ESMCMasked using saved config
    model = ESMCMasked(
        base_model,
        hidden_dim=config['hidden_dim'],
        num_aa=config['num_aa']
    ).to(device)
    
    # 4. Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

model = load_finetuned_model(
    "/global/scratch/users/sergiomar10/models/esm_c/masking/fine_tuned/False-Full_pretraining_20000_seq_2.pt",
    full_pretraining=False,  # Must match training setting!
    device=device
)
model.eval()  # Set to evaluation mode"

#########################################################
# 4. Define a mapping for the 20 standard amino acids
#########################################################
amino_acids = "ARNDCEQGHILKMFPSTWYV"

# Map each amino acid to its tokenizer-assigned ID
aa_to_idx = {
    aa: model.base_model.tokenizer(text=aa).input_ids[1]  # index=1 to skip <cls> or start token
    for aa in amino_acids
}
# Reverse mapping: token_id → amino acid
idx_to_aa = {idx: aa for aa, idx in aa_to_idx.items()}

print(f"aa_to_idx: {aa_to_idx}", flush=True)
print(f"idx_to_aa: {idx_to_aa}", flush=True)

#########################################################
# 5. Loading Variants from Costello's Patients
#########################################################

print(f'USING: {device}', flush = True)


variants_df = pd.read_csv('/global/scratch/users/sergiomar10/data/variants/COSTELLO_VARIANTS_PT.csv')
variants_df = variants_df[variants_df['mutant_protein_sequence'].notna()]
variants_df['effect'] = [x[:10] for x in variants_df['top_effect']]
variants_pt = variants_df.drop_duplicates('top_effect')

allowed_terms = [
    "Substituti",
    "PrematureS",
    "FrameShift",
    "Insertion(",
    "Deletion(",
]

# filtered_df = variant',)s_df[variant',)s_df["mc"].isin(allowed_terms)]
filtered_df = variants_pt[variants_pt["effect"].isin(allowed_terms)]
filtered_df['effect'].value_counts()

####################################
# 1) compute_pll
####################################
def compute_pll(sequence, model, tokenizer, device, max_len=1022):
    """
    Compute the pseudo-log-likelihood (PLL) for a given protein sequence using
    an all-masked approach. Summation of log P(observed_aa | masked_input).
    """

    # Build the model-specific protein object
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.base_model.encode(protein)
    token_ids = protein_tensor.sequence  # e.g., [CLS] + actual AAs + [EOS]

    valid_positions = []
    for pos in range(1, len(token_ids) - 1):
        token_id = int(token_ids[pos].item())
        token_str = tokenizer.decode([token_id]).strip()
        if token_str in aa_to_idx:
            valid_positions.append(pos)

    if not valid_positions:
        return None

    valid_positions_tensor = torch.tensor(valid_positions, device=device)

    # Single forward pass with all valid positions masked
    with torch.no_grad():
        logits, _ = model(protein, mask_positions=valid_positions_tensor)
    log_probs = F.log_softmax(logits, dim=-1)

    # Re-map logits to the standard 20 amino acids
    std_aa_pairs = sorted(aa_to_idx.items(), key=lambda x: x[1])
    std_aas = [aa for aa, _ in std_aa_pairs]
    std_indices = [idx for _, idx in std_aa_pairs]
    log_probs_matrix = log_probs[:, std_indices].cpu()  # shape: (#valid_positions, 20)

    pll = 0.0
    # For each valid position, add log-prob of the actual residue
    for i, pos in enumerate(valid_positions):
        aa = sequence[pos - 1]  # adjusting for leading [CLS]
        if aa in std_aas:
            aa_idx = std_aas.index(aa)
            pll += log_probs_matrix[i, aa_idx].item()

    return pll

####################################
# 2) get_context_window
####################################
def get_context_window(sequence, mut_start, mut_end, max_len=1022, pre_context=511):
    """
    Extract a subsequence from 'sequence' that includes the region [mut_start, mut_end]
    plus up to 'pre_context' residues before. Ensures the total length does not exceed max_len.
    """
    start = max(0, mut_start - pre_context)
    end = start + max_len
    if end > len(sequence):
        end = len(sequence)
        start = max(0, end - max_len)
    return sequence[start:end], start

####################################
# 3) compute_stop_gain_effect
####################################
def compute_stop_gain_effect(wt_seq, stop_position, model, tokenizer, device):
    """
    Compute a single numeric measure of how damaging a premature stop is.
    We do one forward pass on the WT sequence. Then for each position in the
    lost region (i.e. after 'stop_position'), we compute the worst-case LLR
    vs. the WT residue.
    """

    protein = ESMProtein(sequence=wt_seq)
    protein_tensor = model.base_model.encode(protein)
    token_ids = protein_tensor.sequence

    # Identify valid positions
    valid_positions = []
    for pos in range(1, len(token_ids) - 1):
        token_id = int(token_ids[pos].item())
        token_str = tokenizer.decode([token_id]).strip()
        if token_str in aa_to_idx:
            valid_positions.append(pos)
    if not valid_positions:
        return None
    valid_positions_tensor = torch.tensor(valid_positions, device=device)

    # Forward pass
    with torch.no_grad():
        logits, _ = model(protein, mask_positions=valid_positions_tensor)
    log_probs = F.log_softmax(logits, dim=-1)

    # Map to 20 standard AAs
    std_aa_pairs = sorted(aa_to_idx.items(), key=lambda x: x[1])
    std_aas = [aa for aa, _ in std_aa_pairs]
    std_indices = [idx for _, idx in std_aa_pairs]
    log_probs_matrix = log_probs[:, std_indices].cpu()

    # Identify the index in valid_positions that corresponds to stop_position
    # We consider the region *after* stop_position as 'lost'
    if (stop_position + 1) < valid_positions[0] or (stop_position + 1) > valid_positions[-1]:
        return None  # The stop position is out of range for this model's valid positions
    # If you want to include the exact stop_position in "lost region", use >=
    stop_idx = next(i for i, vp in enumerate(valid_positions) if vp > stop_position)

    llrs = []
    # Iterate from stop_idx to end
    for i in range(stop_idx, len(valid_positions)):
        wt_aa = wt_seq[valid_positions[i] - 1]
        if wt_aa not in std_aas:
            continue
        wt_idx = std_aas.index(wt_aa)
        wt_log_prob = log_probs_matrix[i, wt_idx].item()

        # Compare vs. each alternative AA
        for alt_idx, alt_aa in enumerate(std_aas):
            if alt_aa == wt_aa:
                continue
            alt_log_prob = log_probs_matrix[i, alt_idx].item()
            llr = alt_log_prob - wt_log_prob
            llrs.append(llr)

    # Return the minimum (most negative) LLR as the "score"
    return min(llrs) if llrs else None

####################################
# 4) compute_frameshift_effect_local
####################################
def compute_frameshift_effect_local(wt_seq, mut_seq, mut_start, model, tokenizer, device, window_size=300):
    """
    Compare the local region (length=window_size or until sequence ends)
    starting at 'mut_start' in both WT and frameshift mutant. Return the difference
    in PLL of those windows.
    """

    # Protect against out-of-range
    if mut_start >= len(wt_seq) or mut_start >= len(mut_seq):
        return None  # The frameshift start is at or beyond the end

    region_length = min(len(wt_seq) - mut_start, len(mut_seq) - mut_start, window_size)
    wt_region = wt_seq[mut_start : mut_start + region_length]
    mut_region = mut_seq[mut_start : mut_start + region_length]

    wt_pll = compute_pll(wt_region, model, tokenizer, device, max_len=region_length)
    mut_pll = compute_pll(mut_region, model, tokenizer, device, max_len=region_length)

    if wt_pll is None or mut_pll is None:
        return None

    # The frameshift effect score (PLLR)
    return (mut_pll - wt_pll)


import traceback

# File to store the checkpoint (last processed index)
checkpoint_file = '/global/scratch/users/sergiomar10/data/checkpoint_variants_gpu.txt'

# Determine the starting index from the checkpoint file
try:
    with open(checkpoint_file, 'r') as f:
        start_index = int(f.read().strip())
except Exception:
    start_index = 0  # start from the beginning if checkpoint not available

results = []
tokenizer = model.base_model.tokenizer

# Process a subset of rows, e.g. tail(1000) or a slice of your dataframe
# Here, we resume processing from start_index to the end of the dataframe.
for idx, row in filtered_df.iloc[start_index:].iterrows():
    try:
        classification = row.get("effect", "missense")
        mutant_seq = row['mutant_protein_sequence']
        orig_seq = row['original_protein_sequence']
        gene_name = row["Gene"]
        mut_start = row.get("aa_mutation_start_offset", None)

        if classification in ["Substituti", "synonymous"]:
            wt_pll = compute_pll(orig_seq, model, tokenizer, device)
            mut_pll = compute_pll(mutant_seq, model, tokenizer, device)
            pllr = None
            if (wt_pll is not None) and (mut_pll is not None):
                pllr = mut_pll - wt_pll
            results.append({
                "protein": gene_name,
                "classification": classification,
                "pllr": pllr,
                "orig_seq": orig_seq,
                "mut_seq": mutant_seq
            })

        elif classification == "PrematureS":
            stop_position = row["aa_mutation_start_offset"]
            stop_gain_score = compute_stop_gain_effect(
                orig_seq, stop_position, model, tokenizer, device
            )
            results.append({
                "protein": gene_name,
                "classification": classification,
                "stop_position": stop_position,
                "stop_gain_score": stop_gain_score
            })

        elif (classification.startswith("Deletion") or
              classification.startswith("Insertion") or
              classification.startswith("FrameShift")):

            mut_start = row.get("aa_mutation_start_offset", None)
            if mut_start is None:
                results.append({
                    "protein": gene_name,
                    "error": "Missing aa_mutation_start_offset for indel"
                })
                continue
            mut_start = int(mut_start)

            len_diff = len(mutant_seq) - len(orig_seq)
            if len_diff % 3 != 0:
                frameshift_score = compute_frameshift_effect_local(
                    orig_seq, mutant_seq, mut_start, model, tokenizer, device
                )
                results.append({
                    "protein": gene_name,
                    "classification": classification,
                    "frameshift_score": frameshift_score
                })
            else:
                wt_pll = compute_pll(orig_seq, model, tokenizer, device)
                mut_pll = compute_pll(mutant_seq, model, tokenizer, device)
                inframe_score = None
                if wt_pll is not None and mut_pll is not None:
                    inframe_score = mut_pll - wt_pll
                results.append({
                    "protein": gene_name,
                    "classification": classification,
                    "inframe_score": inframe_score
                })

        elif classification == "StopLoss":
            old_stop = len(orig_seq)
            new_tail = mutant_seq[old_stop:]
            new_tail_pll = compute_pll(new_tail, model, tokenizer, device)
            results.append({
                "protein": gene_name,
                "classification": classification,
                "new_tail_pll": new_tail_pll,
                "original_protein_sequence": orig_seq,
                "mutant_protein_sequence": mutant_seq,
            })

        else:
            results.append({
                "protein": gene_name,
                "error": f"Unknown effect classification: {classification}"
            })

        # Optionally: periodically save progress (every N rows)
        if idx % 100 == 0:
            # Save checkpoint and intermediate results
            with open(checkpoint_file, 'w') as f:
                f.write(str(idx))
            variant_df = pd.DataFrame(results)
            variant_df.to_csv(f'/global/scratch/users/sergiomar10/data/variants_costello_seq_checkpoint_{idx}.csv', index=False)
            print(f"Processed up to row {idx}")

    except Exception as e:
        # Save current progress and print error with file information
        with open(checkpoint_file, 'w') as f:
            f.write(str(idx))
        variant_df = pd.DataFrame(results)
        variant_df.to_csv(f'/global/scratch/users/sergiomar10/variants_costello_seq_{idx}.csv', index=False)
        print(f"Error at row {idx} in variant: {e}")
        traceback.print_exc()
        # Optionally exit or continue processing
        break

# After finishing, you might want to clear the checkpoint or mark completion.
print("Processing complete.")