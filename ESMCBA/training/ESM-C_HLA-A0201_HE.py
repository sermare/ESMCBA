import sys
import random
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import json
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from sklearn.model_selection import train_test_split
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Fine-tune ESM model with varying parameters.')
parser.add_argument('--full_pretraining', type=str, default='False', help='True/False Full pretraining')
parser.add_argument('--name_of_model', type=str, default='ESM-C_evaluation_predictions_newconfig_FP', help='Name prefix for your model & CSV')
parser.add_argument('--num_seq', type=int, default=100, help='Number of Sequences')
parser.add_argument('--num_augmentations', type=int, default=1, help='Number of Augmentations')

args = parser.parse_args()
full_pretraining = args.full_pretraining.lower() == 'true'  # Simplified boolean conversion
name_of_model = args.name_of_model
num_seq = args.num_seq
num_augmentations = args.num_augmentations

if not torch.cuda.is_available():
    print("CUDA is not available. Exiting.")
    sys.exit(1)
device = torch.device("cuda")

print(f"Using device: {device}", flush=True)

####################################################
# 1. Extend the model with a masking head (LM Head)
####################################################
class ESMCMasked(nn.Module):
    """
    A simple wrapper that takes a pre-trained ESM C model and adds
    a masking (language modeling) head on top of the final hidden states.
    """
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


############################################################
# 2. Freeze parameters except for the last transformer block
#    and the new masking head.
############################################################
def freeze_model_except_last(base_model):
    """
    Freezes all parameters in the ESM C model except for
    the last transformer block and any custom heads.
    """
    # Freeze all parameters by default
    for param in base_model.parameters():
        param.requires_grad = full_pretraining

    # ESM C has base_model.transformer.blocks, so unfreeze last block:
    # Make sure you adjust the index [-1] if there's any difference in naming or layering
    if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "blocks"):
        for param in base_model.transformer.blocks[-1].parameters():
            param.requires_grad = True

##################################################
# 3. Load the base ESM C model and wrap it
##################################################
print("Loading pretrained ESM_Cambrian model...")
base_model = ESMC.from_pretrained("esmc_300m").to(device)

# Freeze everything except last transformer block
freeze_model_except_last(base_model)

# Create our extended masked model
model_masked = ESMCMasked(base_model, hidden_dim=64, num_aa=33).to(device)

#########################################################
# 4. Define a mapping for the 20 standard amino acids
#########################################################
amino_acids = "ARNDCEQGHILKMFPSTWYV"

# Map each amino acid to its tokenizer-assigned ID
aa_to_idx = {
    aa: base_model.tokenizer(text=aa).input_ids[1]  # index=1 to skip <cls> or start token
    for aa in amino_acids
}

# Reverse mapping: token_id → amino acid
idx_to_aa = {idx: aa for aa, idx in aa_to_idx.items()}

print(f"aa_to_idx: {aa_to_idx}", flush=True)
print(f"idx_to_aa: {idx_to_aa}", flush=True)

####################################################
# 5. Set up training: optimizer and loss function
####################################################
optimizer = optim.Adam(model_masked.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

############################################
# 7. Load and filter the data
############################################

train_fasta = "/global/scratch/users/sergiomar10/data/IEDB_SQL/IEDB_HLAA0201_final.csv"

epitope_data = pd.read_csv(train_fasta, header = None)
epitope_data.columns = ['sequence', 'ref_ID', 'submissionID', 'Epitope_ID','protein_origin', 'ID_SOURCE', "SOURCE_ORGANISM", "IC50_nM", "DESCRIPTION_BINDING", "Year_submission"]
epitope_data = epitope_data[epitope_data['DESCRIPTION_BINDING'].str.contains("Positive")][["ref_ID","sequence"]].values

HLA_0201 = "MIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDMGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWEP"

filtered_data = []
for header, sequence in epitope_data:
    if '+' in sequence:
        continue
    if '(' in sequence:
        continue
    if 'X' in sequence:
        continue
    
    sequence = HLA_0201 + sequence

    filtered_data.append((header, sequence))

filtered_data = filtered_data[:num_seq]

print(f"Filtered {len(filtered_data)} sequences for training.", flush=True)

filtered_data_new = []

for header, sequence in filtered_data:
    for augmentations in range(0, num_augmentations):
        filtered_data_new.append((header, sequence))

print(f"After augmentation: {len(filtered_data_new)} sequences for training.", flush=True)

# Split the data: 80% train, 10% validation, 10% evaluation
train_data, temp_data = train_test_split(filtered_data_new, test_size=0.2, random_state=42)
val_data, eval_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(
    f"Data split: {len(train_data)} train, "
    f"{len(val_data)} validation, {len(eval_data)} evaluation sequences.",
    flush=True
)

############################################################
# 8. Helper to compute accuracy on a given dataset
############################################################
def compute_accuracy(dataset):
    model_masked.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for header, seq in dataset:
            protein_obj = ESMProtein(sequence=seq)
            protein_tensor = base_model.encode(protein_obj)
            token_ids = protein_tensor.sequence
            L = token_ids.size(0)

            if L < 3:
                continue
            
            num_masks = min(1, L - 2)
            mask_positions = random.sample(range(377, L - 1), num_masks)

            labels = []
            valid_positions = []

            for pos in mask_positions:
                token_id = int(token_ids[pos].item())
                token_str = base_model.tokenizer.decode([token_id]).strip()

                if token_str not in aa_to_idx:
                    continue
                labels.append(token_id)
                valid_positions.append(pos)

            if not labels:
                continue

            labels = torch.tensor(labels, device=device)
            valid_positions = torch.tensor(valid_positions, device=device)

            masked_logits, _ = model_masked(protein_obj, mask_positions=valid_positions)
            preds = torch.argmax(masked_logits, dim=-1)

            correct += (preds == labels).sum().item()
            total += len(labels)

    model_masked.train()
    return correct / total if total > 0 else 0

###############################################
# 9. Training loop with validation checks
###############################################
num_epochs = 10  # Adjust as needed

log_file = f"/global/scratch/users/sergiomar10/logs/ESM_C/HLA0201_EO/training_log_{name_of_model}.csv"

with open(log_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Avg Loss", "Train Accuracy", "Val Accuracy"])

    for epoch in range(num_epochs):
        random.shuffle(train_data)
        total_loss = 0.0

        model_masked.train()
        for header, seq in train_data:
            protein_obj = ESMProtein(sequence=seq)
            protein_tensor = base_model.encode(protein_obj)
            token_ids = protein_tensor.sequence
            L = token_ids.size(0)

            if L < 3:
                continue

            num_masks = min(1, L - 2)
            mask_positions = random.sample(range(377, L - 1), num_masks)

            labels = []
            valid_positions = []

            for pos in mask_positions:
                token_id = int(token_ids[pos].item())
                token_str = base_model.tokenizer.decode([token_id]).strip()

                # Only mask if it's one of the 20 standard amino acids
                if token_str not in aa_to_idx:
                    continue

                labels.append(token_id)
                valid_positions.append(pos)

            if not labels:
                continue

            labels = torch.tensor(labels, device=device)
            valid_positions = torch.tensor(valid_positions, device=device)

            masked_logits, _ = model_masked(protein_obj, mask_positions=valid_positions)
            loss = criterion(masked_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Compute average loss and accuracies for the epoch
        avg_loss = total_loss / max(1, len(train_data))
        train_accuracy = compute_accuracy(train_data)
        val_accuracy = compute_accuracy(val_data)

        # Print the metrics for the current epoch
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_accuracy*100:.2f}% | "
            f"Val Acc: {val_accuracy*100:.2f}%",
            flush=True
        )

        # Save the metrics to the CSV file
        writer.writerow([epoch + 1, avg_loss, train_accuracy, val_accuracy])
    
print(f"Saved training epochs loss at {log_file}.")


##########################################################
# 10. Evaluation: Save predictions on the evaluation set
##########################################################
model_masked.eval()
eval_results = []

with torch.no_grad():
    for header, seq in eval_data:
        protein_obj = ESMProtein(sequence=seq)
        protein_tensor = base_model.encode(protein_obj)
        token_ids = protein_tensor.sequence
        L = token_ids.size(0)
        if L < 3:
            continue

        num_masks = min(1, L - 2)
        mask_positions = random.sample(range(377, L - 1), num_masks)

        original_tokens = []
        valid_positions = []
        for pos in mask_positions:
            token_id = int(token_ids[pos].item())
            token_str = base_model.tokenizer.decode([token_id]).strip()

            # Skip if token is not in your 20 AA dictionary
            if token_str not in aa_to_idx:
                continue
            original_tokens.append(token_str)
            valid_positions.append(pos)

        if not original_tokens:
            continue

        valid_positions = torch.tensor(valid_positions, device=device)
        masked_logits, _ = model_masked(protein_obj, mask_positions=valid_positions)

        probs = F.softmax(masked_logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        for i, pos in enumerate(valid_positions):
            pred_idx = preds[i].item()
            pred_prob = probs[i, pred_idx].item()
            eval_results.append({
                "sequence": seq,
                "position": int(pos.cpu().item()),
                "original_aa": original_tokens[i],
                "predicted_aa": idx_to_aa[pred_idx] if pred_idx in idx_to_aa else "<UNK>",
                "predicted_prob": pred_prob
            })

eval_df = pd.DataFrame(eval_results)
eval_csv_path = f"/global/scratch/users/sergiomar10/data/{name_of_model}.csv"
eval_df.to_csv(eval_csv_path, index=False)
print(f"Evaluation predictions saved to {eval_csv_path}", flush=True)

####################################################
# 11. Save the trained model and its configuration
####################################################
model_to_save = {
    "model_state_dict": model_masked.state_dict(),
    "config": {
        "hidden_dim": 64,
        "num_aa": 33,
        "model_type": "ESMCMasked"
    }
}

model_save_path = f"/global/scratch/users/sergiomar10/models/esm_c/HLA-0201_epitope_only/{name_of_model}.pt"
torch.save(model_to_save, model_save_path)

config_save_path = f"/global/scratch/users/sergiomar10/models/esm_c/HLA-0201_epitope_only/{name_of_model}.json"
with open(config_save_path, "w") as f:
    json.dump(model_to_save["config"], f)

print(f"Trained model saved to {model_save_path}")
print(f"Configuration saved to {config_save_path}")
