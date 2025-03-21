from transformers import AutoTokenizer, EsmModel
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Fine-tune ESM model with varying parameters.')
parser.add_argument('--num_augmentations_input', type=int, default=3, help='Number of augmentations')
parser.add_argument('--mask_prob_input', type=float, default=0.15, help='Masking probability')
parser.add_argument('--lr_input', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--hla_type', type=str, default='HLA-A*02:01', help='Allele')
parser.add_argument('--alpha', type=float, default=0.001, help='Weight for the energy loss')
parser.add_argument('--layers_to_unfreeze', type=int, default=11, help='Layers of ESM')

args = parser.parse_args()

# Assign the arguments to variables
num_augmentations_input = args.num_augmentations_input
mask_prob_input = args.mask_prob_input
lr_input = args.lr_input
hla_type = args.hla_type
alpha = args.alpha
layers_to_unfreeze = args.layers_to_unfreeze

run_id = f'numaug_{num_augmentations_input}_maskprob_{mask_prob_input}_lr_{lr_input}_LTU_{layers_to_unfreeze}_EL_alpha_{alpha}_allele_{hla_type}'

strategy = 'fine-tune'
model_dir = f'/global/scratch/users/sergiomar10/models/masking/{run_id}/'
os.makedirs(model_dir, exist_ok=True)

loss_dir = f'/global/scratch/users/sergiomar10/losses/masking/{run_id}/'
os.makedirs(loss_dir, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("Loading Model")

import sys, os
os.path.dirname(sys.executable)

model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer_esm = AutoTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name)

for param in model.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device Used:')
print(device)

model.to(device)

amino_acid_tokens = list('ACDEFGHIKLMNPQRSTVWY')
amino_acid_token_ids = tokenizer_esm.convert_tokens_to_ids(amino_acid_tokens)
num_amino_acid_tokens = len(amino_acid_token_ids)

token_id_to_class_idx = {token_id: idx for idx, token_id in enumerate(amino_acid_token_ids)}
class_idx_to_token_id = {idx: token_id for token_id, idx in token_id_to_class_idx.items()}

if strategy == 'fine-tune':
    for param in model.parameters():
        param.requires_grad = False

    for layer in model.encoder.layer[-layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True

    class ESMFineTune(nn.Module):
        def __init__(self, esm_model, amino_acid_token_ids):
            super(ESMFineTune, self).__init__()
            self.esm_model = esm_model
            self.amino_acid_token_ids = amino_acid_token_ids
            self.fc_layers = nn.Sequential(
                nn.Linear(esm_model.config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, len(amino_acid_token_ids))  # Output size is 20
            )
            
        def forward(self, input_ids, attention_mask=None):
            outputs = self.esm_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state
            logits = self.fc_layers(hidden_states)
            return logits

    masking_model = ESMFineTune(model, amino_acid_token_ids).to(device)
else:
    raise ValueError("Invalid strategy selected.")

df = pd.read_csv('~/epitopes_iedb.csv', sep=',', header=2)
def generate_train_test_loader(dataset, hla_type):
    amino_acids = set(amino_acid_tokens)

    sequences_a1 = df[df['Name.1'] == (hla_type)]['Name'].values
    sequences_a1 = np.unique(sequences_a1)
    train_sequences, eval_sequences = train_test_split(sequences_a1, test_size=0.2, random_state=42)
    
    filtered_train_sequences = [x for x in train_sequences if all(c in amino_acids for c in x) and len(x) <= 15]
    filtered_eval_sequences = [x for x in eval_sequences if all(c in amino_acids for c in x) and len(x) <= 15]
    
    filtered_train_sequences = np.array(filtered_train_sequences)
    filtered_eval_sequences = np.array(filtered_eval_sequences)

    print(f'Began with {len(sequences_a1)} sequences')
    print(f'Now with {len(filtered_train_sequences)} training filtered sequences')
    print(f'Now with {len(filtered_eval_sequences)} evaluation filtered sequences')

    def tokenize_and_mask_sequences(sequences, num_augmentations=num_augmentations_input, mask_prob=mask_prob_input, max_length=15):
        masked_sequences = []
        masked_positions_list = []
        targets_list = []
        for seq in sequences:
            tokenized = tokenizer_esm(
                seq,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids = tokenized.input_ids.squeeze(0)
            attention_mask = (input_ids != tokenizer_esm.pad_token_id).long()
            seq_length = attention_mask.sum().item()
            num_masks = max(1, int(seq_length * mask_prob))
    
            amino_acid_positions = [
                pos for pos in range(1, seq_length - 1)
                if input_ids[pos].item() in token_id_to_class_idx
            ]
    
            for _ in range(num_augmentations):
                if len(amino_acid_positions) >= num_masks:
                    positions = random.sample(amino_acid_positions, num_masks)
                else:
                    positions = amino_acid_positions
    
                masked_input_ids = input_ids.clone()
                targets = []
                positions_in_sample = []
                for pos in positions:
                    token_id = input_ids[pos].item()
                    if token_id in token_id_to_class_idx:
                        targets.append(token_id_to_class_idx[token_id])
                        masked_input_ids[pos] = tokenizer_esm.mask_token_id
                        positions_in_sample.append(pos)
                if targets:
                    masked_sequences.append(masked_input_ids)
                    masked_positions_list.append(positions_in_sample)
                    targets_list.append(targets)

        return masked_sequences, masked_positions_list, targets_list

    train_masked_sequences, train_masked_positions, train_targets = tokenize_and_mask_sequences(filtered_train_sequences)
    eval_masked_sequences, eval_masked_positions, eval_targets = tokenize_and_mask_sequences(filtered_eval_sequences)

    print(f'Total of {len(train_masked_sequences)} training sequences')
    print(f'Total of {len(eval_masked_sequences)} evaluation sequences')
    
    batch_size = 15

    class EpitopeDataset(Dataset):
        def __init__(self, masked_sequences, masked_positions_list, targets_list):
            self.masked_sequences = masked_sequences
            self.masked_positions_list = masked_positions_list
            self.targets_list = targets_list

        def __len__(self):
            return len(self.masked_sequences)

        def __getitem__(self, idx):
            input_ids = self.masked_sequences[idx]
            attention_mask = (input_ids != tokenizer_esm.pad_token_id).long()
            masked_positions = self.masked_positions_list[idx]
            targets = self.targets_list[idx]
            return input_ids, attention_mask, masked_positions, targets

    def collate_fn(batch):
        input_ids_list, attention_mask_list, masked_positions_list, targets_list = zip(*batch)
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        return input_ids, attention_mask, masked_positions_list, targets_list

    train_dataset = EpitopeDataset(train_masked_sequences, train_masked_positions, train_targets)
    eval_dataset = EpitopeDataset(eval_masked_sequences, eval_masked_positions, eval_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, eval_loader, filtered_train_sequences

train_loader, eval_loader, filtered_train_sequences = generate_train_test_loader(df, hla_type)

from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn

# Collect all targets for class weight computation
all_targets = []
for _, _, _, targets_list in train_loader:
    for targets in targets_list:
        all_targets.extend(targets)

all_targets = np.array(all_targets)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_targets),
    y=all_targets
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=logits.size(-1)).float()

        pt = torch.sum(targets_one_hot * probs, dim=-1)
        focal_weight = (1 - pt) ** self.gamma

        ce_loss = -torch.sum(targets_one_hot * torch.log(probs + 1e-6), dim=-1)
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

criterion_focal = FocalLoss(gamma=2.0, weight=class_weights, reduction='mean')

# -----------------------------------------
# Compute Background Frequencies f_i(a)
# -----------------------------------------
max_length = 15
position_counts = np.zeros((max_length, num_amino_acid_tokens), dtype=np.float64)
position_totals = np.zeros(max_length, dtype=np.float64)

# Count occurrences of amino acids at each position from filtered_train_sequences
# Note: position indexing matches how we tokenize: [CLS], seq..., [SEP]
# We'll consider positions [1 ... seq_length-2] for amino acids
for seq in filtered_train_sequences:
    # Tokenize without masking:
    tokenized = tokenizer_esm(seq, add_special_tokens=True, max_length=max_length, truncation=True, padding='max_length', return_tensors='np')
    input_ids = tokenized['input_ids'].squeeze(0)
    # Identify amino acid positions
    attention_mask = (input_ids != tokenizer_esm.pad_token_id)
    seq_length = attention_mask.sum()
    # Valid amino acid positions: 1 to seq_length-2
    for pos in range(1, seq_length - 1):
        token_id = input_ids[pos]
        if token_id in token_id_to_class_idx:
            aa_idx = token_id_to_class_idx[token_id]
            position_counts[pos, aa_idx] += 1
            position_totals[pos] += 1

# Compute frequencies f_i(a)
epsilon = 1e-6
f_matrix = (position_counts + epsilon) / (position_totals[:, None] + epsilon * num_amino_acid_tokens)

# Precompute the energy matrix: E(i,a) = -log(f_i(a))
energy_matrix = -np.log(f_matrix)
energy_matrix = torch.tensor(energy_matrix, dtype=torch.float).to(device)

# -----------------------------------------
# Combine Focal Loss + Energy in masked_cross_entropy_loss
# -----------------------------------------

def masked_cross_entropy_loss(outputs, targets_list, masked_positions_list):
    logits_list = []
    targets_flat = []
    positions_flat = []  # keep track of positions for energy calc

    for i in range(outputs.size(0)):
        logits = outputs[i]  # [seq_length, vocab_size]
        positions = masked_positions_list[i]
        targets = targets_list[i]
        for pos, target in zip(positions, targets):
            logits_masked = logits[pos].unsqueeze(0)  # [1, vocab_size]
            logits_list.append(logits_masked)
            targets_flat.append(target)
            positions_flat.append(pos)

    logits_flat = torch.cat(logits_list, dim=0)  # [num_masks, vocab_size]
    targets_flat = torch.tensor(targets_flat, device=logits_flat.device)
    positions_flat = torch.tensor(positions_flat, device=logits_flat.device)

    # Compute focal loss
    focal_loss = criterion_focal(logits_flat, targets_flat)

    # Compute expected energy:
    # p(a) = softmax(logits_flat)
    # E = sum_over_tokens sum_over_a p(a)*energy_matrix[pos,a]
    p = torch.softmax(logits_flat, dim=-1)
    # Gather energy terms:
    # positions_flat: shape [num_masks]
    # we need energy_matrix[positions_flat, :] -> shape [num_masks, num_amino_acid_tokens]
    energy_for_tokens = torch.gather(energy_matrix, 0, positions_flat.unsqueeze(-1).expand(-1, num_amino_acid_tokens))
    # energy_for_tokens shape: [num_masks, vocab_size]
    # Weighted by p: sum over a p(a)*energy
    token_energy = torch.sum(p * energy_for_tokens, dim=-1)  # [num_masks]
    energy_loss = torch.mean(token_energy)  # average over masked tokens

    # Total loss:
    total_loss = focal_loss + alpha * energy_loss

    predictions = torch.argmax(logits_flat, dim=-1)
    return total_loss, predictions, targets_flat

optimizer = optim.Adam(filter(lambda p: p.requires_grad, masking_model.parameters()), lr=lr_input, weight_decay=1e-3)

train_losses = []
eval_losses = []
train_accuracies = []
eval_accuracies = []

num_epochs = 10

print(f'Training Now for {hla_type}')

for epoch in range(num_epochs):
    print(hla_type)
    print(f'Epoch {epoch+1}/{num_epochs}')
    masking_model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in train_loader:
        input_ids, attention_mask, masked_positions_list, targets_list = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()
        outputs = masking_model(input_ids, attention_mask=attention_mask)
        loss, predictions, targets_flat = masked_cross_entropy_loss(outputs, targets_list, masked_positions_list)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (predictions == targets_flat).sum().item()
        total_samples += len(targets_flat)

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    print(f'Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}')

    # Evaluation
    masking_model.eval()
    total_loss_eval = 0.0
    total_correct_eval = 0
    total_samples_eval = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids, attention_mask, masked_positions_list, targets_list = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = masking_model(input_ids, attention_mask=attention_mask)
            loss, predictions, targets_flat = masked_cross_entropy_loss(outputs, targets_list, masked_positions_list)

            total_loss_eval += loss.item()
            total_correct_eval += (predictions == targets_flat).sum().item()
            total_samples_eval += len(targets_flat)

    avg_loss_eval = total_loss_eval / len(eval_loader)
    accuracy_eval = total_correct_eval / total_samples_eval

    eval_losses.append(avg_loss_eval)
    eval_accuracies.append(accuracy_eval)

    print(f'Eval Loss: {avg_loss_eval:.4f}, Eval Accuracy: {accuracy_eval:.4f}')
    
model_save_path = os.path.join(model_dir, f'model_epoch_{epoch+1}.pt')
torch.save(masking_model.state_dict(), model_save_path)

np.save(os.path.join(loss_dir, f'train_losses_{run_id}.npy'), np.array(train_losses))
np.save(os.path.join(loss_dir, f'eval_losses_{run_id}.npy'), np.array(eval_losses))
np.save(os.path.join(loss_dir, f'train_accuracies_{run_id}.npy'), np.array(train_accuracies))
np.save(os.path.join(loss_dir, f'eval_accuracies_{run_id}.npy'), np.array(eval_accuracies))
