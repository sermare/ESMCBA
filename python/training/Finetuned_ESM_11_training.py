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

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("Loading Model")

model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer_esm = AutoTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name)

for param in model.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device Used:')
print(device)

model.to(device)

# Define the amino acid vocabulary
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

hlas = ['HLA-A*02:01']
# ['HLA-A*03:01',
#  'HLA-A*01:01',
#  'HLA-A*11:01',
#  'HLA-B*15:01',
#  'HLA-A*29:02',
#  'HLA-B*07:02',
#  'H2-Kb',
#  'HLA-A*68:02',
#  'HLA-A*02:03',
#  'HLA-A*02:06',
#  'HLA-B*44:03',
#  'HLA-A*24:02',
#  'H2-Db',
#  'HLA-B*08:01',
#  'HLA-A*31:01',
#  'HLA-B*44:02',
#  'HLA-A*02:02',
#  'HLA-B*40:01']

# Load your dataset (adjust the path as needed)
df = pd.read_csv('~/epitopes_iedb.csv', sep=',', header=2)

def generate_train_test_loader(dataset, hla):
    sequences_a1 = df[df['Name.1'].str.contains(hla)]['Name'].values
    sequences_a1 = np.unique(sequences_a1)
    
    print("Splitting data into training and evaluation sets...\n")
    train_sequences, eval_sequences = train_test_split(sequences_a1, test_size=0.2, random_state=42)
    
    filtered_train_sequences = [
        x for x in train_sequences if all(c in amino_acids for c in x)
    ]
    
    filtered_eval_sequences = [
        x for x in eval_sequences if all(c in amino_acids for c in x)
    ]
    
    filtered_train_sequences = [x for x in filtered_train_sequences if len(x) < 15]
    filtered_eval_sequences = [x for x in eval_sequences if len(x) < 15]
    
    print(f"Size of Train Sequences: {len(filtered_train_sequences)} \n Size of Test Sequences {len(filtered_eval_sequences)}")
    
    filtered_train_sequences = np.array(filtered_train_sequences)
    filtered_eval_sequences = np.array(filtered_eval_sequences)
    
    vocab = list(amino_acids) 
    
    aa_to_idx = {aa: idx for idx, aa in enumerate(vocab)}
    idx_to_aa = {idx: aa for aa, idx in aa_to_idx.items()}

    # hla_sequence = 'MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWEPSSQPTIPIVGIIAGLVLFGAVITGAVVAAVMWRRKSSDRKGGSYSQAASSDSAQGSDVSLTACKV'

    # filtered_train_sequences = [hla_sequence + x for x in filtered_train_sequences]
    # filtered_eval_sequences = [hla_sequence + x for x in filtered_eval_sequences]
    
    def mask_sequence_random_positions(sequence, num_augmentations=2,
                                       mask_prob=0.15, max_length=15):
        masked_sequences = []
        masked_positions_list = []
        targets_list = []
    
        seq_length = min(len(sequence), max_length)
        num_masks = max(1, int(seq_length * mask_prob))
    
        for _ in range(num_augmentations):
            positions = random.sample(range(seq_length), num_masks)
            masked_sequence = list(sequence[:seq_length])
            targets = []
    
            for pos in positions:
                targets.append(sequence[pos])
                masked_sequence[pos] = tokenizer_esm.mask_token
    
            masked_sequence = ''.join(masked_sequence)
            masked_sequences.append(masked_sequence)
            masked_positions_list.append(positions)
            targets_list.append(targets)
    
        return masked_sequences, masked_positions_list, targets_list
        
    def expand_sequences_for_masking(sequences, num_augmentations=5, mask_prob=0.2, max_length=15):
        expanded_sequences = []
        expanded_positions_list = []
        expanded_targets_list = []
        for seq in sequences:
            masked_seqs, masked_positions_list, targets_list = mask_sequence_random_positions(
                seq, num_augmentations=num_augmentations, mask_prob=mask_prob, max_length=max_length
            )
            expanded_sequences.extend(masked_seqs)
            expanded_positions_list.extend(masked_positions_list)
            expanded_targets_list.extend(targets_list)
        return expanded_sequences, expanded_positions_list, expanded_targets_list
    
    
    train_masked_sequences, train_masked_positions, train_targets = expand_sequences_for_masking(filtered_train_sequences)
    eval_masked_sequences, eval_masked_positions, eval_targets = expand_sequences_for_masking(filtered_eval_sequences)
    
    def filter_valid_targets(sequences, positions_list, targets_list, max_length=15):
        valid_sequences = []
        valid_positions_list = []
        valid_targets_list = []
        for seq, positions, targets in zip(sequences, positions_list, targets_list):
            seq_length = min(len(seq), max_length)
            valid_positions = []
            valid_targets = []
            for pos, target in zip(positions, targets):
                if target in aa_to_idx and 0 <= pos < seq_length:
                    valid_positions.append(pos)
                    valid_targets.append(aa_to_idx[target])
                else:
                    print(
                        f"Warning: Found invalid amino acid '{target}' or index '{pos}' out of bounds. Skipping."
                    )
            if valid_positions:
                valid_sequences.append(seq)
                valid_positions_list.append(valid_positions)
                valid_targets_list.append(valid_targets)
        return valid_sequences, valid_positions_list, valid_targets_list
    
    train_masked_sequences, train_masked_positions, train_targets = filter_valid_targets(
        train_masked_sequences, train_masked_positions, train_targets
    )
    eval_masked_sequences, eval_masked_positions, eval_targets = filter_valid_targets(
        eval_masked_sequences, eval_masked_positions, eval_targets
    )
    
    batch_size = 15
    
    class EpitopeDataset(Dataset):
        def __init__(self, sequences, positions_list, targets_list, tokenizer, max_length):
            self.sequences = sequences
            self.positions_list = positions_list
            self.targets_list = targets_list
            self.tokenizer = tokenizer
            self.max_length = max_length
    
        def __len__(self):
            return len(self.sequences)
    
        def __getitem__(self, idx):
            sequence = self.sequences[idx]
            input_ids = self.tokenizer(
                sequence,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            ).input_ids.squeeze(0)
            masked_positions = self.positions_list[idx]
            targets = self.targets_list[idx]
            return input_ids, masked_positions, targets
    
    
     # When tokenizing your sequences, add attention_mask
    def collate_fn(batch):
        input_ids_list, masked_positions_list, targets_list = zip(*batch)
        
        # Generate attention mask where 1 indicates the token is not a pad token, and 0 indicates a pad token.
        input_ids = torch.stack(input_ids_list)
        attention_mask = (input_ids != tokenizer_esm.pad_token_id).long()
    
        return input_ids, attention_mask, masked_positions_list, targets_list
    
    # Create instances of EpitopeDataset
    train_dataset = EpitopeDataset(
        sequences=train_masked_sequences,
        positions_list=train_masked_positions,
        targets_list=train_targets,
        tokenizer=tokenizer_esm,
        max_length=15
    )
    
    eval_dataset = EpitopeDataset(
        sequences=eval_masked_sequences,
        positions_list=eval_masked_positions,
        targets_list=eval_targets,
        tokenizer=tokenizer_esm,
        max_length=15
    )
    
    # Initialize DataLoaders with the dataset instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, eval_loader, train_masked_sequences


def masked_cross_entropy_loss(outputs, targets_list, masked_positions_list):
    
    logits_list = []
    targets_flat = []
    
    for i in range(len(outputs)):
        logits = outputs[i]  # Shape: [seq_length, num_classes]
        positions = masked_positions_list[i]
        targets = targets_list[i]
        logits_masked = logits[positions]  # Shape: [num_masks, num_classes]
        logits_list.append(logits_masked)
        targets_flat.extend(targets)
    
    logits_flat = torch.cat(logits_list, dim=0)  # Combine all the logits from the batch
    targets_flat = torch.tensor(targets_flat, device=logits_flat.device)

    # Compute loss
    loss = criterion(logits_flat, targets_flat)
    
    # Compute predictions by taking the argmax of the logits (the predicted class index)
    predictions = torch.argmax(logits_flat, dim=-1)  # Shape: [num_masks]
    
    return loss, predictions, targets_flat

vocab = list(amino_acids) 

aa_to_idx = {aa: idx for idx, aa in enumerate(vocab)}
idx_to_aa = {idx: aa for aa, idx in aa_to_idx.items()}

strategy = 'fine-tune'  # or 'fine-tune'

if strategy == 'fine-tune':
    # Implement Strategy 2
    for param in model.parameters():
        param.requires_grad = False

    # Corrected line: Access encoder layers directly
    for layer in model.encoder.layer[-11:]:
        for param in layer.parameters():
            param.requires_grad = True

    class ESMFineTune(nn.Module):
        def __init__(self, esm_model, amino_acid_indices):
            super(ESMFineTune, self).__init__()
            self.esm_model = esm_model
            self.fc_layers = nn.Sequential(
                nn.Linear(esm_model.config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5),  # Add dropout with 50% rate
                nn.Linear(512, len(amino_acid_indices))
            )


        def forward(self, input_ids, attention_mask=None):
            outputs = self.esm_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state
            logits = self.fc_layers(hidden_states)
            return logits

    masking_model = ESMFineTune(model, aa_to_idx).to(device)
else:
    raise ValueError("Invalid strategy selected.")

    
criterion = nn.CrossEntropyLoss()
num_epochs = 10
learning_rate = 1e-5

hla = 'HLA-A'
train_loader, eval_loader, _= generate_train_test_loader(df, hla)

# # Display the first 30 tokens from each sequence in the batch
for batch in train_loader:
    sequences = []  # To store sequences of tokens
    for sequence_tensor in batch[0]:  # Iterate over each sequence in Element 0
        sequence = []
        for idx, token_id in enumerate(sequence_tensor[:50]):  # Limit to first 30 tokens
            sequence.append(tokenizer_esm.id_to_token(int(token_id)))  # Convert to token
        sequences.append(sequence)

    # Print the sequences
    for i, seq in enumerate(sequences):
        print(f"Sequence {i+1}: {''.join(seq)}")
    break  # Only process the first batch


# Sequence 1: <cls>RTIILV<mask>Y<mask><eos><pad><pad><pad><pad>
# Sequence 2: <cls><mask>TTL<mask>TIST<eos><pad><pad><pad><pad>
# Sequence 3: <cls>FM<mask>QV<mask>SV<eos><pad><pad><pad><pad><pad>
# Sequence 4: <cls>T<mask>LGGKE<mask>Q<mask>LGV<eos>
# Sequence 5: <cls>PT<mask>DH<mask>PVV<eos><pad><pad><pad><pad>
# Sequence 6: <cls><mask>I<mask>LGGLNL<mask><eos><pad><pad><pad>
# Sequence 7: <cls>M<mask>S<mask>PQKIW<eos><pad><pad><pad><pad>
# Sequence 8: <cls>KLPPPP<mask><mask>A<eos><pad><pad><pad><pad>
# Sequence 9: <cls>ALG<mask>V<mask>AAL<eos><pad><pad><pad><pad>
# Sequence 10: <cls>YQAGI<mask>AA<mask><eos><pad><pad><pad><pad>
# Sequence 11: <cls>F<mask><mask>NYPF<mask>TSVKL<eos>
# Sequence 12: <cls><mask>MFLTDSN<mask><mask>KEV<eos>
# Sequence 13: <cls><mask>LRLL<mask>HQ<mask>L<eos><pad><pad><pad>
# Sequence 14: <cls>DIKD<mask>KEA<mask><eos><pad><pad><pad><pad>
# Sequence 15: <cls>EE<mask>L<mask>CGRL<eos><pad><pad><pad><pad>

for hla in hlas:

    print(f'{hla} modeling')
    
    train_losses = []
    eval_losses = []
    acuraccies = []
    
    model_dir = f'/global/scratch/users/sergiomar10/models/{hla}_masking/finetuned_ESM_{strategy}_hla_a_all'
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, masking_model.parameters()), lr=learning_rate, weight_decay=1e-3)
    
    for epoch in range(num_epochs):
        masking_model.train()
        train_loss = 0.0
        train_correct = 0  # Track correct predictions
        train_total = 0  # Track total masked positions
        total_train_samples = 0
        total_masked_positions = 0
    
        for input_ids, attention_mask, masked_positions_list, targets_list in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        
            optimizer.zero_grad()
            
            # Pass the attention mask to the model
            outputs = masking_model(input_ids, attention_mask)
            
            
            # Get loss and predictions
            loss, predictions, targets_flat = masked_cross_entropy_loss(outputs, targets_list, masked_positions_list)
            num_masked_positions = targets_flat.size(0)
            loss.backward()
            optimizer.step()
    
            # Compute accuracy for this batch
            correct_predictions = (predictions == targets_flat).sum().item()
            train_correct += correct_predictions
            train_total += targets_flat.size(0)
    
            batch_size = input_ids.size(0)
            train_loss += loss.item() * num_masked_positions
            total_masked_positions += num_masked_positions
            total_train_samples += batch_size
    
        average_train_loss = train_loss / total_masked_positions
        train_losses.append(average_train_loss)
    
        # Compute training accuracy
        train_accuracy = train_correct / train_total
    
        # Evaluation
        masking_model.eval()
        eval_loss = 0.0
        eval_correct = 0  # Track correct predictions for evaluation
        eval_total = 0  # Track total masked positions for evaluation
        total_eval_samples = 0
    
        with torch.no_grad():
            for input_ids, attention_mask, masked_positions_list, targets_list in eval_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                outputs = masking_model(input_ids, attention_mask)
                loss, predictions, targets_flat = masked_cross_entropy_loss(outputs, targets_list, masked_positions_list)
    
                # Compute evaluation accuracy
                correct_predictions = (predictions == targets_flat).sum().item()
                eval_correct += correct_predictions
                eval_total += targets_flat.size(0)
    
                batch_size = input_ids.size(0)
                eval_loss += loss.item() * batch_size
                total_eval_samples += batch_size
    
        average_eval_loss = eval_loss / total_eval_samples
        eval_losses.append(average_eval_loss)
    
        # Compute evaluation accuracy
        eval_accuracy = eval_correct / eval_total
    
        # Print losses and accuracy for the current epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {average_train_loss:.6f}, Eval Loss: {average_eval_loss:.6f}, '
              f'Train Accuracy: {train_accuracy:.4f}, Eval Accuracy: {eval_accuracy:.4f}')

        acuraccies.append([train_accuracy, eval_accuracy])
        
        # Save the model after each epoch
        model_path = os.path.join(model_dir, f"masking_model_epoch_{epoch+1}.pth")
        torch.save(masking_model.state_dict(), model_path)

    pd.DataFrame(train_losses).to_csv('/global/scratch/users/sergiomar10/losses/train_losses_hla_a_all.csv')
    pd.DataFrame(eval_losses).to_csv('/global/scratch/users/sergiomar10/losses/eval_losses_hla_a_all_not2onkys.csv')
    pd.DataFrame(acuraccies).to_csv('/global/scratch/users/sergiomar10/losses/acuraccies_hla_as_all_not2onlys_.csv')

    
    

        
