import sys
import os
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel

# ----------------------------
#         ARGUMENTS
# ----------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description='Fine-tune ESM model for classification with varying parameters.')
    parser.add_argument('--lr_input', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--layers_to_unfreeze', type=int, default=11, help='Number of ESM layers to unfreeze')
    parser.add_argument('--size_of_train', type=float, default=0.1, help='Proportion of training data to sample')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--run', type=int, default=1, help='Run/trial ID')
    parser.add_argument('--loss', type=str, default='CrossEntropy', help='Loss function: CrossEntropy or BCE')
    parser.add_argument('--hla', type=str, default='HLA*02:01', help='HLA type (e.g., A*02:01, B*57:01)')
    parser.add_argument('--path', type=str, default='None', help='Path to pretrained model checkpoint (.pth)')
    parser.add_argument('--filter_20knM', type=str, default='False', help='Filter measurements >= 20,000 nM if True')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for classification')
    return parser.parse_args()

# ----------------------------
#       CONFIGURATION
# ----------------------------
def configure(args):
    run_id = (
        f'cESM_BA_lr_{args.lr_input}_LTU_{args.layers_to_unfreeze}_'
        f'ts_{args.size_of_train}_b_{args.batch_size}_r_{args.run}_loss_{args.loss}_'
        f'{args.hla}_20kf_{args.filter_20knM}'
    )

    config = {
        'MODEL_NAME': "facebook/esm2_t33_650M_UR50D",
        'MAX_LENGTH': 15,
        'LEARNING_RATE': args.lr_input,
        'WEIGHT_DECAY': 1e-5,
        'NUM_EPOCHS': 10,
        'BATCH_SIZE': args.batch_size,
        'RUN_ID': run_id,
        'HLA': args.hla,
        'FILTER_20K': args.filter_20knM,
        'PATH_TO_TRAIN': args.path,
        'LOSS_FUNCTION': args.loss,
        'NUM_CLASSES': args.num_classes,
        'layers_to_unfreeze': args.layers_to_unfreeze,
        'size_of_train':args.size_of_train
    }

    return config

# ----------------------------
#     SET SEEDS & DEVICE
# ----------------------------
def set_seeds_and_device():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device Used:", device, flush=True)
    torch.cuda.empty_cache()
    return device

# ----------------------------
#        LOSS FUNCTION
# ----------------------------
def get_criterion(loss_type, num_classes):
    if loss_type == "CrossEntropy":
        return nn.CrossEntropyLoss()
    elif loss_type == "BCE":
        if num_classes == 2:
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError("BCEWithLogitsLoss is typically used for binary classification.")
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

# ----------------------------
#        DATA FUNCTIONS
# ----------------------------
def aggregate_sequences(allele, filter_20k='False'):
    allele_to_path = {
        'HLA-B*57:01': '/global/home/users/sergiomar10/HLA-B5701_training_data.csv',
        'HLA-B5701': '/global/home/users/sergiomar10/HLA-B5701_training_data.csv',
        'HLA-B*40:02': '/global/home/users/sergiomar10/HLA-B4002_training_data.csv',
        'HLA-B4002': '/global/home/users/sergiomar10/HLA-B4002_training_data.csv',
        'HLA-A*29:02': '/global/home/users/sergiomar10/HLA-A2902_training_data.csv',
        'HLA-A2902': '/global/home/users/sergiomar10/HLA-A2902_training_data.csv',
        'HLA-C*05:01': '/global/home/users/sergiomar10/HLA-C0501_training_data.csv',
        'HLA-C0501': '/global/home/users/sergiomar10/HLA-C0501_training_data.csv',
        'HLA-B*15:01': '/global/home/users/sergiomar10/HLA-B1501_training_data.csv',
        'HLA-B1501': '/global/home/users/sergiomar10/HLA-B1501_training_data.csv',
        'HLA-B*27:05': '/global/home/users/sergiomar10/HLA-B2705_training_data.csv',
        'HLA-B2705': '/global/home/users/sergiomar10/HLA-B2705_training_data.csv',
        'HLA-A*03:01': '/global/home/users/sergiomar10/HLA-A0301_training_data.csv',
        'HLA-A0301': '/global/home/users/sergiomar10/HLA-A0301_training_data.csv',
        'HLA-A*01:01': '/global/home/users/sergiomar10/HLA-A0101_training_data.csv',
        'HLA-A0101': '/global/home/users/sergiomar10/HLA-A0101_training_data.csv',
        'HLA-A*02:01': '/global/home/users/sergiomar10/HLA-A0201_training_data.csv',
        'HLA-A0201': '/global/home/users/sergiomar10/HLA-A0201_training_data.csv',
        'HLA-A*11:01': '/global/home/users/sergiomar10/HLA-A1101_training_data.csv',
        'HLA-A1101': '/global/home/users/sergiomar10/HLA-A1101_training_data.csv'
    }

    csv_path = allele_to_path.get(allele)
    if not csv_path:
        raise ValueError(f"Unknown or unsupported allele: {allele}")

    data = pd.read_csv(csv_path, header=0)
    print(f'Measurements (including >=20,000 nM): {len(data)}', flush=True)

    if filter_20k == 'True':
        data = data[data['label'] < 4.3]  # 4.3 ~ log(20000)
        print(f'Measurements (filtered < 20,000 nM): {len(data)}', flush=True)

    return data

def split_data(aggregated, size_of_train=1.0, TESTING=False, num_classes=2):
    train_data, test_data = train_test_split(
        aggregated, test_size=0.2, random_state=10, shuffle=True
    )

    if num_classes == 2:
        # Filter out labels between 2 and 3 (inclusive)
        train_data = train_data[(train_data['label'] < 2) | (train_data['label'] > 3)].reset_index(drop=True)
        test_data = test_data[(test_data['label'] < 2) | (test_data['label'] > 3)].reset_index(drop=True)
        
        # Assign labels: <2 as 0 (Positive), >3 as 1 (Negative)
        train_data['label'] = (train_data['label'] > 3).astype(int)
        test_data['label'] = (test_data['label'] > 3).astype(int)

        # Assertions to ensure labels are correct
        assert train_data['label'].isin([0,1]).all(), "Train labels contain values other than 0 and 1."
        assert test_data['label'].isin([0,1]).all(), "Test labels contain values other than 0 and 1."
    
    elif num_classes > 2:
        # Existing multi-class handling logic
        bins = num_classes
        train_data['label'] = pd.qcut(train_data['label'], q=bins, labels=False)
        test_data['label'] = pd.qcut(test_data['label'], q=bins, labels=False)
    
    if TESTING:
        # In testing mode, use all data without additional sampling
        output_table = pd.concat([train_data, test_data])
    else:
        output_table = train_data
    
    return output_table, test_data


# ----------------------------
#         DATASET
# ----------------------------
class EpitopeDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    input_ids_list, attention_mask_list, labels_list = zip(*batch)
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    labels = torch.stack(labels_list)
    return input_ids, attention_mask, labels

# ----------------------------
#       DATALOADERS
# ----------------------------
def prepare_dataloaders(hla_type, tokenizer, batch_size=10, max_length=15, 
                        size_of_train=0.1, TESTING=False, filter_20k='False', num_classes=2):
    """
    Aggregate sequences for a specific HLA type, split into train/test,
    and return train/eval DataLoaders along with data tables.
    """
    print(f"Preparing data for {hla_type}...", flush=True)
    aggregated = aggregate_sequences(hla_type, filter_20k=filter_20k)
    output_table, test_data = split_data(aggregated, size_of_train=size_of_train, TESTING=TESTING, num_classes=num_classes)
    print(f"Training samples: {len(output_table)}, Test samples: {len(test_data)}", flush=True)

    train_dataset = EpitopeDataset(
        sequences=output_table["sequence"].values,
        labels=output_table["label"].values,
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_dataset = EpitopeDataset(
        sequences=test_data["sequence"].values,
        labels=test_data["label"].values,
        tokenizer=tokenizer,
        max_length=max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, eval_loader, output_table, test_data

# ----------------------------
#        MODEL CLASS
# ----------------------------
class ESMFineTune(nn.Module):
    def __init__(self, esm_model, num_classes=2):
        super(ESMFineTune, self).__init__()
        self.esm_model = esm_model
        self.config = esm_model.config
        self.dropout = nn.Dropout(0.3)
        self.classification_head = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, seq_len, hidden_size)
        
        attention_mask = attention_mask.unsqueeze(-1).float()
        masked_hidden_states = hidden_states * attention_mask
        sum_embeddings = masked_hidden_states.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1)
        
        pooled_output = sum_embeddings / sum_mask
        pooled_output = self.dropout(pooled_output)
        return self.classification_head(pooled_output)  # Raw logits

# ----------------------------
#         TRAINING
# ----------------------------
def train_model(model, criterion, optimizer, train_loader, eval_loader, device, num_epochs, loss_dir, model_dir, run_id, num_classes):
    train_losses = []
    eval_losses = []
    train_accuracies = []
    eval_accuracies = []
    train_f1s = []
    eval_f1s = []

    print("Starting training...", flush=True)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}", flush=True)
        model.train()
        train_loss_sum = 0.0
        total_train_samples = 0
        train_predictions = []
        train_targets = []

        for input_ids, attention_mask, targets in train_loader:
            input_ids, attention_mask, targets = (
                input_ids.to(device),
                attention_mask.to(device),
                targets.to(device)
            )

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss_value = criterion(outputs, targets)
            loss_value.backward()
            optimizer.step()

            bs_current = input_ids.size(0)
            train_loss_sum += loss_value.item() * bs_current
            total_train_samples += bs_current

            # Store predictions and targets for metrics
            if num_classes == 2:
                preds = torch.sigmoid(outputs).round()
            else:
                preds = torch.argmax(outputs, dim=1)
            train_predictions.extend(preds.detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

        avg_train_loss = train_loss_sum / total_train_samples
        train_losses.append(avg_train_loss)

        train_accuracy = accuracy_score(train_targets, train_predictions)
        train_f1 = f1_score(train_targets, train_predictions, average='weighted')
        train_accuracies.append(train_accuracy)
        train_f1s.append(train_f1)

        # ----------------------------
        #        EVALUATION
        # ----------------------------
        model.eval()
        eval_loss_sum = 0.0
        total_eval_samples = 0
        eval_predictions = []
        eval_targets_list = []

        with torch.no_grad():
            for input_ids, attention_mask, targets in eval_loader:
                input_ids, attention_mask, targets = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    targets.to(device)
                )
                outputs = model(input_ids, attention_mask=attention_mask)
                loss_value = criterion(outputs, targets)

                bs_current = input_ids.size(0)
                eval_loss_sum += loss_value.item() * bs_current
                total_eval_samples += bs_current

                # Store predictions and targets for metrics
                if num_classes == 2:
                    preds = torch.sigmoid(outputs).round()
                else:
                    preds = torch.argmax(outputs, dim=1)
                eval_predictions.extend(preds.cpu().numpy())
                eval_targets_list.extend(targets.cpu().numpy())

        avg_eval_loss = eval_loss_sum / total_eval_samples
        eval_losses.append(avg_eval_loss)

        eval_accuracy = accuracy_score(eval_targets_list, eval_predictions)
        eval_f1 = f1_score(eval_targets_list, eval_predictions, average='weighted')
        eval_accuracies.append(eval_accuracy)
        eval_f1s.append(eval_f1)

        print(
            f'Epoch {epoch+1}/{num_epochs}, '
            f'Train Loss: {avg_train_loss:.6f}, Eval Loss: {avg_eval_loss:.6f}, '
            f'Train Acc: {train_accuracy:.4f}, Eval Acc: {eval_accuracy:.4f}, '
            f'Train F1: {train_f1:.4f}, Eval F1: {eval_f1:.4f}',
            flush=True
        )

        # Save losses and metrics
        np.save(os.path.join(loss_dir, f'train_losses_{run_id}.npy'), np.array(train_losses))
        np.save(os.path.join(loss_dir, f'eval_losses_{run_id}.npy'), np.array(eval_losses))
        np.save(os.path.join(loss_dir, f'train_accuracies_{run_id}.npy'), np.array(train_accuracies))
        np.save(os.path.join(loss_dir, f'eval_accuracies_{run_id}.npy'), np.array(eval_accuracies))
        np.save(os.path.join(loss_dir, f'train_f1s_{run_id}.npy'), np.array(train_f1s))
        np.save(os.path.join(loss_dir, f'eval_f1s_{run_id}.npy'), np.array(eval_f1s))

        # Save model checkpoint
        model_path = os.path.join(model_dir, f"{run_id}_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'eval_loss': avg_eval_loss,
            'train_accuracy': train_accuracy,
            'eval_accuracy': eval_accuracy,
            'train_f1': train_f1,
            'eval_f1': eval_f1
        }, model_path)

    print("Training completed.", flush=True)
    return train_losses, eval_losses, train_accuracies, eval_accuracies, train_f1s, eval_f1s

# ----------------------------
#    FULL DATA PREDICTIONS
# ----------------------------
def make_full_predictions(model, tokenizer, loader, device, run_id, loss_dir, run_id_full, num_classes):
    print("Starting predictions on the full (train+test) set.", flush=True)
    model.eval()

    sequences_list = []
    predictions_list = []
    measured_list = []

    with torch.no_grad():
        for input_ids, attention_mask, targets in loader:
            input_ids, attention_mask, targets = (
                input_ids.to(device),
                attention_mask.to(device),
                targets.to(device)
            )
            outputs = model(input_ids, attention_mask=attention_mask)

            # Convert predictions to integers
            if num_classes == 2:
                preds = torch.sigmoid(outputs).round().long()
            else:
                preds = torch.argmax(outputs, dim=1)

            for i in range(input_ids.size(0)):
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu())
                seq = ''.join(tokens).replace('<pad>', '').replace('<cls>', '').replace('<eos>', '')
                sequences_list.append(seq)
                predictions_list.append(preds[i].item())
                measured_list.append(targets[i].item())

    predictions_finetuned_esm = pd.DataFrame({
        'sequence': sequences_list,
        'prediction': predictions_list,
        'measured': measured_list
    })

    # Save prediction outputs
    np.save(os.path.join(loss_dir, f'predictions_list_{run_id_full}.npy'), np.array(predictions_list))
    np.save(os.path.join(loss_dir, f'sequences_list_{run_id_full}.npy'), np.array(sequences_list))
    np.save(os.path.join(loss_dir, f'measured_list_{run_id_full}.npy'), np.array(measured_list))

    print("Done with predictions. All tasks completed.", flush=True)
    return predictions_finetuned_esm


# ----------------------------
#           MAIN
# ----------------------------
def main():
    # Parse arguments
    args = parse_arguments()

    # Configuration
    config = configure(args)

    # Set seeds and device
    device = set_seeds_and_device()

    if device.type == 'cpu':
        print("No GPU available. Exiting the script.", flush=True)
        sys.exit()


    # Get loss criterion
    criterion = get_criterion(config['LOSS_FUNCTION'], config['NUM_CLASSES'])

    # Load tokenizer and base model
    print("Loading tokenizer and base model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config['MODEL_NAME'])
    base_model = EsmModel.from_pretrained(config['MODEL_NAME'])

    # Initialize the fine-tuned model
    finetuned_model = ESMFineTune(base_model, num_classes=config['NUM_CLASSES']).to(device)

    LTU = config['layers_to_unfreeze']
    print(LTU, flush=True)
    # Unfreeze the last `layers_to_unfreeze` layers
    layers = list(finetuned_model.esm_model.named_children())
    for layer_name, layer in layers[-LTU:]:
        for param in layer.parameters():
            param.requires_grad = True
        print(f"Unfrozen layer: {layer_name}")

    # Always train the classification head
    for param in finetuned_model.classification_head.parameters():
        param.requires_grad = True

    # Load a previously saved checkpoint (if provided)
    if config['PATH_TO_TRAIN'] != "None":
        checkpoint = torch.load(config['PATH_TO_TRAIN'], map_location=device)
        finetuned_model.load_state_dict(checkpoint, strict=False)
        print("Loaded pretrained model checkpoint.", flush=True)

    # Prepare DataLoaders
    train_loader, eval_loader, train_data, test_data = prepare_dataloaders(
        hla_type=config['HLA'],
        tokenizer=tokenizer,
        batch_size=config['BATCH_SIZE'],
        max_length=config['MAX_LENGTH'],
        size_of_train=config['size_of_train'],
        filter_20k=config['FILTER_20K'],
        num_classes=config['NUM_CLASSES']
    )

    train_loader_full, eval_loader_full, train_data_full, test_data_full = prepare_dataloaders(
        hla_type=config['HLA'],
        tokenizer=tokenizer,
        batch_size=config['BATCH_SIZE'],
        max_length=config['MAX_LENGTH'],
        size_of_train=config['size_of_train'],
        TESTING=True,
        filter_20k=config['FILTER_20K'],
        num_classes=config['NUM_CLASSES']
    )

    # Setup optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, finetuned_model.parameters()),
        lr=config['LEARNING_RATE'],
        weight_decay=config['WEIGHT_DECAY']
    )

    # Prepare directories
    HLA_folder = config['HLA'].replace("*", "").replace(":", "")
    model_dir = f'/global/scratch/users/sergiomar10/models/BA_23012025/{HLA_folder}/'
    os.makedirs(model_dir, exist_ok=True)

    loss_dir = '/global/scratch/users/sergiomar10/losses/CA_ft/'
    os.makedirs(loss_dir, exist_ok=True)

    # Train the model
    train_losses, eval_losses, train_accuracies, eval_accuracies, train_f1s, eval_f1s = train_model(
        model=finetuned_model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        num_epochs=config['NUM_EPOCHS'],
        loss_dir=loss_dir,
        model_dir=model_dir,
        run_id=config['RUN_ID'],
        num_classes=config['NUM_CLASSES']
    )

    # Make predictions on the full dataset
    predictions_finetuned_esm = make_full_predictions(
        model=finetuned_model,
        tokenizer=tokenizer,
        loader=train_loader_full,
        device=device,
        run_id=config['RUN_ID'],
        loss_dir=loss_dir,
        run_id_full=config['RUN_ID'],
        num_classes=config['NUM_CLASSES']
    )

if __name__ == "__main__":
    main()
