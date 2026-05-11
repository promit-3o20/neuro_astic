"""
EEGNet Pipeline for Poem Type Classification
Handles .fif files that may be raw or already epoched
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import mne
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================= CONFIGURATION =================
class Config:
    DATA_PATH = "data/intrmd_data/raw_label"
    OUTPUT_PATH = "results/dl_results/eegnet"
    
    # EEG parameters
    MONTAGE_NAME = 'biosemi64'
    REFERENCE = 'average'
    
    # Data parameters
    SAMPLE_RATE = 512  # Hz
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    N_FOLDS = 5
    EARLY_STOPPING_PATIENCE = 15
    RANDOM_SEED = 42

os.makedirs(Config.OUTPUT_PATH, exist_ok=True)

# Set random seeds
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)

# ================= DATASET CLASS =================
class PoemTypeDataset(Dataset):
    def __init__(self, eeg_data, labels, subject_ids):
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)
        self.subject_ids = torch.LongTensor(subject_ids)
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx], self.subject_ids[idx]

# ================= EEGNET MODEL =================
class EEGNet(nn.Module):
    def __init__(self, n_channels, n_classes=3, subject_embed_dim=20):
        super(EEGNet, self).__init__()
        
        # Adjust kernel size based on input length
        kernel_size = min(64, n_channels)
        padding = kernel_size // 2
        
        # Temporal convolution
        self.conv1 = nn.Conv2d(1, 16, (1, kernel_size), padding=(0, padding))
        self.bn1 = nn.BatchNorm2d(16)
        
        # Depthwise convolution (spatial filtering)
        self.depthwise = nn.Conv2d(16, 32, (n_channels, 1), groups=16)
        self.bn2 = nn.BatchNorm2d(32)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 8))
        self.dropout1 = nn.Dropout(0.5)
        
        # Separable convolution
        self.separable = nn.Conv2d(32, 32, (1, 16))
        self.bn3 = nn.BatchNorm2d(32)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.5)
        
        # Global pooling
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Subject embedding
        self.subject_embed = nn.Embedding(51, subject_embed_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32 + subject_embed_dim, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x, subject_ids):
        x = x.unsqueeze(1)  # (batch, 1, channels, time)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        x = self.separable(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        
        subject_features = self.subject_embed(subject_ids)
        combined = torch.cat([x, subject_features], dim=1)
        
        output = self.classifier(combined)
        return output

# ================= DATA LOADING FUNCTIONS =================
def load_fif_file(fif_path):
    """Load fif file - works for both raw and epoched data"""
    try:
        # First try reading as raw
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        return raw, 'raw'
    except:
        try:
            # Try reading as epochs
            epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
            return epochs, 'epochs'
        except:
            try:
                # Try reading as evoked
                evoked = mne.read_evokeds(fif_path, verbose=False)
                return evoked, 'evoked'
            except Exception as e:
                print(f"    Cannot read file: {e}")
                return None, None

def extract_eeg_data_and_labels(data, file_type, config):
    """Extract EEG data and labels from loaded fif object"""
    
    label_map = {'C': 0, 'H': 1, 'S': 2, 
                 'Control': 0, 'Haiku': 1, 'Senryu': 2,
                 'poemtype_C': 0, 'poemtype_H': 1, 'poemtype_S': 2}
    
    if file_type == 'raw':
        # Raw data - need to extract epochs from annotations
        print(f"    Processing as raw data...")
        
        # Pick only EEG channels
        if 'eeg' in data:
            data.pick_types(meg=False, eeg=True, eog=False, ecg=False)
        
        # Get annotations
        annotations = data.annotations
        if annotations is None or len(annotations) == 0:
            print(f"    No annotations found in raw file")
            return None, None
        
        # Extract events from annotations
        events, event_id = mne.events_from_annotations(data, verbose=False)
        
        # Create epochs (use full duration or specified window)
        tmax = data.times[-1]
        epochs = mne.Epochs(data, events, event_id=event_id,
                           tmin=0, tmax=tmax,
                           baseline=None, preload=True, verbose=False)
        
        epoch_data = epochs.get_data()
        
        # Extract labels from event descriptions
        labels = []
        for event in events:
            event_desc = list(event_id.keys())[list(event_id.values()).index(event[2])]
            label = None
            for key in label_map:
                if key.lower() in event_desc.lower():
                    label = label_map[key]
                    break
            if label is None:
                # Try to extract from description
                desc_lower = event_desc.lower()
                if 'c' in desc_lower or 'control' in desc_lower:
                    label = 0
                elif 'h' in desc_lower or 'haiku' in desc_lower:
                    label = 1
                elif 's' in desc_lower or 'senryu' in desc_lower:
                    label = 2
                else:
                    label = 0
            labels.append(label)
        
        return epoch_data, np.array(labels)
        
    elif file_type == 'epochs':
        # Already epoched data
        print(f"    Processing as epoched data...")
        
        # Pick only EEG channels
        if 'eeg' in data:
            data.pick_types(meg=False, eeg=True, eog=False, ecg=False)
        
        epoch_data = data.get_data()
        
        # Extract labels from metadata or event_id
        labels = []
        
        # Try to get labels from metadata
        if hasattr(data, 'metadata') and data.metadata is not None:
            if 'PoemType' in data.metadata.columns:
                labels = data.metadata['PoemType'].map(label_map).values
            elif 'condition' in data.metadata.columns:
                labels = data.metadata['condition'].map(label_map).values
            elif 'label' in data.metadata.columns:
                labels = data.metadata['label'].map(label_map).values
        
        # If metadata didn't work, try from event_id
        if len(labels) == 0 and hasattr(data, 'event_id'):
            # Map event_id keys to labels
            for i, event_id_val in enumerate(data.events[:, 2]):
                for key in data.event_id:
                    if data.event_id[key] == event_id_val:
                        desc = key
                        label = None
                        for label_key in label_map:
                            if label_key.lower() in desc.lower():
                                label = label_map[label_key]
                                break
                        if label is None:
                            label = 0
                        labels.append(label)
                        break
        
        # If still no labels, try from annotations in epochs
        if len(labels) == 0 and hasattr(data, 'annotations') and data.annotations is not None:
            for ann in data.annotations:
                desc = ann['description']
                label = None
                for key in label_map:
                    if key.lower() in desc.lower():
                        label = label_map[key]
                        break
                if label is None:
                    label = 0
                labels.append(label)
        
        # Final fallback - use trial indices modulo 3 (for testing)
        if len(labels) == 0:
            print(f"    Warning: No labels found, using trial index modulo 3")
            labels = [i % 3 for i in range(len(epoch_data))]
        
        return epoch_data, np.array(labels)
        
    elif file_type == 'evoked':
        print(f"    Skipping evoked data (not suitable for trial classification)")
        return None, None
    
    return None, None

def load_all_data_mne(config):
    """Load all subjects from .fif files"""
    all_data = []
    all_labels = []
    all_subject_ids = []
    all_channel_names = None
    min_timepoints = float('inf')
    
    # Find all .fif files
    fif_files = [f for f in os.listdir(config.DATA_PATH) if f.endswith('.fif')]
    fif_files.sort()
    
    print(f"Found {len(fif_files)} .fif files")
    
    for subject_idx, fif_file in enumerate(tqdm(fif_files, desc="Processing subjects")):
        print(f"\nSubject {subject_idx+1}: {fif_file}")
        fif_path = os.path.join(config.DATA_PATH, fif_file)
        
        # Load the file
        data, file_type = load_fif_file(fif_path)
        
        if data is None:
            print(f"  Failed to load file")
            continue
        
        print(f"  File type: {file_type}")
        
        # Extract EEG data and labels
        epoch_data, labels = extract_eeg_data_and_labels(data, file_type, config)
        
        if epoch_data is None or len(epoch_data) == 0:
            print(f"  No valid data extracted")
            continue
        
        print(f"  Extracted: {epoch_data.shape[0]} trials, {epoch_data.shape[1]} channels, {epoch_data.shape[2]} timepoints")
        
        # Check for consistent channel count
        if all_channel_names is None:
            all_channel_names = data.ch_names if hasattr(data, 'ch_names') else [f'ch_{i}' for i in range(epoch_data.shape[1])]
            min_timepoints = epoch_data.shape[2]
        else:
            # Handle channel mismatch
            if epoch_data.shape[1] != len(all_channel_names):
                print(f"  Channel mismatch: {epoch_data.shape[1]} vs {len(all_channel_names)}")
                # Try to find common channels
                current_ch_names = data.ch_names if hasattr(data, 'ch_names') else [f'ch_{i}' for i in range(epoch_data.shape[1])]
                common_ch = [ch for ch in all_channel_names if ch in current_ch_names]
                if len(common_ch) > 0:
                    ch_idx = [current_ch_names.index(ch) for ch in common_ch]
                    epoch_data = epoch_data[:, ch_idx, :]
                    print(f"  Using {len(common_ch)} common channels")
                else:
                    print(f"  No common channels, trimming to {len(all_channel_names)}")
                    epoch_data = epoch_data[:, :len(all_channel_names), :]
            
            # Handle timepoint mismatch (trim or pad)
            if epoch_data.shape[2] != min_timepoints:
                if epoch_data.shape[2] > min_timepoints:
                    epoch_data = epoch_data[:, :, :min_timepoints]
                else:
                    # Pad with zeros
                    pad_width = ((0, 0), (0, 0), (0, min_timepoints - epoch_data.shape[2]))
                    epoch_data = np.pad(epoch_data, pad_width, mode='constant')
        
        all_data.append(epoch_data)
        all_labels.append(labels)
        all_subject_ids.append(np.full(len(epoch_data), subject_idx))
    
    if not all_data:
        raise ValueError("No data loaded. Check your .fif files.")
    
    # Concatenate all subjects
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    subjects = np.concatenate(all_subject_ids, axis=0)
    
    print(f"\n{'='*50}")
    print("DATA SUMMARY")
    print(f"{'='*50}")
    print(f"Total trials: {X.shape[0]}")
    print(f"EEG channels: {X.shape[1]}")
    print(f"Time points: {X.shape[2]}")
    print(f"Total subjects: {len(np.unique(subjects))}")
    print(f"Class distribution: C={np.sum(y==0)}, H={np.sum(y==1)}, S={np.sum(y==2)}")
    
    # Normalize per subject (no leakage between subjects)
    print("\nNormalizing data per subject...")
    for subj in tqdm(np.unique(subjects), desc="Normalizing"):
        subj_mask = subjects == subj
        if subj_mask.sum() > 0:
            subj_data = X[subj_mask]
            mean = subj_data.mean()
            std = subj_data.std() + 1e-8
            X[subj_mask] = (subj_data - mean) / std
    
    return X, y, subjects, all_channel_names

# ================= TRAINING FUNCTION =================
def train_fold(model, train_loader, val_loader, config, fold):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    train_losses = []
    val_accs = []
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for batch_data, batch_labels, batch_subjects in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            batch_subjects = batch_subjects.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data, batch_subjects)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_data, batch_labels, batch_subjects in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                batch_subjects = batch_subjects.to(device)
                outputs = model(batch_data, batch_subjects)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_accs.append(val_acc)
        train_losses.append(train_loss / len(train_loader))
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1
            }, f"{config.OUTPUT_PATH}/fold{fold}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f"  Fold {fold}, Epoch {epoch}, Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
    
    return best_val_acc, best_val_f1, train_losses, val_accs

# ================= MAIN PIPELINE =================
def main():
    print("="*60)
    print("EEGNet PIPELINE - Poem Type Classification")
    print("Handles both raw and epoched .fif files")
    print("="*60)
    
    # Load data with MNE
    try:
        X, y, subjects, channel_names = load_all_data_mne(Config)
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that your .fif files contain EEG data")
        print("2. Verify that labels are stored in annotations, metadata, or event_id")
        print("3. Try loading a single file manually with: mne.read_epochs(fif_file)")
        return
    
    # Save channel info
    with open(f"{Config.OUTPUT_PATH}/channels_used.txt", 'w') as f:
        f.write(f"Number of EEG channels: {len(channel_names)}\n")
        f.write("Channel names:\n")
        for ch in channel_names:
            f.write(f"  - {ch}\n")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED)
    fold_results = []
    all_preds = []
    all_true = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/{Config.N_FOLDS}")
        print(f"{'='*40}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        subj_train, subj_val = subjects[train_idx], subjects[val_idx]
        
        print(f"Training trials: {len(train_idx)}, Validation trials: {len(val_idx)}")
        print(f"Train classes: C={np.sum(y_train==0)}, H={np.sum(y_train==1)}, S={np.sum(y_train==2)}")
        
        # Create DataLoaders
        train_dataset = PoemTypeDataset(X_train, y_train, subj_train)
        val_dataset = PoemTypeDataset(X_val, y_val, subj_val)
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        # Initialize model
        model = EEGNet(n_channels=X.shape[1], n_classes=len(np.unique(y))).to(device)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")
        
        # Train
        best_acc, best_f1, train_losses, val_accs = train_fold(model, train_loader, val_loader, Config, fold)
        
        # Evaluate on validation set
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_data, batch_labels, batch_subjects in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                batch_subjects = batch_subjects.to(device)
                outputs = model(batch_data, batch_subjects)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())
        
        fold_results.append({
            'fold': fold, 
            'accuracy': best_acc, 
            'f1_score': best_f1,
            'n_trials_train': len(train_idx),
            'n_trials_val': len(val_idx)
        })
        all_preds.extend(val_preds)
        all_true.extend(val_labels)
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_accs, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.title(f'EEGNet - Fold {fold} Training')
        
        plt.subplot(1, 2, 2)
        fold_accs = [r['accuracy'] for r in fold_results]
        plt.plot(fold_accs, marker='o', label='Fold Accuracy')
        plt.axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{Config.OUTPUT_PATH}/fold{fold}_curves.png")
        plt.close()
        
        print(f"Fold {fold} - Best Accuracy: {best_acc:.4f}, Best F1: {best_f1:.4f}")
    
    # Final results
    print("\n" + "="*60)
    print("EEGNet - FINAL RESULTS")
    print("="*60)
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc = np.std([r['accuracy'] for r in fold_results])
    mean_f1 = np.mean([r['f1_score'] for r in fold_results])
    
    print(f"Mean CV Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"Mean F1 Score: {mean_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=['Control', 'Haiku', 'Senryu']))
    
    # Confusion matrix
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Control', 'Haiku', 'Senryu'],
                yticklabels=['Control', 'Haiku', 'Senryu'])
    plt.title('EEGNet - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{Config.OUTPUT_PATH}/confusion_matrix.png")
    plt.show()
    
    # Save results
    pd.DataFrame(fold_results).to_csv(f"{Config.OUTPUT_PATH}/results.csv", index=False)
    
    # Save summary
    with open(f"{Config.OUTPUT_PATH}/summary.txt", 'w') as f:
        f.write("="*60 + "\n")
        f.write("EEGNet PIPELINE RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Number of EEG channels: {X.shape[1]}\n")
        f.write(f"Total trials: {X.shape[0]}\n")
        f.write(f"Subjects: {len(np.unique(subjects))}\n")
        f.write(f"Mean CV Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
        f.write(f"Mean F1 Score: {mean_f1:.4f}\n\n")
        f.write("Per-fold results:\n")
        for r in fold_results:
            f.write(f"  Fold {r['fold']}: Acc={r['accuracy']:.4f}, F1={r['f1_score']:.4f}\n")
    
    print(f"\n✅ Results saved to {Config.OUTPUT_PATH}")

if __name__ == "__main__":
    main()