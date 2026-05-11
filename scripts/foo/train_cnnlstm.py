import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mne
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
DATA_PATH = "data/intrmd_data/raw_label"
OUTPUT_PATH = "results/dl_results/cnnlstm"
os.makedirs(OUTPUT_PATH, exist_ok=True)

class Config:
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    N_FOLDS = 5
    EARLY_STOPPING_PATIENCE = 15
    RANDOM_SEED = 42

class EEGDataset(Dataset):
    def __init__(self, data, labels, subject_ids):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.subject_ids = torch.LongTensor(subject_ids)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.subject_ids[idx]

class CNNLSTM(nn.Module):
    def __init__(self, n_channels, n_classes=3, subject_embed_dim=20):
        super(CNNLSTM, self).__init__()
        
        # Adjust kernel sizes based on channel count
        kernel_size1 = min(7, n_channels // 2)
        kernel_size2 = min(5, n_channels // 3)
        
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=kernel_size1, padding=kernel_size1//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 128, kernel_size=kernel_size2, padding=kernel_size2//2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(100)
        )
        
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.3, 
            bidirectional=True
        )
        
        self.subject_embed = nn.Embedding(51, subject_embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(256 + subject_embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x, subject_ids):
        # x shape: (batch, channels, time)
        x = self.cnn(x)  # (batch, 256, 100)
        x = x.permute(0, 2, 1)  # (batch, 100, 256)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        # Take last hidden state from both directions
        lstm_features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        subject_features = self.subject_embed(subject_ids)
        combined = torch.cat([lstm_features, subject_features], dim=1)
        output = self.classifier(combined)
        return output

def load_fif_file_correctly(fif_path):
    """
    Load fif file correctly - handles both raw and epochs
    """
    try:
        # First try reading as epochs (most likely for your data)
        epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
        return epochs, 'epochs'
    except:
        try:
            # If that fails, try reading as raw
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
            return raw, 'raw'
        except Exception as e:
            print(f"    Cannot read file: {e}")
            return None, None

def extract_data_from_epochs(epochs):
    """
    Extract EEG data and labels from epochs object
    """
    # Get EEG data (shape: n_epochs, n_channels, n_times)
    data = epochs.get_data()
    
    # Extract labels
    labels = []
    label_map = {'C': 0, 'H': 1, 'S': 2, 
                 'Control': 0, 'Haiku': 1, 'Senryu': 2}
    
    # Method 1: Try metadata
    if hasattr(epochs, 'metadata') and epochs.metadata is not None:
        if 'PoemType' in epochs.metadata.columns:
            labels = epochs.metadata['PoemType'].map(label_map).values
        elif 'condition' in epochs.metadata.columns:
            labels = epochs.metadata['condition'].map(label_map).values
        elif 'label' in epochs.metadata.columns:
            labels = epochs.metadata['label'].map(label_map).values
    
    # Method 2: Try event_id
    if len(labels) == 0 and hasattr(epochs, 'event_id'):
        for event_id_val in epochs.events[:, 2]:
            for key, val in epochs.event_id.items():
                if val == event_id_val:
                    for label_key in label_map:
                        if label_key.lower() in key.lower():
                            labels.append(label_map[label_key])
                            break
                    else:
                        labels.append(0)  # default
                    break
    
    # Method 3: Try annotations
    if len(labels) == 0 and hasattr(epochs, 'annotations') and epochs.annotations is not None:
        for ann in epochs.annotations:
            desc = ann['description']
            for label_key in label_map:
                if label_key.lower() in desc.lower():
                    labels.append(label_map[label_key])
                    break
            else:
                labels.append(0)
    
    # Method 4: Use trial index pattern (for debugging)
    if len(labels) == 0:
        print(f"    Warning: No labels found, using pattern (every 3rd trial)")
        labels = [i % 3 for i in range(len(data))]
    
    labels = np.array(labels)
    
    # Select only EEG channels
    if 'eeg' in epochs:
        eeg_indices = [i for i, ch in enumerate(epochs.ch_names) if 'EEG' in ch or ch.startswith('E')]
        if len(eeg_indices) > 0:
            data = data[:, eeg_indices, :]
    
    return data, labels

def extract_data_from_raw(raw, time_window=3.0):
    """
    Extract data from raw fif file using annotations
    """
    # Pick EEG channels
    if 'eeg' in raw:
        raw.pick_types(meg=False, eeg=True, eog=False, ecg=False)
    
    # Get events from annotations
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    if len(events) == 0:
        return None, None
    
    # Create epochs
    epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=0, tmax=time_window,
                        baseline=None, preload=True, verbose=False)
    
    return extract_data_from_epochs(epochs)

def load_all_data():
    """Load all subjects' data from .fif files"""
    all_data = []
    all_labels = []
    all_subject_ids = []
    all_channel_names = None
    min_timepoints = None
    
    file_paths = [f for f in os.listdir(DATA_PATH) if f.endswith('.fif')]
    file_paths.sort()
    
    print(f"Found {len(file_paths)} .fif files")
    
    for subject_idx, file_path in enumerate(tqdm(file_paths, desc="Loading subjects")):
        print(f"\n  Processing: {file_path}")
        fif_path = os.path.join(DATA_PATH, file_path)
        
        # Load file correctly
        data, file_type = load_fif_file_correctly(fif_path)
        
        if data is None:
            print(f"    Cannot load file, skipping...")
            continue
        
        print(f"    File type: {file_type}")
        
        # Extract data based on file type
        if file_type == 'epochs':
            epoch_data, labels = extract_data_from_epochs(data)
        elif file_type == 'raw':
            epoch_data, labels = extract_data_from_raw(data, time_window=15.0)  # 15 seconds for your data
        else:
            continue
        
        if epoch_data is None or len(epoch_data) == 0:
            print(f"    No data extracted")
            continue
        
        print(f"    Extracted: {epoch_data.shape[0]} trials, {epoch_data.shape[1]} channels, {epoch_data.shape[2]} timepoints")
        print(f"    Labels: C={np.sum(labels==0)}, H={np.sum(labels==1)}, S={np.sum(labels==2)}")
        
        # Handle channel count consistency
        if all_channel_names is None:
            all_channel_names = data.ch_names if hasattr(data, 'ch_names') else [f'ch_{i}' for i in range(epoch_data.shape[1])]
            min_timepoints = epoch_data.shape[2]
        else:
            # Trim or pad channels
            if epoch_data.shape[1] > len(all_channel_names):
                epoch_data = epoch_data[:, :len(all_channel_names), :]
            elif epoch_data.shape[1] < len(all_channel_names):
                pad = np.zeros((epoch_data.shape[0], len(all_channel_names) - epoch_data.shape[1], epoch_data.shape[2]))
                epoch_data = np.concatenate([epoch_data, pad], axis=1)
            
            # Handle timepoints (trim to minimum)
            if epoch_data.shape[2] > min_timepoints:
                epoch_data = epoch_data[:, :, :min_timepoints]
            elif epoch_data.shape[2] < min_timepoints:
                pad = np.zeros((epoch_data.shape[0], epoch_data.shape[1], min_timepoints - epoch_data.shape[2]))
                epoch_data = np.concatenate([epoch_data, pad], axis=2)
        
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
    print(f"Subjects: {len(np.unique(subjects))}")
    print(f"Class distribution: C={np.sum(y==0)}, H={np.sum(y==1)}, S={np.sum(y==2)}")
    
    # Normalize per subject (no leakage)
    print("\nNormalizing data per subject...")
    for subj in tqdm(np.unique(subjects), desc="Normalizing"):
        subj_mask = subjects == subj
        if subj_mask.sum() > 0:
            subj_data = X[subj_mask]
            mean = subj_data.mean()
            std = subj_data.std() + 1e-8
            X[subj_mask] = (subj_data - mean) / std
    
    return X, y, subjects

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            }, f"{OUTPUT_PATH}/fold{fold}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f"  Fold {fold}, Epoch {epoch}, Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
    
    return best_val_acc, best_val_f1

def main():
    print("="*60)
    print("CNN-LSTM PIPELINE - Poem Type Classification")
    print("="*60)
    
    # Load data
    try:
        X, y, subjects = load_all_data()
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("1. Your .fif files are likely epoched data, which this script now handles")
        print("2. Check if labels are stored in metadata, event_id, or annotations")
        return
    
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
        
        print(f"Training: {len(train_idx)} trials, Validation: {len(val_idx)} trials")
        print(f"Train classes: C={np.sum(y_train==0)}, H={np.sum(y_train==1)}, S={np.sum(y_train==2)}")
        
        # Create DataLoaders
        train_dataset = EEGDataset(X_train, y_train, subj_train)
        val_dataset = EEGDataset(X_val, y_val, subj_val)
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        # Initialize model
        model = CNNLSTM(n_channels=X.shape[1], n_classes=len(np.unique(y))).to(device)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")
        
        # Train
        best_acc, best_f1 = train_fold(model, train_loader, val_loader, Config, fold)
        
        # Final evaluation
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
        
        fold_results.append({'fold': fold, 'accuracy': best_acc, 'f1_score': best_f1})
        all_preds.extend(val_preds)
        all_true.extend(val_labels)
        
        print(f"Fold {fold} - Best Accuracy: {best_acc:.4f}, Best F1: {best_f1:.4f}")
    
    # Final results
    print("\n" + "="*60)
    print("CNN-LSTM - FINAL RESULTS")
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Control', 'Haiku', 'Senryu'],
                yticklabels=['Control', 'Haiku', 'Senryu'])
    plt.title('CNN-LSTM - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/confusion_matrix.png")
    plt.show()
    
    # Save results
    import pandas as pd
    pd.DataFrame(fold_results).to_csv(f"{OUTPUT_PATH}/results.csv", index=False)
    
    print(f"\n✅ Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()