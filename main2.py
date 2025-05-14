import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict

# Konfiguracja
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_path = "audio_and_txt_files/"
diagnosis_file = "patient_diagnosis.csv"
random_state = 42
torch.manual_seed(random_state)
np.random.seed(random_state)

# Wczytanie danych diagnostycznych
df_diag = pd.read_csv(diagnosis_file)
df_diag.columns = ["PatientID", "Diagnosis"]

# Automatyczne tworzenie mapowania diagnoz
label_encoder = LabelEncoder()
df_diag["Label"] = label_encoder.fit_transform(df_diag["Diagnosis"])
diagnosis_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

print("Mapowanie diagnoz:")
for disease, code in diagnosis_map.items():
    print(f"{disease}: {code}")


# Funkcja do ekstrakcji cech
def extract_features(file_path, max_len=100, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]

        # Dodatkowe cechy
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

        # Agregacja cech
        additional_features = np.concatenate([
            np.mean(chroma, axis=1),
            np.mean(spectral_contrast, axis=1),
            np.mean(tonnetz, axis=1),
            [librosa.feature.zero_crossing_rate(y)[0, 0]],
            [np.mean(librosa.feature.rms(y=y))]
        ])

        return mfcc.flatten(), additional_features.flatten()
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None


# Przygotowanie danych
patient_recordings = defaultdict(list)
audio_files = []

for fname in os.listdir(audio_path):
    if fname.endswith(".wav"):
        try:
            patient_id = int(fname.split("_")[0])
            patient_recordings[patient_id].append(fname)
            audio_files.append((patient_id, fname))
        except:
            continue

# Ekstrakcja cech
features_mfcc = []
features_extra = []
labels = []
patient_ids = []
failed_files = []

for patient_id, fname in tqdm(audio_files, desc="Processing audio files"):
    file_path = os.path.join(audio_path, fname)
    mfcc, extra = extract_features(file_path)

    if mfcc is not None and extra is not None:
        row = df_diag[df_diag["PatientID"] == patient_id]
        if not row.empty and not pd.isna(row["Label"].values[0]):
            features_mfcc.append(mfcc)
            features_extra.append(extra)
            labels.append(int(row["Label"].values[0]))
            patient_ids.append(patient_id)
        else:
            failed_files.append(fname)
    else:
        failed_files.append(fname)

print(f"\nSuccessfully processed {len(features_mfcc)} files")
print(f"Failed to process {len(failed_files)} files")

# Przygotowanie danych wejściowych
X_mfcc = np.array(features_mfcc)
X_extra = np.array(features_extra)
y = np.array(labels)
patient_ids = np.array(patient_ids)

# Normalizacja
scaler_mfcc = StandardScaler()
X_mfcc = scaler_mfcc.fit_transform(X_mfcc)

scaler_extra = StandardScaler()
X_extra = scaler_extra.fit_transform(X_extra)

X = np.hstack([X_mfcc, X_extra])

# Podział danych na train/val/test z zachowaniem pacjentów
unique_patient_ids = np.unique(patient_ids)

# Najpierw podział na train+val (80%) i test (20%)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
train_val_idx, test_idx = next(gss.split(unique_patient_ids, groups=unique_patient_ids))

# Następnie podział train_val na train (75%) i val (25%)
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
train_idx, val_idx = next(gss2.split(unique_patient_ids[train_val_idx],
                                     groups=unique_patient_ids[train_val_idx]))

# Mapowanie z powrotem do oryginalnych indeksów
train_patients = unique_patient_ids[train_val_idx][train_idx]
val_patients = unique_patient_ids[train_val_idx][val_idx]
test_patients = unique_patient_ids[test_idx]

train_mask = np.isin(patient_ids, train_patients)
val_mask = np.isin(patient_ids, val_patients)
test_mask = np.isin(patient_ids, test_patients)

X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]


# Dataset i DataLoader
class AudioDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment and torch.rand(1) > 0.5:
            noise = torch.randn_like(x) * 0.02
            x = x + noise
        return x, self.y[idx]


train_dataset = AudioDataset(X_train, y_train, augment=True)
val_dataset = AudioDataset(X_val, y_val)
test_dataset = AudioDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


# Model sieci neuronowej
class EnhancedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.3):
        super(EnhancedMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),

            nn.Linear(hidden_size // 4, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# Inicjalizacja modelu
input_size = X.shape[1]
hidden_size = 256
num_classes = len(diagnosis_map)
model = EnhancedMLP(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)


# Trening
def train_model():
    best_val_loss = float('inf')
    patience = 5
    no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(100):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Walidacja
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return train_losses, val_losses


# Uruchomienie treningu
train_losses, val_losses = train_model()

# Wykres strat
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Ocena na zbiorze testowym
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Znajdź faktycznie występujące klasy w danych testowych
present_labels = np.unique(all_labels)
target_names = [label_encoder.classes_[i] for i in present_labels]

# Metryki
test_acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)

print(f"\nTest Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(report)

# Macierz pomyłek
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Ocena na poziomie pacjenta (głosowanie większościowe)
patient_results = defaultdict(lambda: {'preds': [], 'label': None})

for pid, pred in zip(patient_ids[test_mask], all_preds):
    patient_results[pid]['preds'].append(pred)

for pid, label in zip(patient_ids[test_mask], y_test):
    patient_results[pid]['label'] = label

patient_preds = []
patient_true = []

for pid, data in patient_results.items():
    pred_counts = np.bincount(data['preds'], minlength=len(diagnosis_map))
    majority_pred = np.argmax(pred_counts)
    patient_preds.append(majority_pred)
    patient_true.append(data['label'])

# Uwzględnij tylko faktycznie występujące klasy
patient_present_labels = np.unique(patient_true)
patient_target_names = [label_encoder.classes_[i] for i in patient_present_labels]

# Metryki na poziomie pacjenta
patient_acc = accuracy_score(patient_true, patient_preds)
patient_cm = confusion_matrix(patient_true, patient_preds, labels=patient_present_labels)
patient_report = classification_report(patient_true, patient_preds,
                                       target_names=patient_target_names, zero_division=0)

print(f"\nPatient-Level Accuracy: {patient_acc:.4f}")
print("\nPatient-Level Classification Report:")
print(patient_report)

# Zapis modelu
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_mfcc': scaler_mfcc,
    'scaler_extra': scaler_extra,
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_classes': num_classes,
    'diagnosis_map': diagnosis_map,
    'label_encoder': label_encoder
}, "final_model_all_diseases.pth")

print("\nModel saved to 'final_model_all_diseases.pth'")
