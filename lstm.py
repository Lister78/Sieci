import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --------- Parametry ---------
audio_path = "audio_and_txt_files/"
diagnosis_file = "patient_diagnosis.csv"
test_size = 0.8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_count = 70
num_epochs = 20
patience = 7  # Early stopping patience

# --------- Wczytywanie CSV ---------
df_diag = pd.read_csv(diagnosis_file, header=None)
df_diag.columns = ["PatientID", "Diagnosis"]

unique_diagnoses = sorted(df_diag["Diagnosis"].unique())
diagnosis_map = {diag: idx for idx, diag in enumerate(unique_diagnoses)}
df_diag["Label"] = df_diag["Diagnosis"].map(diagnosis_map)

print("Diagnosis mapping:", diagnosis_map)

# --------- Normalizacja MFCC ---------
def normalize_mfcc(mfcc):
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True) + 1e-9
    return (mfcc - mean) / std

# --------- Ekstrakcja MFCC z normalizacją ---------
def extract_mfcc(file_path, max_len=100):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    mfcc = normalize_mfcc(mfcc)
    return mfcc  # shape (40, max_len)

# --------- Augmentacja MFCC ---------
def augment_mfcc(mfcc, noise_level=0.01, gain_db_range=(-6, 6), time_stretch_range=(0.9, 1.1)):
    augmented = mfcc.copy()
    noise = np.random.randn(*augmented.shape) * noise_level
    augmented += noise
    gain_db = np.random.uniform(*gain_db_range)
    gain = 10 ** (gain_db / 20)
    augmented *= gain
    stretch_factor = np.random.uniform(*time_stretch_range)
    num_frames = int(augmented.shape[1] * stretch_factor)
    augmented_stretched = np.zeros((augmented.shape[0], num_frames))
    for i in range(augmented.shape[0]):
        augmented_stretched[i] = np.interp(
            np.linspace(0, 1, num_frames),
            np.linspace(0, 1, augmented.shape[1]),
            augmented[i]
        )
    if augmented_stretched.shape[1] < 100:
        pad_width = 100 - augmented_stretched.shape[1]
        augmented_final = np.pad(augmented_stretched, ((0, 0), (0, pad_width)), mode='constant')
    else:
        augmented_final = augmented_stretched[:, :100]
    augmented_final = normalize_mfcc(augmented_final)
    return augmented_final

# --------- Ładowanie danych ---------
all_mfccs = []
all_labels = []

for fname in os.listdir(audio_path):
    if fname.endswith(".wav"):
        try:
            patient_id = int(fname[:3])
            row = df_diag[df_diag["PatientID"] == patient_id]
            if row.empty or pd.isna(row["Label"].values[0]):
                continue
            label = int(row["Label"].values[0])
            mfcc = extract_mfcc(os.path.join(audio_path, fname))
            all_mfccs.append(mfcc)
            all_labels.append(label)
        except Exception as e:
            print(f"Error in file {fname}: {e}")
            continue

# Grupowanie MFCC wg etykiet
label_to_features = {}
for mfcc, label in zip(all_mfccs, all_labels):
    if label not in label_to_features:
        label_to_features[label] = []
    label_to_features[label].append(mfcc)

# --------- Balansowanie danych ---------
final_features = []
final_labels = []

for label, mfcc_list in label_to_features.items():
    count = len(mfcc_list)
    print(f"Label {label} before balance: {count}")
    if count >= target_count:
        sampled = np.random.choice(len(mfcc_list), target_count, replace=False)
        selected = [mfcc_list[i] for i in sampled]
    else:
        selected = mfcc_list.copy()
        while len(selected) < target_count:
            base = selected[np.random.randint(0, len(selected))]
            augmented = augment_mfcc(base)
            selected.append(augmented)
    print(f"Label {label} after balance: {len(selected)}")
    final_features.extend(selected)
    final_labels.extend([label] * len(selected))

X = np.array(final_features)  # shape (samples, 40, 100)
y = np.array(final_labels)

# --------- Podział na train/test ---------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

# --------- Przygotowanie tensora PyTorch ---------
X_train_tensor = torch.tensor(np.transpose(X_train, (0, 2, 1)), dtype=torch.float32).to(device)  # (batch, seq_len=100, features=40)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(np.transpose(X_test, (0, 2, 1)), dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# --------- Definicja modelu LSTM z bidirectional i dropout ---------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 bo bidirectional

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        # hn: (num_layers * num_directions, batch, hidden_size)
        # scalamy ostatnią warstwę obu kierunków
        out = torch.cat((hn[-2], hn[-1]), dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

model = LSTMClassifier(input_size=40, hidden_size=128, num_layers=3, num_classes=len(diagnosis_map), dropout=0.3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# --------- Trenowanie z early stopping ---------
loss_history = []
best_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # Scheduler krok
    scheduler.step(avg_loss)

    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_lstm_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping on epoch {epoch + 1}")
            break

plt.plot(loss_history)
plt.title(f"Training loss - test size: {test_size*100:.0f}%")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# --------- Ewaluacja ---------
model.load_state_dict(torch.load("best_lstm_model.pth"))
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_true.extend(batch_y.cpu().numpy())

acc = accuracy_score(all_true, all_preds)
cm = confusion_matrix(all_true, all_preds, labels=list(range(len(diagnosis_map))))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=unique_diagnoses,
            yticklabels=unique_diagnoses)
plt.xlabel("Prediction")
plt.ylabel("Reality")
plt.title(f"Confusion Matrix - test data size: "+str(100*test_size)+"%  - Accuracy: "+str(round(acc, 2))+"%")
plt.show()

print("Model saved as 'best_lstm_model.pth'")
