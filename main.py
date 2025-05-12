import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

audio_path = "audio_and_txt_files/"
diagnosis_file = "patient_diagnosis.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_diag = pd.read_csv(diagnosis_file)
df_diag.columns = ["PatientID", "Diagnosis"]
diagnosis_map = {'URTI': 0, 'LRTI': 1, 'COPD': 2}
df_diag["Label"] = df_diag["Diagnosis"].map(diagnosis_map)

def extract_mfcc(file_path, max_len=100):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.flatten()

features, labels = [], []

for fname in os.listdir(audio_path):
    if fname.endswith(".wav"):
        try:
            patient_id = int(fname.split("_")[0])
            row = df_diag[df_diag["PatientID"] == patient_id]
            if row.empty or pd.isna(row["Label"].values[0]):
                continue
            label = int(row["Label"].values[0])
            mfcc = extract_mfcc(os.path.join(audio_path, fname))
            features.append(mfcc)
            labels.append(label)
        except:
            continue

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(input_size=X.shape[1], hidden_size=128, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_history = []

for epoch in range(20):
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
    print(f"Epoka {epoch+1}, Strata: {avg_loss:.4f}")
plt.plot(loss_history)
plt.title("Strata treningowa w czasie")
plt.xlabel("Epoka")
plt.ylabel("Strata")
plt.grid(True)
plt.show()
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"\n Dokładność na zbiorze testowym: {acc:.2f}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["URTI", "LRTI", "COPD"],
            yticklabels=["URTI", "LRTI", "COPD"])
plt.xlabel("Predykcja")
plt.ylabel("Rzeczywistość")
plt.title("Macierz pomyłek")
plt.show()
def add_noise(data, noise_level=0.02):
    noise = np.random.randn(*data.shape) * noise_level
    return data + noise

X_test_noisy = add_noise(X_test)
X_test_noisy_tensor = torch.tensor(X_test_noisy, dtype=torch.float32).to(device)
noisy_loader = DataLoader(TensorDataset(X_test_noisy_tensor, y_test_tensor), batch_size=32)

noisy_preds = []
with torch.no_grad():
    for batch_x, _ in noisy_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        noisy_preds.extend(predicted.cpu().numpy())

acc_noisy = accuracy_score(y_test, noisy_preds)
print(f"\n Dokładność na danych z szumem: {acc_noisy:.2f}")
torch.save(model.state_dict(), "mlp_model.pth")
print(" Model zapisany jako 'mlp_model.pth'")
