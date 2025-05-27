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

audio_path = "audio_and_txt_files/"
diagnosis_file = "patient_diagnosis.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_diag = pd.read_csv(diagnosis_file, header=None)
df_diag.columns = ["PatientID", "Diagnosis"]

unique_diagnoses = sorted(df_diag["Diagnosis"].unique())
diagnosis_map = {diag: idx for idx, diag in enumerate(unique_diagnoses)}
df_diag["Label"] = df_diag["Diagnosis"].map(diagnosis_map)

print("Diagnosis mapping:", diagnosis_map)

def extract_mfcc(file_path, max_len=100):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def augment_mfcc(mfcc, noise_level=0.01, gain_db_range=(-6, 6), time_stretch_range=(0.9, 1.1)):
    augmented = mfcc.copy()
    noise = np.random.randn(*augmented.shape) * noise_level
    augmented += noise
    gain_db = np.random.uniform(*gain_db_range)
    gain = 10**(gain_db / 20)
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
    return augmented_final

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

label_to_features = {}
for mfcc, label in zip(all_mfccs, all_labels):
    if label not in label_to_features:
        label_to_features[label] = []
    label_to_features[label].append(mfcc)

target_count = 70
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
            base = mfcc_list[np.random.randint(0, len(mfcc_list))]
            augmented = augment_mfcc(base)
            selected.append(augmented)
    print(f"Label {label} after balance: {len(selected)}")
    for mfcc in selected:
        final_features.append(mfcc)
        final_labels.append(label)

X = np.array(final_features).astype(np.float32)  # shape (N, 40, 100)
y = np.array(final_labels)

# Add channel dimension for CNN: (N, 1, 40, 100)
X = X[:, np.newaxis, :, :]

test_size = 0.4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# CNN MODEL
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1, 40, 100) -> (16, 40, 100)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16, 20, 50)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> (32, 20, 50)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 10, 25)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 10 * 25, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc(x)
        return x

model = CNN(num_classes=len(diagnosis_map)).to(device)

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
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

plt.plot(loss_history)
plt.title("Training loss in time - test data size: "+str(100*test_size)+"%")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

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
plt.title("Confusion Matrix - test data size: "+str(100*test_size)+"% - Accuracy: "+str(100*round(acc,2))+"%")
plt.show()

torch.save(model.state_dict(), "cnn_model.pth")
print("Model saved as 'cnn_model.pth'")
