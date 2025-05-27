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

audio_path = "audio_and_txt_files/"  # path to directory with audio files
diagnosis_file = "patient_diagnosis.csv"  # CSV file with patient diagnoses
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available

df_diag = pd.read_csv(diagnosis_file, header=None)  # load diagnosis data
df_diag.columns = ["PatientID", "Diagnosis"]  # set column names

unique_diagnoses = sorted(df_diag["Diagnosis"].unique())  # get sorted list of unique diagnoses
diagnosis_map = {diag: idx for idx, diag in enumerate(unique_diagnoses)}  # create diagnosis-to-index mapping
df_diag["Label"] = df_diag["Diagnosis"].map(diagnosis_map)  # add numerical labels column

print("Diagnosis mapping:", diagnosis_map)

def extract_mfcc(file_path, max_len=100):
    y, sr = librosa.load(file_path, sr=None)  # load audio file
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # extract 40 MFCC features, default frame size 2048 samples
    
    if mfcc.shape[1] < max_len:  # if sequence shorter than max_len
        pad = max_len - mfcc.shape[1]  # calculate padding needed
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')  # pad with zeros
    else:
        mfcc = mfcc[:, :max_len]  # truncate to max_len if too long
    
    return mfcc  # return normalized MFCC matrix

def augment_mfcc(mfcc, noise_level=0.01, gain_db_range=(-6, 6), time_stretch_range=(0.9, 1.1)):
    augmented = mfcc.copy()  # create copy to avoid modifying original
    
    # Add Gaussian noise
    noise = np.random.randn(*augmented.shape) * noise_level  # generate noise
    augmented += noise  # apply noise to MFCCs
    
    # Apply random gain (volume change)
    gain_db = np.random.uniform(*gain_db_range)  # random dB value in range
    gain = 10**(gain_db / 20)  # convert dB to linear scale
    augmented *= gain  # apply gain
    
    # Time stretching
    stretch_factor = np.random.uniform(*time_stretch_range)  # random stretch factor
    num_frames = int(augmented.shape[1] * stretch_factor)  # calculate new length
    
    # Interpolate MFCCs for time stretching
    augmented_stretched = np.zeros((augmented.shape[0], num_frames))
    for i in range(augmented.shape[0]):  # process each MFCC coefficient
        augmented_stretched[i] = np.interp(
            np.linspace(0, 1, num_frames),  # new time points
            np.linspace(0, 1, augmented.shape[1]),  # original time points
            augmented[i]  # original values
        )
    
    # Ensure fixed length (100 frames)
    if augmented_stretched.shape[1] < 100:  # if too short
        pad_width = 100 - augmented_stretched.shape[1]
        augmented_final = np.pad(augmented_stretched, ((0, 0), (0, pad_width)), mode='constant')
    else:  # if too long
        augmented_final = augmented_stretched[:, :100]
    
    return augmented_final.flatten()  # return as 1D array

all_mfccs = []
all_labels = []

for fname in os.listdir(audio_path):  # iterate through all files in directory
    if fname.endswith(".wav"):  # process only WAV files
        try:
            patient_id = int(fname[:3])  # extract patient ID from first 3 chars
            row = df_diag[df_diag["PatientID"] == patient_id]  # find patient in diagnosis data
            
            # Skip if no diagnosis or missing label
            if row.empty or pd.isna(row["Label"].values[0]):
                continue
                
            label = int(row["Label"].values[0])  # get numerical label
            mfcc = extract_mfcc(os.path.join(audio_path, fname))  # extract MFCC features
            all_mfccs.append(mfcc)  # store features
            all_labels.append(label)  # store corresponding label
            
        except Exception as e:
            print(f"Error in file {fname}: {e}")  # show error but continue processing
            continue


# Create dictionary to group MFCCs by their labels
label_to_features = {}

# Iterate through all MFCC features and their corresponding labels
for mfcc, label in zip(all_mfccs, all_labels):
    # Initialize empty list for new labels
    if label not in label_to_features:
        label_to_features[label] = []
    
    # Add current MFCC to its label group
    label_to_features[label].append(mfcc)

# Balansing
target_count = 70  # desired number of samples per class
final_features = []  # will store balanced features
final_labels = []    # will store corresponding labels

for label, mfcc_list in label_to_features.items():  # process each class separately
    count = len(mfcc_list)
    print(f"Label {label} before balance: {count}")  # show pre-balance count
    
    # Handle classes with sufficient samples
    if count >= target_count:
        sampled = np.random.choice(len(mfcc_list), target_count, replace=False)  # random selection
        selected = [mfcc_list[i] for i in sampled]  # get chosen samples
    
    # Handle under-represented classes
    else:
        selected = mfcc_list.copy()  # start with all available samples
        while len(selected) < target_count:  # until we reach target
            base = mfcc_list[np.random.randint(0, len(mfcc_list))]  # pick random base sample
            augmented = augment_mfcc(base)  # create augmented version
            selected.append(augmented.reshape(40, 100))  # add to selection
    
    print(f"Label {label} after balance: {len(selected)}")  # show post-balance count
    
    # Add processed samples to final datasets
    for mfcc in selected:
        final_features.append(mfcc.flatten())  # flatten MFCCs
        final_labels.append(label)             # add corresponding label

# Convert lists to numpy arrays
X = np.array(final_features)  # feature matrix
y = np.array(final_labels)    # label vector

# Split into train/test sets
test_size = 0.4
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=test_size,
)

# Convert to PyTorch tensors and move to device (GPU/CPU)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Create PyTorch datasets and data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # training set
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)     # test set

train_loader = DataLoader(
    train_dataset, 
    batch_size=32,  # 32 samples per batch
    shuffle=True    # shuffle training data
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=32   # 32 samples per batch
)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()  # initialize parent class
        self.net = nn.Sequential(  # define network architecture
            nn.Linear(input_size, hidden_size),  # input layer
            nn.ReLU(),                           # activation function
            nn.Linear(hidden_size, hidden_size), # hidden layer
            nn.ReLU(),                           # activation function
            nn.Linear(hidden_size, num_classes)  # output layer
        )
        
    def forward(self, x):
        return self.net(x)  # forward pass through network

# Initialize model with specific parameters
model = MLP(
    input_size=X.shape[1],       # number of input features (flattened MFCCs)
    hidden_size=128,             # number of neurons in hidden layers
    num_classes=len(diagnosis_map)  # number of output classes
).to(device)  # move model to GPU if available

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # classification loss
optimizer = optim.Adam(
    model.parameters(),  # parameters to optimize
    lr=0.001             # learning rate
)

loss_history = []  # to track loss per epoch

for epoch in range(20):  # train for 20 epochs
    model.train()  # set model to training mode
    total_loss = 0  # reset epoch loss counter
    
    # Batch training loop
    for batch_x, batch_y in train_loader:  # process batches
        optimizer.zero_grad()  # clear previous gradients
        outputs = model(batch_x)  # forward pass
        loss = criterion(outputs, batch_y)  # compute loss
        loss.backward()  # backpropagation
        optimizer.step()  # update weights
        total_loss += loss.item()  # accumulate batch loss
    
    # Epoch statistics
    avg_loss = total_loss / len(train_loader)  # calculate average loss
    loss_history.append(avg_loss)  # store for tracking
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")  # show progress

plt.plot(loss_history)
plt.title("Training loss in time - test data size: "+str(100*test_size)+"%")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

model.eval()  # set model to evaluation mode (disables dropout etc.)
all_preds = []  # store all model predictions
all_true = []   # store all ground truth labels

with torch.no_grad():  # disable gradient calculation for evaluation
    for batch_x, batch_y in test_loader:  # process test data in batches
        outputs = model(batch_x)  # get model predictions
        _, predicted = torch.max(outputs, 1)  # get class with highest probability
        
        # Move tensors to CPU and convert to numpy
        all_preds.extend(predicted.cpu().numpy())  # collect predictions
        all_true.extend(batch_y.cpu().numpy())     # collect true labels

# Calculate overall accuracy
acc = accuracy_score(all_true, all_preds)  # compare predictions vs ground truth

cm = confusion_matrix(all_true, all_preds, labels=list(range(len(diagnosis_map))))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=unique_diagnoses,
            yticklabels=unique_diagnoses)
plt.xlabel("Prediction")
plt.ylabel("Reality")
plt.title("Confusion Matrix - test data size: "+str(100*test_size)+"% - Accuracy: "+str(100*round(acc,2))+"%")
plt.show()

torch.save(model.state_dict(), "mlp_model.pth")
print("Model saved as 'mlp_model.pth'")
