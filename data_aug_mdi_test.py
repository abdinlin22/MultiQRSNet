import torch
import os
import numpy as np
import glob
import random
from torch import nn
import torch.utils.data
from scipy.signal import butter, filtfilt

# -----------------------------
# Butterworth filter
# -----------------------------
def butter_filter(data, lowcut=None, highcut=None, fs=1000, order=5):
    nyquist = 0.5 * fs
    if lowcut and highcut:
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
    elif lowcut:
        low = lowcut / nyquist
        b, a = butter(order, low, btype='high')
    elif highcut:
        high = highcut / nyquist
        b, a = butter(order, high, btype='low')
    else:
        raise ValueError("Either lowcut or highcut must be specified")
    return filtfilt(b, a, data)

# -----------------------------
# MDI computation
# -----------------------------
def compute_mdi(signal):
    baseline = np.median(signal)
    signal = signal - baseline
    mid = len(signal) // 2
    window = signal[mid - 100: mid + 100]
    idx_max = np.argmax(np.abs(window))
    mdi_value = idx_max / len(window)
    return mdi_value

# -----------------------------
# Text file -> ECG array
# -----------------------------
def process_text_file_2(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    headers = lines[2].split()

    v1_index = headers.index('V1(22)')
    v2_index = headers.index('V2(23)')
    v3_index = headers.index('V3(24)')
    v4_index = headers.index('V4(25)')
    v5_index = headers.index('V5(26)')
    v6_index = headers.index('V6(27)')
    i_index = headers.index('I(110)')
    ii_index = headers.index('II(111)')
    iii_index = headers.index('III(112)')
    avl_index = headers.index('aVL(171)')
    avr_index = headers.index('aVR(172)')
    avf_index = headers.index('aVF(173)')

    data = np.zeros((12, 2500))
    counter = 0

    for line in lines[3:]:
        items = line.split()
        data[0, counter] = int(items[v1_index])
        data[1, counter] = int(items[v2_index])
        data[2, counter] = int(items[v3_index])
        data[3, counter] = int(items[v4_index])
        data[4, counter] = int(items[v5_index])
        data[5, counter] = int(items[v6_index])
        data[6, counter] = int(items[i_index])
        data[7, counter] = int(items[ii_index])
        data[8, counter] = int(items[iii_index])
        data[9, counter] = int(items[avl_index])
        data[10, counter] = int(items[avr_index])
        data[11, counter] = int(items[avf_index])
        counter += 1

    # If you want filtered data instead of raw, replace `data` with `filtered_data` below
    fs = 1000
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = butter_filter(data[i, :], lowcut=0.5, highcut=120, fs=fs, order=5)

    labels = np.array([])
    # return data, labels          # raw
    return data, labels            # keep identical to training script you showed

# -----------------------------
# Dataset
# -----------------------------
class CartoDataset(torch.utils.data.Dataset):
    def __init__(self, set_type='train', class_count=2, augment=False):
        self.set_type = set_type
        self.class_count = class_count
        self.augment = augment

        base_path = "/mnt/disk2/carto/train-test-folder/"
        self.lvot_files = []
        self.rvot_files = []
        self.lrvot_files = []

        if self.set_type == 'train':
            for sub_folder in ['train_cured', 'train_notcured']:
                lvot_folder = base_path + "LVOT/" + sub_folder
                rvot_folder = base_path + "RVOT/" + sub_folder

                self.lvot_files.extend(sorted(glob.glob(lvot_folder + '/**/*.txt', recursive=True)))
                self.rvot_files.extend(sorted(glob.glob(rvot_folder + '/**/*.txt', recursive=True)))

                if self.class_count == 3:
                    lrvot_folder = base_path + "RLVOT/" + sub_folder
                    self.lrvot_files.extend(sorted(glob.glob(lrvot_folder + '/**/*.txt', recursive=True)))
        else:
            lvot_folder = base_path + "LVOT/" + self.set_type
            rvot_folder = base_path + "RVOT/" + self.set_type

            self.lvot_files = sorted(glob.glob(lvot_folder + '/**/*.txt', recursive=True))
            self.rvot_files = sorted(glob.glob(rvot_folder + '/**/*.txt', recursive=True))

            if self.class_count == 3:
                lrvot_folder = base_path + "RLVOT/" + self.set_type
                self.lrvot_files = sorted(glob.glob(lrvot_folder + '/**/*.txt', recursive=True))

        print("lvot", len(self.lvot_files))
        print("rvot", len(self.rvot_files))
        if self.class_count == 3:
            print("lrvot", len(self.lrvot_files))

        if self.class_count == 2:
            self.files = self.lvot_files + self.rvot_files
            self.labels = [0] * len(self.lvot_files) + [1] * len(self.rvot_files)

        if self.class_count == 3:
            self.files = self.lvot_files + self.rvot_files + self.lrvot_files
            self.labels = (
                [0] * len(self.lvot_files)
                + [1] * len(self.rvot_files)
                + [2] * len(self.lrvot_files)
            )

    def __len__(self):
        return len(self.labels)

    def augment_signal(self, signal):
        if random.random() < 0.5:
            signal += np.random.normal(0, 0.01, signal.shape)
        if random.random() < 0.5:
            signal *= random.uniform(0.9, 1.1)
        if random.random() < 0.5:
            signal = np.roll(signal, random.randint(-50, 50))
        if random.random() < 0.2:
            signal = -signal
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            indices = np.round(np.arange(0, len(signal), factor)).astype(int)
            indices = indices[indices < len(signal)]
            signal = signal[indices]
        return signal

    def __getitem__(self, idx):
        txt = self.files[idx]
        label = self.labels[idx]
        data, _ = process_text_file_2(txt)

        # Apply augmentation only on training data if enabled
        if self.augment and self.set_type == 'train':
            for i in range(data.shape[0]):
                data[i] = self.augment_signal(data[i])

        # Compute MDI channel
        mdi_values = np.array([compute_mdi(data[i]) for i in range(12)])
        mdi_normalized = mdi_values.mean() / (np.max(mdi_values) + 1e-6)
        mdi_channel = np.full((1, data.shape[1]), mdi_normalized)
        data = np.concatenate([data, mdi_channel], axis=0)  # (13, N)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label), self.files[idx]

# -----------------------------
# Model
# -----------------------------
class LargerSimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = nn.Sequential(
            nn.Conv1d(13, 64, 3, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, 3, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(256, 128, 3, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 64, 3, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(4736, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# -----------------------------
# Evaluation only
# -----------------------------
def evaluate_saved_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model and load weights
    model = LargerSimpleNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Build test datasets and loaders (same as in training)
    test_cured_dataset = CartoDataset('test_cured', class_count=2, augment=False)
    test_not_cured_dataset = CartoDataset('test_notcured', class_count=2, augment=False)

    test_cured_loader = torch.utils.data.DataLoader(
        test_cured_dataset, batch_size=128, shuffle=False, drop_last=False
    )
    test_not_cured_loader = torch.utils.data.DataLoader(
        test_not_cured_dataset, batch_size=128, shuffle=False, drop_last=False
    )

    # --- Cured test ---
    correct_rvot_cured = 0
    correct_lvot_cured = 0
    correct_rlvot_cured = 0
    total_rvot_cured = 0
    total_lvot_cured = 0
    total_rlvot_cured = 0

    print("Cured Test:")
    with torch.no_grad():
        for batch in test_cured_loader:
            signals = batch[0].float().to(device)
            labels = batch[1].to(device)
            txt = batch[2]

            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)

            total_lvot_cured += (labels == 0).sum().item()
            total_rvot_cured += (labels == 1).sum().item()
            total_rlvot_cured += (labels == 2).sum().item()

            correct_lvot_cured += ((predicted == labels) * (labels == 0)).sum().item()
            correct_rvot_cured += ((predicted == labels) * (labels == 1)).sum().item()
            correct_rlvot_cured += ((predicted == labels) * (labels == 2)).sum().item()

    lvot_accuracy_cured = 100 * correct_lvot_cured / total_lvot_cured if total_lvot_cured > 0 else 0
    rvot_accuracy_cured = 100 * correct_rvot_cured / total_rvot_cured if total_rvot_cured > 0 else 0
    rlvot_accuracy_cured = 100 * correct_rlvot_cured / total_rlvot_cured if total_rlvot_cured > 0 else 0

    print(f'LVOT Test Cured Accuracy: {lvot_accuracy_cured:.2f}%')
    print(f'RVOT Test Cured Accuracy: {rvot_accuracy_cured:.2f}%')
    # print(f'RLVOT Test Cured Accuracy: {rlvot_accuracy_cured:.2f}%')

    # --- Not cured test ---
    correct_rvot_notcured = 0
    correct_lvot_notcured = 0
    correct_rlvot_notcured = 0
    total_rvot_notcured = 0
    total_lvot_notcured = 0
    total_rlvot_notcured = 0

    print("Not Cured Test:")
    with torch.no_grad():
        for batch in test_not_cured_loader:
            signals = batch[0].float().to(device)
            labels = batch[1].to(device)
            txt = batch[2]

            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)

            total_lvot_notcured += (labels == 0).sum().item()
            total_rvot_notcured += (labels == 1).sum().item()
            total_rlvot_notcured += (labels == 2).sum().item()

            correct_lvot_notcured += ((predicted == labels) * (labels == 0)).sum().item()
            correct_rvot_notcured += ((predicted == labels) * (labels == 1)).sum().item()
            correct_rlvot_notcured += ((predicted == labels) * (labels == 2)).sum().item()

    lvot_accuracy_not_cured = 100 * correct_lvot_notcured / total_lvot_notcured if total_lvot_notcured > 0 else 0
    rvot_accuracy_not_cured = 100 * correct_rvot_notcured / total_rvot_notcured if total_rvot_notcured > 0 else 0
    rlvot_accuracy_not_cured = 100 * correct_rlvot_notcured / total_rlvot_notcured if total_rlvot_notcured > 0 else 0

    print(f'LVOT Test Not Cured Accuracy: {lvot_accuracy_not_cured:.2f}%')
    print(f'RVOT Test Not Cured Accuracy: {rvot_accuracy_not_cured:.2f}%')
    # print(f'RLVOT Test Not Cured Accuracy: {rlvot_accuracy_not_cured:.2f}%')

    # Total accuracy over both test sets
    total_correct = (
        correct_lvot_cured + correct_rvot_cured + correct_rlvot_cured +
        correct_lvot_notcured + correct_rvot_notcured + correct_rlvot_notcured
    )
    total_samples = (
        total_lvot_cured + total_rvot_cured + total_rlvot_cured +
        total_lvot_notcured + total_rvot_notcured + total_rlvot_notcured
    )
    test_total_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0

    print(f'Total Test Accuracy (cured + not cured): {test_total_accuracy:.2f}%')

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Replace this with the path to your saved .pth file
    MODEL_PATH = "mdi-newdata-2class-notcuredchanged/model_7-98.21782178217822.pth"
    evaluate_saved_model(MODEL_PATH)
