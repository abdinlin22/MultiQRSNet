import torch
import os
import numpy as np
import glob
import random
from torch import nn, optim
import torch.utils.data
from scipy.signal import butter, filtfilt

# Butterworth filter
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

# MDI hesaplama
def compute_mdi(signal):
    baseline = np.median(signal)
    signal = signal - baseline
    mid = len(signal) // 2
    window = signal[mid - 100: mid + 100]
    idx_max = np.argmax(np.abs(window))
    mdi_value = idx_max / len(window)
    return mdi_value


# Function to process text file
def process_text_file_2(file_path):
    # Read the file skipping the first 2 rows
    
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    headers = lines[2].split()

    # Find the indexes
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
        # Extract data based on their indexes
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

    # Apply BS ECG filter: high-pass 0.5 Hz and low-pass 120 Hz
    filtered_data = np.zeros_like(data)
    fs = 1000  # Assuming a sampling frequency of 1000 Hz

    for i in range(data.shape[0]):
        filtered_data[i, :] = butter_filter(data[i, :], lowcut=0.5, highcut=120, fs=fs, order=5)

    labels = np.array([])  # Placeholder or keep as None if not used
    return data, labels
    #return filtered_data, labels

# Dataset
class CartoDataset(torch.utils.data.Dataset):
    def __init__(self, set_type='train', class_count=2, augment=False):
        self.set_type = set_type  # Değişken adı 'set_type' olarak güncellendi, 'train' sadece train olup olmadığını belirlemede kullanılacak.
        self.class_count = class_count
        self.augment = augment  # Enable/Disable augmentation

        base_path = "/mnt/disk2/carto/train-test-folder/"
        self.lvot_files = []
        self.rvot_files = []
        self.lrvot_files = []

        if self.set_type == 'train':
            # 'train' seti için 'train_cured' ve 'train_notcured' klasörlerini birleştir
            for sub_folder in ['train_cured', 'train_notcured']:
                lvot_folder = base_path + "LVOT/" + sub_folder
                rvot_folder = base_path + "RVOT/" + sub_folder
                
                self.lvot_files.extend(sorted(glob.glob(lvot_folder + '/**/*.txt', recursive=True)))
                self.rvot_files.extend(sorted(glob.glob(rvot_folder + '/**/*.txt', recursive=True)))

                if self.class_count == 3:
                    lrvot_folder = base_path + "RLVOT/" + sub_folder
                    self.lrvot_files.extend(sorted(glob.glob(lrvot_folder + '/**/*.txt', recursive=True)))

        else:
            # Diğer setler ('test_cured', 'test_notcured') için mevcut yapıyı koru
            lvot_folder = base_path + "LVOT/" + self.set_type
            rvot_folder = base_path + "RVOT/" + self.set_type
            
            self.lvot_files = sorted(glob.glob(lvot_folder+ '/**/*.txt', recursive=True))
            self.rvot_files = sorted(glob.glob(rvot_folder+ '/**/*.txt', recursive=True))

            if self.class_count == 3:
                lrvot_folder = base_path + "RLVOT/" + self.set_type
                self.lrvot_files = sorted(glob.glob(lrvot_folder+ '/**/*.txt', recursive=True))


        print("lvot", len(self.lvot_files))
        print("rvot", len(self.rvot_files))
        if self.class_count == 3:
            print("lrvot", len(self.lrvot_files))

        if self.class_count == 2:
            self.files = self.lvot_files + self.rvot_files
            self.labels = [0] * len(self.lvot_files) + [1] * len(self.rvot_files)

        if self.class_count == 3:
            self.files = self.lvot_files + self.rvot_files + self.lrvot_files
            self.labels = [0] * len(self.lvot_files) + [1] * len(self.rvot_files)  + [2] * len(self.lrvot_files)

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

        if self.augment and self.set_type == 'train':  # Apply augmentation only on training data
            for i in range(data.shape[0]):  # Apply augmentation to each channel
                data[i] = self.augment_signal(data[i])
        # MDI hesapla ve kanal olarak ekle
        mdi_values = np.array([compute_mdi(data[i]) for i in range(12)])
        mdi_normalized = mdi_values.mean() / (np.max(mdi_values) + 1e-6)
        mdi_channel = np.full((1, data.shape[1]), mdi_normalized)  # (1, N)
        data = np.concatenate([data, mdi_channel], axis=0)  # (13, N)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label), self.files[idx]

# Model (Aynı kaldığı için tekrar yazılmadı)
class LargerSimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = nn.Sequential(nn.Conv1d(13, 64, 3, 1), nn.ReLU(), nn.Conv1d(64, 64, 3, 1), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2))
        self.block1 = nn.Sequential(nn.Conv1d(64, 128, 3, 1), nn.ReLU(), nn.Conv1d(128, 128, 3, 1), nn.ReLU(), nn.BatchNorm1d(128), nn.MaxPool1d(2))
        self.block2 = nn.Sequential(nn.Conv1d(128, 256, 3, 1), nn.ReLU(), nn.Conv1d(256, 256, 3, 1), nn.ReLU(), nn.BatchNorm1d(256), nn.MaxPool1d(2))
        self.block3 = nn.Sequential(nn.Conv1d(256, 128, 3, 1), nn.ReLU(), nn.Conv1d(128, 128, 3, 1), nn.ReLU(), nn.BatchNorm1d(128), nn.MaxPool1d(2))
        self.block4 = nn.Sequential(nn.Conv1d(128, 64, 3, 1), nn.ReLU(), nn.Conv1d(64, 64, 3, 1), nn.ReLU(), nn.MaxPool1d(2))
        self.fc = nn.Sequential(nn.Linear(4736, 1024), nn.ReLU(), nn.Dropout(0.6),
                                nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.6),
                                nn.Linear(256, 2))

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Train Model (Aynı kaldığı için tekrar yazılmadı)
def train_model():

    model = LargerSimpleNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    train_dataset = CartoDataset('train',2)
    test_cured_dataset = CartoDataset('test_cured',2)
    test_not_cured_dataset = CartoDataset('test_notcured',2)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)
    test_cured_loader = torch.utils.data.DataLoader(test_cured_dataset, batch_size=128, shuffle=False, drop_last=False)
    test_not_cured_loader = torch.utils.data.DataLoader(test_not_cured_dataset, batch_size=128, shuffle=False, drop_last=False)


    # Initialize lists to store accuracy values
    train_accuracies = []
    test_cured_accuracies = []
    test_not_cured_accuracies = []

    num_epochs = 100
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
    

        for (i, batch) in enumerate(train_loader):
            signals = batch[0].float()
            labels = batch[1]
            txt = batch[2]

            signals = signals.cuda()
            labels = labels.cuda()

            if len(labels) == 0:
                continue

            result = model(signals)
            loss = criterion(result, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            _, predicted = torch.max(result.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if i % 100 == 99 or i == train_loader.dataset.__len__()//128 -1:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), running_loss / 10))
                running_loss = 0.0
                print("Train accuracy: ", 100 * correct_train / total_train)

        #scheduler.step()

        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        model.eval()

        # Initialize counters for RVOT and LVOT accuracies
        correct_rvot_cured = 0
        correct_lvot_cured = 0
        correct_rlvot_cured = 0
        total_rvot_cured = 0
        total_lvot_cured = 0
        total_rlvot_cured = 0

        print("Cured Test:")
        with torch.no_grad():
            for batch in test_cured_loader:
                signals = batch[0].float()
                labels = batch[1]  # Labels: 0 -> LVOT, 1 -> RVOT
                txt = batch[2]
                
                signals = signals.cuda()
                labels = labels.cuda()
                
                result = model(signals)
                _, predicted = torch.max(result.data, 1)

                # Calculate total and correct for each class
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
        #print(f'RLVOT Test Cured Accuracy: {rlvot_accuracy_cured:.2f}%')

        correct_rvot_notcured = 0
        correct_lvot_notcured = 0
        correct_rlvot_notcured = 0
        total_rvot_notcured = 0
        total_lvot_notcured = 0
        total_rlvot_notcured = 0

        print("Not Cured Test:")
        with torch.no_grad():
            for batch in test_not_cured_loader:
                signals = batch[0].float()
                labels = batch[1]  # Labels: 0 -> LVOT, 1 -> RVOT
                txt = batch[2]

                signals = signals.cuda()
                labels = labels.cuda()

                result = model(signals)
                _, predicted = torch.max(result.data, 1)

                # Calculate total and correct for each class
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
        #print(f'RLVOT Test Not Cured Accuracy: {rlvot_accuracy_not_cured:.2f}%')

        test_total_accuracy = 100 * (correct_lvot_cured + correct_rvot_cured + correct_rlvot_cured + correct_lvot_notcured + correct_rvot_notcured + correct_rlvot_notcured) / (total_lvot_cured + total_rvot_cured + total_rlvot_cured + total_lvot_notcured + total_rvot_notcured + total_rlvot_notcured)

        print('Epoch [{}/{}], Test Accuracy: {:.2f}%'.format(epoch+1, num_epochs, test_total_accuracy))


        model_filename = f'mdi-newdata-2class-notcuredchanged/model_{epoch+1}-{test_total_accuracy}.pth'
        torch.save(model.state_dict(), model_filename)


    print('Training finished')

if __name__ == "__main__":
    train_model()