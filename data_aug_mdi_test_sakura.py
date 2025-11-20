import os
import glob
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt

# =========================
# Kullanıcı ayarları (yalnızca bunları değiştirin)
# =========================
EVAL_CHECKPOINT = "mdi-newdata-2class-notcuredchanged/model_7-98.21782178217822.pth"  # Kayıtlı .pth modeli
DATA_ROOT       = "/mnt/disk2/carto/train-test-folder"                      # Kök klasör
CLASS_COUNT     = 2   # 2 veya 3
BATCH_SIZE      = 128

# =========================
# Butterworth filtre (eğitim koduyla aynı)
# =========================
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

# =========================
# MDI hesaplama (eğitim koduyla aynı)
# =========================
def compute_mdi(signal):
    baseline = np.median(signal)
    signal = signal - baseline
    mid = len(signal) // 2
    window = signal[mid - 100: mid + 100]
    idx_max = np.argmax(np.abs(window))
    mdi_value = idx_max / len(window)
    return mdi_value

# =========================
# TXT → numpy okuma (eğitim kodunuzla aynı mantık)
# =========================
def process_text_file_2(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 0 ve 1. satır atlanıyor, 2. satır header kabul ediliyor
    headers = lines[2].split()

    # İlgili sütun indeksleri (eğitimdekiyle aynı)
    v1_index   = headers.index('V1(22)')
    v2_index   = headers.index('V2(23)')
    v3_index   = headers.index('V3(24)')
    v4_index   = headers.index('V4(25)')
    v5_index   = headers.index('V5(26)')
    v6_index   = headers.index('V6(27)')
    i_index    = headers.index('I(110)')
    ii_index   = headers.index('II(111)')
    iii_index  = headers.index('III(112)')
    avl_index  = headers.index('aVL(171)')
    avr_index  = headers.index('aVR(172)')
    avf_index  = headers.index('aVF(173)')

    data = np.zeros((12, 2500))
    counter = 0

    for line in lines[3:]:
        items = line.split()
        data[0,  counter] = int(items[v1_index])
        data[1,  counter] = int(items[v2_index])
        data[2,  counter] = int(items[v3_index])
        data[3,  counter] = int(items[v4_index])
        data[4,  counter] = int(items[v5_index])
        data[5,  counter] = int(items[v6_index])
        data[6,  counter] = int(items[i_index])
        data[7,  counter] = int(items[ii_index])
        data[8,  counter] = int(items[iii_index])
        data[9,  counter] = int(items[avl_index])
        data[10, counter] = int(items[avr_index])
        data[11, counter] = int(items[avf_index])
        counter += 1
        if counter >= 2500:
            break

    # İsterseniz filtreyi burada uygulayabilirsiniz; eğitimde raw döndürüldüğü için raw bırakıyoruz.
    labels = np.array([])  # placeholder
    return data, labels

# =========================
# Test-only Dataset (tek split: class/test/)
# =========================
class CartoDatasetTestOnly(Dataset):
    """
    Beklenen yapı:
      DATA_ROOT/
        LVOT/test/**/*.txt
        RVOT/test/**/*.txt
        (opsiyonel) RLVOT/test/**/*.txt
    """
    def __init__(self, base_path, class_count=2):
        self.base_path = base_path.rstrip("/")
        self.class_count = class_count

        self.lvot_files = sorted(glob.glob(f"{self.base_path}/LVOT/test_sakura/**/**/*.txt", recursive=True))
        self.rvot_files = sorted(glob.glob(f"{self.base_path}/RVOT/test_sakura/**/**/*.txt", recursive=True))
        self.rlvot_files = []
        if self.class_count == 3:
            self.rlvot_files = sorted(glob.glob(f"{self.base_path}/RLVOT/test_sakura/**/**/*.txt", recursive=True))

        if self.class_count == 2:
            self.files  = self.lvot_files + self.rvot_files
            self.labels = [0] * len(self.lvot_files) + [1] * len(self.rvot_files)
        else:
            self.files  = self.lvot_files + self.rvot_files + self.rlvot_files
            self.labels = [0] * len(self.lvot_files) + [1] * len(self.rvot_files) + [2] * len(self.rlvot_files)

        print(f"[TEST] lvot={len(self.lvot_files)} rvot={len(self.rvot_files)}"
              + (f" rlvot={len(self.rlvot_files)}" if self.class_count == 3 else ""))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        txt = self.files[idx]
        label = self.labels[idx]
        data, _ = process_text_file_2(txt)  # (12, N)

        # Eğitimdeki gibi MDI kanalını ekleyelim (birebir aynı olmalı!)
        mdi_values = np.array([compute_mdi(data[i]) for i in range(12)])
        mdi_normalized = mdi_values.mean() / (np.max(mdi_values) + 1e-6)
        mdi_channel = np.full((1, data.shape[1]), mdi_normalized)  # (1, N)
        data = np.concatenate([data, mdi_channel], axis=0)  # (13, N)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label), txt

# =========================
# Model (eğitimdekiyle aynı mimari)
# =========================
class LargerSimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = nn.Sequential(
            nn.Conv1d(13, 64, 3, 1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1), nn.ReLU(),
            nn.BatchNorm1d(64), nn.MaxPool1d(2)
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1), nn.ReLU(),
            nn.Conv1d(128, 128, 3, 1), nn.ReLU(),
            nn.BatchNorm1d(128), nn.MaxPool1d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, 3, 1), nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1), nn.ReLU(),
            nn.BatchNorm1d(256), nn.MaxPool1d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(256, 128, 3, 1), nn.ReLU(),
            nn.Conv1d(128, 128, 3, 1), nn.ReLU(),
            nn.BatchNorm1d(128), nn.MaxPool1d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 64, 3, 1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1), nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Not: Eğitimde Linear giriş boyutu 4736 olarak sabitlenmişti.
        self.fc = nn.Sequential(
            nn.Linear(4736, 1024), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(256, 2)  # 2 sınıf için; 3 sınıf model eğittiyseniz burada 3 olmalı.
        )

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# =========================
# Değerlendirme yardımcıları
# =========================
@torch.no_grad()
def _eval_loader(model, loader, class_count=2, device="cuda"):
    model.eval()
    correct = {0: 0, 1: 0, 2: 0}
    total   = {0: 0, 1: 0, 2: 0}

    for signals, labels, _ in loader:
        signals = signals.to(device).float()
        labels  = labels.to(device)

        logits = model(signals)
        preds  = logits.argmax(dim=1)

        for c in [0, 1] + ([2] if class_count == 3 else []):
            mask = (labels == c)
            total[c]   += mask.sum().item()
            correct[c] += (preds[mask] == c).sum().item()

    acc_per_class = {}
    for c in [0, 1] + ([2] if class_count == 3 else []):
        acc_per_class[c] = 100.0 * correct[c] / total[c] if total[c] > 0 else 0.0

    overall = 100.0 * (sum(correct.values()) / max(1, sum(total.values())))
    return acc_per_class, overall

def evaluate_saved_model_single_split(checkpoint_path, data_root, class_count=3, batch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Modeli yükle ---
    model = LargerSimpleNet().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- Test dataset/loader ---
    ds_test = CartoDatasetTestOnly(data_root, class_count=class_count)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)

    acc, overall = _eval_loader(model, dl_test, class_count=class_count, device=device)

    print("\nTest (single split = class/test):")
    print(f"LVOT Test Accuracy: {acc.get(0, 0.0):.2f}%")
    print(f"RVOT Test Accuracy: {acc.get(1, 0.0):.2f}%")
    if class_count == 3:
        print(f"RLVOT Test Accuracy: {acc.get(2, 0.0):.2f}%")
    print(f"Overall Test Accuracy: {overall:.2f}%")

# =========================
# Main (argümansız)
# =========================
if __name__ == "__main__":
    evaluate_saved_model_single_split(
        checkpoint_path=EVAL_CHECKPOINT,
        data_root=DATA_ROOT,
        class_count=CLASS_COUNT,
        batch_size=BATCH_SIZE
    )
