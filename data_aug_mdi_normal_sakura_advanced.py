import os
import glob
import random
import numpy as np
import torch
from torch import nn, optim
import torch.utils.data
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# =============================================================================
# 1. YARDIMCI FONKSİYONLAR
# =============================================================================

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

def robust_normalize(signal):
    """
    Sadece RAM'deki veri üzerinde çalışır. Orijinal dosyayı değiştirmez.
    Medyan ve IQR bazlı ölçekleme.
    signal: (C, T)
    """
    signal = signal.astype(np.float32, copy=False)
    for i in range(signal.shape[0]):
        median = np.median(signal[i])
        q75, q25 = np.percentile(signal[i], [75, 25])
        iqr = q75 - q25
        signal[i] = (signal[i] - median) / (iqr + 1e-6)
    return signal

def compute_mdi(signal):
    baseline = np.median(signal)
    signal = signal - baseline
    mid = len(signal) // 2
    start = max(0, mid - 100)
    end = min(len(signal), mid + 100)
    window = signal[start:end]

    if len(window) == 0:
        return 0.0

    idx_max = np.argmax(np.abs(window))
    mdi_value = idx_max / len(window)
    return mdi_value

def process_text_file_2(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if len(lines) < 4:
            raise ValueError("Dosya çok kısa")

        headers = lines[2].split()

        v1_index  = headers.index('V1(22)')
        v2_index  = headers.index('V2(23)')
        v3_index  = headers.index('V3(24)')
        v4_index  = headers.index('V4(25)')
        v5_index  = headers.index('V5(26)')
        v6_index  = headers.index('V6(27)')
        i_index   = headers.index('I(110)')
        ii_index  = headers.index('II(111)')
        iii_index = headers.index('III(112)')
        avl_index = headers.index('aVL(171)')
        avr_index = headers.index('aVR(172)')
        avf_index = headers.index('aVF(173)')

        data = np.zeros((12, 2500), dtype=np.float32)
        counter = 0

        for line in lines[3:]:
            items = line.split()
            if len(items) < max(v1_index, avf_index):
                continue

            data[0,  counter] = float(items[v1_index])
            data[1,  counter] = float(items[v2_index])
            data[2,  counter] = float(items[v3_index])
            data[3,  counter] = float(items[v4_index])
            data[4,  counter] = float(items[v5_index])
            data[5,  counter] = float(items[v6_index])
            data[6,  counter] = float(items[i_index])
            data[7,  counter] = float(items[ii_index])
            data[8,  counter] = float(items[iii_index])
            data[9,  counter] = float(items[avl_index])
            data[10, counter] = float(items[avr_index])
            data[11, counter] = float(items[avf_index])

            counter += 1
            if counter >= 2500:
                break

        return data, np.array([])

    except Exception as e:
        print(f"Hata ({file_path}): {e}")
        return np.zeros((12, 2500), dtype=np.float32), np.array([])

def safe_float_str(x, nd=2):
    return f"{x:.{nd}f}".replace(".", "p")

# =============================================================================
# 2. DATASET SINIFLARI
# =============================================================================

class CartoDataset(torch.utils.data.Dataset):
    def __init__(self, set_type='train', class_count=2, augment=False, fs=1000):
        self.set_type = set_type
        self.class_count = class_count
        self.augment = augment
        self.fs = fs

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
                self.lrvot_files.extend(sorted(glob.glob(lrvot_folder + '/**/*.txt', recursive=True)))

        print(f"[{set_type.upper()}] LVOT: {len(self.lvot_files)}, RVOT: {len(self.rvot_files)}")

        if self.class_count == 2:
            self.files = self.lvot_files + self.rvot_files
            self.labels = [0] * len(self.lvot_files) + [1] * len(self.rvot_files)
        else:
            self.files = self.lvot_files + self.rvot_files + self.lrvot_files
            self.labels = [0] * len(self.lvot_files) + [1] * len(self.rvot_files) + [2] * len(self.lrvot_files)

    def __len__(self):
        return len(self.labels)

    # --------- Domain-style augmentations (Sakura-like) ----------
    def augment_signal_domain(self, sig):
        """
        sig: (T,)
        """
        sig = sig.astype(np.float32, copy=True)
        T = len(sig)
        fs = self.fs

        # Lead gain variation
        if random.random() < 0.7:
            sig *= random.uniform(0.7, 1.3)

        # Baseline wander (low freq sine)
        if random.random() < 0.5:
            t = np.arange(T, dtype=np.float32) / float(fs)
            freq = random.uniform(0.05, 0.5)
            amp = random.uniform(0.01, 0.08) * (np.std(sig) + 1e-6)
            phase = random.uniform(0, 2*np.pi)
            sig = sig + amp * np.sin(2*np.pi*freq*t + phase).astype(np.float32)

        # Band-limited noise (device difference)
        if random.random() < 0.5:
            noise = np.random.randn(T).astype(np.float32)
            highcut = random.uniform(15.0, 50.0)
            try:
                noise = butter_filter(noise, highcut=highcut, fs=fs, order=2).astype(np.float32)
            except Exception:
                pass
            sig += random.uniform(0.005, 0.05) * (np.std(sig) + 1e-6) * noise

        # Time shift
        if random.random() < 0.5:
            sig = np.roll(sig, random.randint(-50, 50)).astype(np.float32)

        # Polarity flip (sometimes leads are inverted)
        if random.random() < 0.15:
            sig = (-sig).astype(np.float32)

        # Cutout / dropout segment
        if random.random() < 0.3:
            mask_len = random.randint(50, 250)
            if mask_len < T:
                start = random.randint(0, T - mask_len)
                sig[start:start + mask_len] = 0.0

        return sig

    def __getitem__(self, idx):
        txt = self.files[idx]
        label = self.labels[idx]
        data, _ = process_text_file_2(txt)  # (12, 2500)

        # Apply augmentations only on TRAIN
        if self.augment and self.set_type == 'train':
            for i in range(data.shape[0]):
                data[i] = self.augment_signal_domain(data[i])

        # MDI from ORIGINAL leads (before derivative)
        mdi_vector = np.array([compute_mdi(data[i]) for i in range(12)], dtype=np.float32)

        # ✅ CHANGE 1: Normalize ONLY original 12 leads
        data = robust_normalize(data)  # (12, 2500)

        # ✅ CHANGE 2: Derivative computed AFTER normalization
        derivative = np.gradient(data, axis=1).astype(np.float32)  # (12, 2500)

        combined_signal = np.concatenate([data, derivative], axis=0).astype(np.float32)  # (24, 2500)

        return (torch.tensor(combined_signal, dtype=torch.float32),
                torch.tensor(mdi_vector, dtype=torch.float32)), torch.tensor(label), txt

class CartoDatasetTestSakura(torch.utils.data.Dataset):
    def __init__(self, base_path="/mnt/disk2/carto/train-test-folder/", class_count=2, fs=1000):
        self.base_path = base_path.rstrip("/")
        self.class_count = class_count
        self.fs = fs

        self.lvot_files = sorted(glob.glob(f"{self.base_path}/LVOT/test_sakura_filtered_median/**/*.txt", recursive=True))
        self.rvot_files = sorted(glob.glob(f"{self.base_path}/RVOT/test_sakura_filtered_median/**/*.txt", recursive=True))
        self.rlvot_files = []
        if self.class_count == 3:
            self.rlvot_files = sorted(glob.glob(f"{self.base_path}/RLVOT/test_sakura_filtered_median/**/*.txt", recursive=True))

        if self.class_count == 2:
            self.files = self.lvot_files + self.rvot_files
            self.labels = [0] * len(self.lvot_files) + [1] * len(self.rvot_files)
        else:
            self.files = self.lvot_files + self.rvot_files + self.rlvot_files
            self.labels = [0] * len(self.lvot_files) + [1] * len(self.rvot_files) + [2] * len(self.rlvot_files)

        print(f"[SAKURA TEST] LVOT: {len(self.lvot_files)}, RVOT: {len(self.rvot_files)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        txt = self.files[idx]
        label = self.labels[idx]
        data, _ = process_text_file_2(txt)  # (12, 2500)

        mdi_vector = np.array([compute_mdi(data[i]) for i in range(12)], dtype=np.float32)

        # ✅ SAME preprocessing idea: normalize 12, then derivative
        data = robust_normalize(data)
        derivative = np.gradient(data, axis=1).astype(np.float32)
        combined_signal = np.concatenate([data, derivative], axis=0).astype(np.float32)

        return (torch.tensor(combined_signal, dtype=torch.float32),
                torch.tensor(mdi_vector, dtype=torch.float32)), torch.tensor(label), txt

# =============================================================================
# 3. MODEL
# =============================================================================

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.InstanceNorm1d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.InstanceNorm1d(out_channels, affine=True)
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm1d(out_channels, affine=True)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class AdvancedECGNet(nn.Module):
    def __init__(self, in_ch=24, num_classes=2):
        super(AdvancedECGNet, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm1d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(32, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 + 12, num_classes)
        )

    def _make_layer(self, in_ch, out_ch, stride):
        return ResidualBlock(in_ch, out_ch, stride)

    def forward(self, x, mdi):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        combined = torch.cat((x, mdi), dim=1)
        x = self.fc(combined)
        return x

# =============================================================================
# 4. EVALUATION & LOGGING
# =============================================================================

@torch.no_grad()
def eval_loss_acc(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for (signals, mdis), labels, _ in loader:
        signals = signals.to(device).float()
        mdis = mdis.to(device).float()
        labels = labels.to(device)
        logits = model(signals, mdis)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

    avg_loss = total_loss / max(1, total)
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc

@torch.no_grad()
def eval_confusion_matrix(model, loader, device, num_classes=2):
    """
    ✅ CHANGE 3: Confusion matrix + per-class accuracy for Sakura (or any loader).
    """
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)  # rows=true, cols=pred

    for (signals, mdis), labels, _ in loader:
        signals = signals.to(device).float()
        mdis = mdis.to(device).float()
        labels = labels.to(device)

        logits = model(signals, mdis)
        preds = logits.argmax(dim=1)

        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

    cm_np = cm.cpu().numpy()
    per_class_acc = []
    for c in range(num_classes):
        denom = cm_np[c].sum()
        per_class_acc.append(100.0 * (cm_np[c, c] / denom) if denom > 0 else 0.0)

    return cm_np, per_class_acc

def save_metric_plots(out_dir, epochs, train_loss, test_loss, sakura_loss, train_acc, test_acc, sakura_acc):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, test_loss, label="Test Loss")
    plt.plot(epochs, sakura_loss, label="Sakura Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, test_acc, label="Test Acc")
    plt.plot(epochs, sakura_acc, label="Sakura Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "acc_curves.png"), dpi=200)
    plt.close()

def append_summary_to_txt(txt_path, epoch, num_epochs, train_loss, train_acc,
                          test_loss, test_acc, sakura_loss, sakura_acc, lr,
                          sakura_cm=None, sakura_per_class=None):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    cm_block = ""
    if sakura_cm is not None and sakura_per_class is not None:
        cm_block = (
            "SAKURA CONFUSION MATRIX (rows=true, cols=pred):\n"
            f"{sakura_cm}\n"
            f"SAKURA per-class acc: {['%.2f%%' % a for a in sakura_per_class]}\n"
        )

    block = (
        "============================================================\n"
        f"Epoch {epoch}/{num_epochs} | LR: {lr:.6f}\n"
        f"TRAIN : loss={train_loss:.4f}  acc={train_acc:.2f}%\n"
        f"TEST  : loss={test_loss:.4f}  acc={test_acc:.2f}%\n"
        f"SAKURA: loss={sakura_loss:.4f}  acc={sakura_acc:.2f}%\n"
        f"{cm_block}"
        "============================================================\n\n"
    )
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(block)

# =============================================================================
# 4.5 CHECKPOINT HELPERS
# =============================================================================

def save_checkpoint(path, epoch, model, optimizer, scheduler, best_avg_acc,
                    hist_epoch, hist_train_loss, hist_test_loss, hist_sakura_loss,
                    hist_train_acc, hist_test_acc, hist_sakura_acc, mixup_prob):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_avg_acc": best_avg_acc,
        "mixup_prob": mixup_prob,
        "history": {
            "epoch": hist_epoch,
            "train_loss": hist_train_loss,
            "test_loss": hist_test_loss,
            "sakura_loss": hist_sakura_loss,
            "train_acc": hist_train_acc,
            "test_acc": hist_test_acc,
            "sakura_acc": hist_sakura_acc,
        }
    }
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch = int(ckpt.get("epoch", 0))
    best_avg_acc = float(ckpt.get("best_avg_acc", 0.0))
    mixup_prob = float(ckpt.get("mixup_prob", 0.5))

    hist = ckpt.get("history", {})
    hist_epoch = hist.get("epoch", [])
    hist_train_loss = hist.get("train_loss", [])
    hist_test_loss = hist.get("test_loss", [])
    hist_sakura_loss = hist.get("sakura_loss", [])
    hist_train_acc = hist.get("train_acc", [])
    hist_test_acc = hist.get("test_acc", [])
    hist_sakura_acc = hist.get("sakura_acc", [])

    return (start_epoch, best_avg_acc, mixup_prob,
            hist_epoch, hist_train_loss, hist_test_loss, hist_sakura_loss,
            hist_train_acc, hist_test_acc, hist_sakura_acc)

# =============================================================================
# 5. MAIN TRAINING LOOP
# =============================================================================

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = AdvancedECGNet(in_ch=24, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    # ✅ augment=True now includes Sakura-like domain augmentations
    train_dataset = CartoDataset("train", 2, augment=True, fs=1000)
    test_cured_dataset = CartoDataset("test_cured", 2, augment=False, fs=1000)
    test_not_cured_dataset = CartoDataset("test_notcured", 2, augment=False, fs=1000)
    sakura_test_dataset = CartoDatasetTestSakura("/mnt/disk2/carto/train-test-folder/", class_count=2, fs=1000)

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_cured_loader = torch.utils.data.DataLoader(test_cured_dataset, batch_size=batch_size, shuffle=False)
    test_not_cured_loader = torch.utils.data.DataLoader(test_not_cured_dataset, batch_size=batch_size, shuffle=False)
    sakura_test_loader = torch.utils.data.DataLoader(sakura_test_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 50
    out_dir = "mdi_resnet_domain_robust_v2_norm12_then_deriv_mixupMDIoff_cm"
    os.makedirs(out_dir, exist_ok=True)

    summary_txt_path = os.path.join(out_dir, "training_summary.txt")

    last_ckpt_path = os.path.join(out_dir, "last_checkpoint.pth")
    best_ckpt_path = os.path.join(out_dir, "best_checkpoint.pth")

    hist_epoch, hist_train_loss, hist_test_loss, hist_sakura_loss = [], [], [], []
    hist_train_acc, hist_test_acc, hist_sakura_acc = [], [], []

    best_avg_acc = 0.0
    mixup_prob = 0.5

    RESUME = True
    start_epoch = 0
    if RESUME and os.path.exists(last_ckpt_path):
        print(f"[RESUME] Loading checkpoint: {last_ckpt_path}")
        (start_epoch, best_avg_acc, mixup_prob,
         hist_epoch, hist_train_loss, hist_test_loss, hist_sakura_loss,
         hist_train_acc, hist_test_acc, hist_sakura_acc) = load_checkpoint(
            last_ckpt_path, model, optimizer, scheduler, device=device
        )
    else:
        print("[START] Training from scratch...")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for batch_idx, ((signals, mdis), labels, _) in enumerate(train_loader):
            signals = signals.to(device).float()
            mdis = mdis.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()

            if random.random() < mixup_prob:
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)

                index = torch.randperm(signals.size(0)).to(device)

                # ✅ CHANGE 4: Mixup only signals
                mixed_signals = lam * signals + (1 - lam) * signals[index, :]

                # ✅ keep MDI from dominant sample (winner-takes-all by lam)
                if lam >= 0.5:
                    mixed_mdis = mdis
                    y_a, y_b = labels, labels[index]
                else:
                    mixed_mdis = mdis[index, :]
                    # swap so y_a corresponds to dominant sample
                    y_a, y_b = labels[index], labels

                logits = model(mixed_signals, mixed_mdis)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                logits = model(signals, mdis)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step(epoch + batch_idx / len(train_loader))

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / max(1, total_train)
        train_acc = 100.0 * correct_train / max(1, total_train)

        loss_cured, acc_cured = eval_loss_acc(model, test_cured_loader, criterion, device)
        loss_not, acc_not = eval_loss_acc(model, test_not_cured_loader, criterion, device)

        n_cured, n_not = len(test_cured_dataset), len(test_not_cured_dataset)
        test_loss_combined = (loss_cured * n_cured + loss_not * n_not) / (n_cured + n_not)
        test_acc_combined = (acc_cured * n_cured + acc_not * n_not) / (n_cured + n_not)

        sakura_loss, sakura_acc = eval_loss_acc(model, sakura_test_loader, criterion, device)

        # ✅ CHANGE 5: Confusion matrix to diagnose class-specific Sakura error
        sakura_cm, sakura_per_class = eval_confusion_matrix(model, sakura_test_loader, device, num_classes=2)

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.6f} | "
            f"TrA: {train_acc:.2f}% | TeA: {test_acc_combined:.2f}% | SaA: {sakura_acc:.2f}% | "
            f"Sa per-class: {[round(x,2) for x in sakura_per_class]}"
        )
        print("Sakura CM (rows=true, cols=pred):")
        print(sakura_cm)

        append_summary_to_txt(
            summary_txt_path, epoch+1, num_epochs,
            train_loss, train_acc,
            test_loss_combined, test_acc_combined,
            sakura_loss, sakura_acc,
            current_lr,
            sakura_cm=sakura_cm,
            sakura_per_class=sakura_per_class
        )

        hist_epoch.append(epoch+1)
        hist_train_loss.append(train_loss)
        hist_test_loss.append(test_loss_combined)
        hist_sakura_loss.append(sakura_loss)
        hist_train_acc.append(train_acc)
        hist_test_acc.append(test_acc_combined)
        hist_sakura_acc.append(sakura_acc)

        save_metric_plots(
            out_dir,
            hist_epoch, hist_train_loss, hist_test_loss, hist_sakura_loss,
            hist_train_acc, hist_test_acc, hist_sakura_acc
        )

        current_avg_acc = (test_acc_combined + sakura_acc) / 2

        # ✅ BEST checkpoint (NEW BEST olduğunda isimlendirilmiş kayıt)
        if current_avg_acc > best_avg_acc:
            best_avg_acc = current_avg_acc

            tr_s = safe_float_str(train_acc, 2)
            te_s = safe_float_str(test_acc_combined, 2)
            sa_s = safe_float_str(sakura_acc, 2)
            avg_s = safe_float_str(current_avg_acc, 2)

            best_snap_name = f"best_avg_{avg_s}_tr_{tr_s}_te_{te_s}_sak_{sa_s}_ep{epoch+1}.pth"
            best_snap_path = os.path.join(out_dir, best_snap_name)

            print(f"\n*** YENİ EN İYİ MODEL KAYDEDİLDİ! ***")
            print(f"Path: {best_snap_name}\n")

            save_checkpoint(
                best_ckpt_path, epoch+1, model, optimizer, scheduler, best_avg_acc,
                hist_epoch, hist_train_loss, hist_test_loss, hist_sakura_loss,
                hist_train_acc, hist_test_acc, hist_sakura_acc, mixup_prob
            )

            torch.save(model.state_dict(), best_snap_path)

        # ✅ LAST checkpoint (Her epoch sonunda üzerine yazılır)
        save_checkpoint(
            last_ckpt_path, epoch+1, model, optimizer, scheduler, best_avg_acc,
            hist_epoch, hist_train_loss, hist_test_loss, hist_sakura_loss,
            hist_train_acc, hist_test_acc, hist_sakura_acc, mixup_prob
        )

if __name__ == "__main__":
    train_model()                     