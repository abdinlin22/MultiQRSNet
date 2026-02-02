import os
import glob
import random
import numpy as np
import torch
from torch import nn, optim
import torch.utils.data
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# =========================
# Butterworth filter
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
# MDI hesaplama
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
# TXT okuma
# =========================
def process_text_file_2(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

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

    labels = np.array([])  # placeholder
    return data, labels

# =========================
# Train/Test Dataset
# =========================
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
        else:
            self.files = self.lvot_files + self.rvot_files + self.lrvot_files
            self.labels = [0] * len(self.lvot_files) + [1] * len(self.rvot_files) + [2] * len(self.lrvot_files)

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

        if self.augment and self.set_type == 'train':
            for i in range(data.shape[0]):
                data[i] = self.augment_signal(data[i])

        mdi_values = np.array([compute_mdi(data[i]) for i in range(12)])
        mdi_normalized = mdi_values.mean() / (np.max(mdi_values) + 1e-6)
        mdi_channel = np.full((1, data.shape[1]), mdi_normalized)
        data = np.concatenate([data, mdi_channel], axis=0)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label), txt

# =========================
# Sakura Test-Only Dataset
# =========================
class CartoDatasetTestSakura(torch.utils.data.Dataset):
    """
    Beklenen yapı:
      /mnt/disk2/carto/train-test-folder/
        LVOT/test_sakura/**/*.txt
        RVOT/test_sakura/**/*.txt
        (opsiyonel) RLVOT/test_sakura/**/*.txt
    """
    def __init__(self, base_path="/mnt/disk2/carto/train-test-folder/", class_count=2):
        self.base_path = base_path.rstrip("/")
        self.class_count = class_count

        self.lvot_files = sorted(glob.glob(f"{self.base_path}/LVOT/test_sakura/**/*.txt", recursive=True))
        self.rvot_files = sorted(glob.glob(f"{self.base_path}/RVOT/test_sakura/**/*.txt", recursive=True))
        self.rlvot_files = []
        if self.class_count == 3:
            self.rlvot_files = sorted(glob.glob(f"{self.base_path}/RLVOT/test_sakura/**/*.txt", recursive=True))

        if self.class_count == 2:
            self.files = self.lvot_files + self.rvot_files
            self.labels = [0] * len(self.lvot_files) + [1] * len(self.rvot_files)
        else:
            self.files = self.lvot_files + self.rvot_files + self.rlvot_files
            self.labels = [0] * len(self.lvot_files) + [1] * len(self.rvot_files) + [2] * len(self.rlvot_files)

        print(
            f"[SAKURA TEST] lvot={len(self.lvot_files)} rvot={len(self.rvot_files)}"
            + (f" rlvot={len(self.rlvot_files)}" if self.class_count == 3 else "")
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        txt = self.files[idx]
        label = self.labels[idx]
        data, _ = process_text_file_2(txt)

        mdi_values = np.array([compute_mdi(data[i]) for i in range(12)])
        mdi_normalized = mdi_values.mean() / (np.max(mdi_values) + 1e-6)
        mdi_channel = np.full((1, data.shape[1]), mdi_normalized)
        data = np.concatenate([data, mdi_channel], axis=0)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label), txt

# =========================
# Model
# =========================
class LargerSimpleNet(nn.Module):
    def __init__(self, in_ch=13, num_classes=2, input_len=2500):
        super().__init__()

        c0, c1, c2 = 16, 32, 64

        self.block0 = nn.Sequential(
            nn.Conv1d(in_ch, c0, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(c0),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(c0, c1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
        )

        # flatten dim otomatik
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, input_len)
            feat = self._forward_features(dummy)
            feat_dim = feat.shape[1]

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def _forward_features(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        return self.fc(self._forward_features(x))



# =========================
# Eval helper: per-class acc
# =========================
@torch.no_grad()
def eval_loader_per_class(model, loader, class_count=2, device="cuda"):
    model.eval()
    correct = {0: 0, 1: 0, 2: 0}
    total = {0: 0, 1: 0, 2: 0}

    for signals, labels, _ in loader:
        signals = signals.to(device).float()
        labels = labels.to(device)

        logits = model(signals)
        preds = logits.argmax(dim=1)

        classes = [0, 1] + ([2] if class_count == 3 else [])
        for c in classes:
            mask = labels == c
            total[c] += mask.sum().item()
            correct[c] += (preds[mask] == c).sum().item()

    classes = [0, 1] + ([2] if class_count == 3 else [])
    acc_per_class = {c: (100.0 * correct[c] / total[c] if total[c] > 0 else 0.0) for c in classes}
    overall = 100.0 * (sum(correct[c] for c in classes) / max(1, sum(total[c] for c in classes)))
    return acc_per_class, overall

# =========================
# Eval helper: loss + acc
# =========================
@torch.no_grad()
def eval_loss_acc(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for signals, labels, _ in loader:
        signals = signals.to(device).float()
        labels = labels.to(device)

        logits = model(signals)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

    avg_loss = total_loss / max(1, total)
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc

# =========================
# Plot saver
# =========================
def save_metric_plots(out_dir, epochs, train_loss, test_loss, sakura_loss,
                      train_acc, test_acc, sakura_acc):
    os.makedirs(out_dir, exist_ok=True)

    # LOSS
    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.plot(epochs, sakura_loss, label="sakura_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=200)
    plt.close()

    # ACC
    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, test_acc, label="test_acc")
    plt.plot(epochs, sakura_acc, label="sakura_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_curves.png"), dpi=200)
    plt.close()

# =========================
# NEW: Summary txt append
# =========================
def append_summary_to_txt(txt_path, epoch, num_epochs,
                          train_loss, train_acc,
                          test_loss, test_acc,
                          sakura_loss, sakura_acc,
                          gap_train_test, gap_train_sakura):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    block = (
        "============================================================\n"
        f"Epoch {epoch}/{num_epochs} SUMMARY\n"
        f"TRAIN : loss={train_loss:.4f}  acc={train_acc:.2f}%\n"
        f"TEST  : loss={test_loss:.4f}  acc={test_acc:.2f}%   (cured+notcured)\n"
        f"SAKURA: loss={sakura_loss:.4f}  acc={sakura_acc:.2f}%\n"
        f"GAP   : acc(train-test)={gap_train_test:+.2f}   acc(train-sakura)={gap_train_sakura:+.2f}\n"
        "============================================================\n\n"
    )

    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(block)

# =========================
# Train
# =========================
def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = LargerSimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    train_dataset = CartoDataset("train", 2)
    test_cured_dataset = CartoDataset("test_cured", 2)
    test_not_cured_dataset = CartoDataset("test_notcured", 2)

    sakura_test_dataset = CartoDatasetTestSakura("/mnt/disk2/carto/train-test-folder/", class_count=2)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)
    test_cured_loader = torch.utils.data.DataLoader(test_cured_dataset, batch_size=128, shuffle=False, drop_last=False)
    test_not_cured_loader = torch.utils.data.DataLoader(test_not_cured_dataset, batch_size=128, shuffle=False, drop_last=False)
    sakura_test_loader = torch.utils.data.DataLoader(sakura_test_dataset, batch_size=128, shuffle=False, drop_last=False)

    num_epochs = 100
    out_dir = "mdi-newdata-normal-sakura-combined"
    os.makedirs(out_dir, exist_ok=True)
    summary_txt_path = os.path.join(out_dir, "training_summary.txt")

    # =========================
    # Metric history (plot için)
    # =========================
    hist_epoch = []
    hist_train_loss = []
    hist_test_loss = []
    hist_sakura_loss = []
    hist_train_acc = []
    hist_test_acc = []
    hist_sakura_acc = []

    for epoch in range(num_epochs):
        # =========================
        # TRAIN
        # =========================
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loss_sum = 0.0
        train_seen = 0

        for (i, batch) in enumerate(train_loader):
            signals = batch[0].float().to(device)
            labels = batch[1].to(device)

            if len(labels) == 0:
                continue

            logits = model(signals)
            loss = criterion(logits, labels)

            bs = labels.size(0)
            train_loss_sum += loss.item() * bs
            train_seen += bs

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)

            total_train += bs
            correct_train += (preds == labels).sum().item()

            if i % 100 == 99 or i == len(train_loader) - 1:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, len(train_loader), running_loss / 10
                    )
                )
                running_loss = 0.0
                print("Train accuracy (so far): ", 100 * correct_train / max(1, total_train))

        # scheduler.step()  # istersen aç
        train_accuracy = 100 * correct_train / max(1, total_train)
        train_loss_epoch = train_loss_sum / max(1, train_seen)

        # =========================
        # TEST per-class acc (mevcut logic)
        # =========================
        model.eval()

        # --- cured
        correct_lvot_cured = correct_rvot_cured = 0
        total_lvot_cured = total_rvot_cured = 0

        print("Cured Test:")
        with torch.no_grad():
            for signals, labels, _ in test_cured_loader:
                signals = signals.float().to(device)
                labels = labels.to(device)

                logits = model(signals)
                preds = logits.argmax(dim=1)

                total_lvot_cured += (labels == 0).sum().item()
                total_rvot_cured += (labels == 1).sum().item()
                correct_lvot_cured += ((preds == labels) & (labels == 0)).sum().item()
                correct_rvot_cured += ((preds == labels) & (labels == 1)).sum().item()

        lvot_accuracy_cured = 100 * correct_lvot_cured / total_lvot_cured if total_lvot_cured > 0 else 0
        rvot_accuracy_cured = 100 * correct_rvot_cured / total_rvot_cured if total_rvot_cured > 0 else 0
        print(f"LVOT Test Cured Accuracy: {lvot_accuracy_cured:.2f}%")
        print(f"RVOT Test Cured Accuracy: {rvot_accuracy_cured:.2f}%")

        # --- notcured
        correct_lvot_notcured = correct_rvot_notcured = 0
        total_lvot_notcured = total_rvot_notcured = 0

        print("Not Cured Test:")
        with torch.no_grad():
            for signals, labels, _ in test_not_cured_loader:
                signals = signals.float().to(device)
                labels = labels.to(device)

                logits = model(signals)
                preds = logits.argmax(dim=1)

                total_lvot_notcured += (labels == 0).sum().item()
                total_rvot_notcured += (labels == 1).sum().item()
                correct_lvot_notcured += ((preds == labels) & (labels == 0)).sum().item()
                correct_rvot_notcured += ((preds == labels) & (labels == 1)).sum().item()

        lvot_accuracy_notcured = 100 * correct_lvot_notcured / total_lvot_notcured if total_lvot_notcured > 0 else 0
        rvot_accuracy_notcured = 100 * correct_rvot_notcured / total_rvot_notcured if total_rvot_notcured > 0 else 0
        print(f"LVOT Test Not Cured Accuracy: {lvot_accuracy_notcured:.2f}%")
        print(f"RVOT Test Not Cured Accuracy: {rvot_accuracy_notcured:.2f}%")

        # --- overall test acc (en doğru)
        test_total_correct = (
            correct_lvot_cured + correct_rvot_cured + correct_lvot_notcured + correct_rvot_notcured
        )
        test_total_total = total_lvot_cured + total_rvot_cured + total_lvot_notcured + total_rvot_notcured
        test_total_accuracy = 100 * test_total_correct / max(1, test_total_total)

        print(
            "Epoch [{}/{}], Train Acc: {:.2f}%, Test Acc (cured+notcured): {:.2f}%".format(
                epoch + 1, num_epochs, train_accuracy, test_total_accuracy
            )
        )

        # =========================
        # TEST loss (weighted cured+notcured)
        # =========================
        test_cured_loss, _ = eval_loss_acc(model, test_cured_loader, criterion, device)
        test_not_loss, _ = eval_loss_acc(model, test_not_cured_loader, criterion, device)

        n_cured = len(test_cured_dataset)
        n_not = len(test_not_cured_dataset)
        n_all = max(1, n_cured + n_not)
        test_loss_combined = (test_cured_loss * n_cured + test_not_loss * n_not) / n_all

        # =========================
        # SAKURA per-class acc + overall + loss+acc
        # =========================
        acc_sakura, overall_sakura = eval_loader_per_class(model, sakura_test_loader, class_count=2, device=device)
        sakura_loss, sakura_acc_overall = eval_loss_acc(model, sakura_test_loader, criterion, device)

        print("Sakura Test:")
        print(f"LVOT Sakura Test Accuracy: {acc_sakura.get(0, 0.0):.2f}%")
        print(f"RVOT Sakura Test Accuracy: {acc_sakura.get(1, 0.0):.2f}%")
        print(f"Overall Sakura Test Accuracy: {overall_sakura:.2f}%")

        # =========================
        # SUMMARY (console + txt)
        # =========================
        gap_train_test = train_accuracy - test_total_accuracy
        gap_train_sakura = train_accuracy - sakura_acc_overall

        summary_block = (
            "\n" + "=" * 60 + "\n"
            f"Epoch {epoch+1}/{num_epochs} SUMMARY\n"
            f"TRAIN : loss={train_loss_epoch:.4f}  acc={train_accuracy:.2f}%\n"
            f"TEST  : loss={test_loss_combined:.4f}  acc={test_total_accuracy:.2f}%   (cured+notcured)\n"
            f"SAKURA: loss={sakura_loss:.4f}  acc={sakura_acc_overall:.2f}%\n"
            f"GAP   : acc(train-test)={gap_train_test:+.2f}   acc(train-sakura)={gap_train_sakura:+.2f}\n"
            + "=" * 60 + "\n"
        )
        print(summary_block)

        append_summary_to_txt(
            txt_path=summary_txt_path,
            epoch=epoch + 1,
            num_epochs=num_epochs,
            train_loss=train_loss_epoch,
            train_acc=train_accuracy,
            test_loss=test_loss_combined,
            test_acc=test_total_accuracy,
            sakura_loss=sakura_loss,
            sakura_acc=sakura_acc_overall,
            gap_train_test=gap_train_test,
            gap_train_sakura=gap_train_sakura
        )

        # =========================
        # History + Plot Save (her epoch overwrite)
        # =========================
        hist_epoch.append(epoch + 1)
        hist_train_loss.append(train_loss_epoch)
        hist_test_loss.append(test_loss_combined)
        hist_sakura_loss.append(sakura_loss)

        hist_train_acc.append(train_accuracy)
        hist_test_acc.append(test_total_accuracy)
        hist_sakura_acc.append(sakura_acc_overall)

        save_metric_plots(
            out_dir=out_dir,
            epochs=hist_epoch,
            train_loss=hist_train_loss,
            test_loss=hist_test_loss,
            sakura_loss=hist_sakura_loss,
            train_acc=hist_train_acc,
            test_acc=hist_test_acc,
            sakura_acc=hist_sakura_acc
        )

        # =========================
        # Save model
        # =========================
        model_filename = (
            f"{out_dir}/model_{epoch+1}"
            f"-trL{train_loss_epoch:.4f}"
            f"-teL{test_loss_combined:.4f}"
            f"-saL{sakura_loss:.4f}"
            f"-testA{test_total_accuracy:.2f}"
            f"-sakuraA{sakura_acc_overall:.2f}.pth"
        )
        torch.save(model.state_dict(), model_filename)

    print("Training finished")

if __name__ == "__main__":
    train_model()