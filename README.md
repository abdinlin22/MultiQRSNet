# LVOT–RVOT ECG Classification with MDI

This repository contains the code for training and evaluating a 1D CNN model that classifies ECG signals into **LVOT** and **RVOT** using 12 ECG leads + 1 MDI channel (13-channel input).

The main scripts are:

- `data_aug_mdi.py` — training + evaluation on standard test sets  
- `data_aug_mdi_test.py` — evaluation on the standard test sets using a saved model  
- `data_aug_mdi_test_sakura.py` — evaluation on the external **Sakura** test set  

---

## 1. Dataset

### 1.1. Download

The dataset is provided as a ZIP file on Google Drive:

**Dataset ZIP:**  
[`dataset_link`](https://drive.google.com/drive/folders/18Sr4_fAU2R6-MhkKye8zR3zzcQ4lQYC8?usp=drive_link)

After downloading and extracting, you must update the dataset path inside the scripts:

- In `data_aug_mdi.py`:

```python
base_path = "/path/to/train-test-folder/"
```

- In `data_aug_mdi_test.py`:

```python
base_path = "/path/to/train-test-folder/"
```

- In `data_aug_mdi_test_sakura.py`:

```python
DATA_ROOT = "/path/to/train-test-folder"
```

---

### 1.2. Dataset Structure

After extracting the ZIP file, the dataset structure must be:

```
train-test-folder/
├── LVOT/
│   ├── train_cured/
│   ├── train_notcured/
│   ├── test_cured/
│   ├── test_notcured/
│   └── test_sakura/
├── RLVOT/
│   ├── train_cured/
│   ├── train_notcured/
│   ├── test_cured/
│   ├── test_notcured/
│   └── test_sakura/
└── RVOT/
    ├── train_cured/
    ├── train_notcured/
    ├── test_cured/
    ├── test_notcured/
    └── test_sakura/
```

Notes:

- `train_*` folders are used for training.
- `test_cured` and `test_notcured` are the standard evaluation sets.
- `test_sakura` is an external dataset and is never used during training.

All ECG samples are `.txt` files and these txt files are used during both training and tests.

---

## 2. Training + Online Evaluation on Standard Test Sets

Training is performed using:

```
data_aug_mdi.py
```

This script:

- Loads `train_cured` and `train_notcured` from LVOT and RVOT  
- Calculates the MDI channel and forms 13-channel inputs  
- Trains a 1D CNN model (LargerSimpleNet)  
- Evaluates at each epoch on `test_cured` and `test_notcured`  
- Saves checkpoints automatically

### 2.1. Update Paths

In the script, modify:

```python
base_path = "/path/to/train-test-folder/"
```

### 2.2. Create model output directory

The script saves checkpoints in:

```
mdi-newdata-2class-notcuredchanged/
```

If needed, create it manually:

```
mkdir -p mdi-newdata-2class-notcuredchanged
```

### 2.3. Run Training

Execute:

```
python data_aug_mdi.py
```

The script prints:

- Training loss  
- LVOT and RVOT accuracy values  
- Test set performance (for `test_cured` and `test_notcured`)  
- Saved checkpoint filenames  

Example saved model:

```
mdi-newdata-2class-notcuredchanged/model_7-98.21782178217822.pth
```

---

## 3. Offline Evaluation on Standard Test Sets (Saved Model)

If you only want to evaluate a **saved checkpoint** on the standard test sets (without re-running training), use:

```
data_aug_mdi_test.py
```

This script:

- Loads `test_cured` and `test_notcured` datasets  
- Loads a trained checkpoint  
- Runs inference  
- Prints per-class accuracies and total accuracy

### 3.1. Update Dataset Path

Inside `data_aug_mdi_test.py`, set:

```python
base_path = "/path/to/train-test-folder/"
```

### 3.2. Select the Model Checkpoint

At the bottom of the script:

```python
if __name__ == "__main__":
    MODEL_PATH = "mdi-newdata-2class-notcuredchanged/model_7-98.21782178217822.pth"
    evaluate_saved_model(MODEL_PATH)
```

Change `MODEL_PATH` to any checkpoint you want to test.

### 3.3. Run Offline Test Evaluation

Execute:

```
python data_aug_mdi_test.py
```

You will see:

- LVOT Test Cured Accuracy  
- RVOT Test Cured Accuracy  
- LVOT Test Not Cured Accuracy  
- RVOT Test Not Cured Accuracy  
- Total Test Accuracy (cured + not cured combined)

---

## 4. Evaluation on the Sakura Test Set

To evaluate the trained model on the external Sakura dataset, use:

```
data_aug_mdi_test_sakura.py
```

This script:

- Loads a trained checkpoint  
- Reads LVOT/RVOT `test_sakura` samples  
- Computes MDI  
- Runs inference only (no training)  
- Prints per-class and overall accuracy

### 4.1. Update Script Settings

Inside `data_aug_mdi_test_sakura.py`, set:

```python
EVAL_CHECKPOINT = "/path/to/model.pth"
DATA_ROOT = "/path/to/train-test-folder"
CLASS_COUNT = 2
```

If you want to use a pretrained model from Google Drive:

**Pretrained checkpoint:**  
[`pretrained_model`](https://drive.google.com/drive/folders/1LmCotNQU11439YSarSdTx9Vceyz0Av38?usp=drive_link)

Download it and update `EVAL_CHECKPOINT`.

### 4.2. Run Sakura Evaluation

Execute:

```
python data_aug_mdi_test_sakura.py
```

This prints:

- LVOT accuracy on Sakura  
- RVOT accuracy on Sakura  
- (Optional) RLVOT accuracy  
- Overall Sakura test accuracy  

---

## 5. Full Pipeline Summary

```
1. Download dataset from Drive.
2. Extract to: /path/to/train-test-folder
3. Update "base_path" in data_aug_mdi.py and data_aug_mdi_test.py
   and "DATA_ROOT" in data_aug_mdi_test_sakura.py
4. (Optional) Download pretrained model and update EVAL_CHECKPOINT / MODEL_PATH
5. Train the model:
   python data_aug_mdi.py
6. Evaluate a saved model on the standard test sets:
   python data_aug_mdi_test.py
7. Evaluate the same model on the external Sakura test set:
   python data_aug_mdi_test_sakura.py
```

This README fully describes how to reproduce training and evaluation with the provided scripts and dataset.
