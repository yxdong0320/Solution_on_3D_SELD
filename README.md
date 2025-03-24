# Solution_on_3D_SELD

The program ranked first in Audio-only track of DCASE2024 Challenge task3.

Papers related to the code have been accepted by the ICASSP2025 and are visible in the address https://arxiv.org/abs/2501.10755.

## Installation Guide

```python
cd Solution_on_3D_SELD
pip install -r requirements.txt
```

## Extract Audio Features

Change dataset path and feature path in `utils/cls_tools/parameters.py` script

```python
dataset_dir = '...'
feat_label_dir = '...'

# Extract audio features
python utils/cls_tools/batch_feature_extraction.py
```

## Package in LMDB Format

Change feature path and lmdb path in `utils/lmdb_tools/convert_lmdb.py` script

```python
npy_data_dir = '...'
npy_label_dir = '...'
lmdb_out_dir = '...'

# Package lmdb
python utils/lmdb_tools/convert_lmdb.py
```

## Training

Change lmdb path, ground truth label path, feature normalization file path, and training result path in `config/A_RC_SED-SDE.yaml`

```python
data:
  train_lmdb_dir: '...'
  test_lmdb_dir: '...'
  ref_files_dir: '...'
  norm_file: '...'
result:
  log_output_path: '...'
  log_interval: 100
  checkpoint_output_dir: '...'
  dcase_output_dir: '...'
# Load pre-trained file
model:
  pre-train: False
  # pre-train_model: 'checkpoint_epoch146_step39858'
# Model training
python train_A_sedsde.py -c A_RC_SED-SDE
```

## Testing

Change lmdb path, ground truth label path, feature normalization file path, and training result path in `config/A_RC_SED-SDE.yaml`

```python
data:
  train_lmdb_dir: '...'
  test_lmdb_dir: '...'
  ref_files_dir: '...'
  norm_file: '...'
result:
  log_output_path: '...'
  log_interval: 100
  checkpoint_output_dir: '...'
  dcase_output_dir: '...'
model:
  pre-train: True
  pre-train_model: 'checkpoint_epoch146_step39858'
# Model testing
python test_A.py -c A_RC_SED-SDE
```

Please feel free to contact me at yxdong0320@mail.ustc.edu.cn. if you have any questions about the implementation or encounter any issues while using the code. I'm happy to provide additional information or assistance as needed.