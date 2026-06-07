# Few-Shot Hyperspectral Grain Classification

**"Hyperspectral Imaging-Based Grain Quality Assessment with limited labelled data"**  
Priyabrata Karmakar, Manzur Murshed, Shyh Wei Teng

---

## Overview

This repository provides code to reproduce the FSL-based grain classification experiments from the paper. The pipeline has five stages:

1. Download the HSI database
2. Organise files into train/test directories
3. Train the prototypical network
4. Compute Collective Class Prototypes (CCP)
5. Evaluate and visualise results

---

## Requirements

Python 3.9 or higher is recommended. Install all dependencies:

```bash
pip install -r requirements.txt
```

**Note on spacepy:** spacepy requires the NASA CDF library. Install it before `pip install spacepy`:
- Linux: `sudo apt-get install libcdf-dev`
- Windows/macOS: download from https://cdf.gsfc.nasa.gov/html/sw_and_docs.html

---

## Step 1 — Download the data

The HSI database is publicly available from Dreier et al. (2022) hosted at:
https://erda.ku.dk/archives/89a8b2b044d458e487fc2ce56927f420/published-archive.html

Scripts to automate this are in `data_access/`.

```bash
cd data_access

# Step 1a: scrape download links (requires Chrome browser and chromedriver installed)
python scrape_data_links.py
# This produces HSI_GT_Source.csv listing all file URLs.

# Step 1b: download all .cdf and .jpg files
# Edit data_download.py to set store_path to your desired download location, then:
python data_download.py
```

`HSI_GT_Source.csv` is already included in `data_access/` so you can skip Step 1a
and go directly to Step 1b.

If the download is interrupted, uncomment the resume section in `data_download.py`
before re-running — it will skip files already downloaded.

---

## Step 2 — Organise files

```bash
python data_preparation.py \
    --download_dir /path/to/downloaded_files \
    --csv_path     data_access/HSI_GT_Source.csv \
    --output_dir   /path/to/organised_data \
    --max_train    360 \
    --max_test     1500 \
    --seed         42
```

This creates:
```
organised_data/
  fsl_train/    <- 360 .cdf files per class (from database Test split)
  fsl_test/     <- up to 1500 .cdf files per class (from database Train split)
```

**Why the split is inverted:** The database's Test folder has fewer images per class
and is used as the FSL training set (360 per class). The larger Train folder is used
as the FSL evaluation set (up to 1500 per class). This follows the convention in the
original database paper.

---

## Step 3 — Train

### 8-way classification (all classes, paper best config)

```bash
python train.py \
    --train_dir   /path/to/organised_data/fsl_train \
    --save_dir    /path/to/checkpoints/8way \
    --k_shot      5 \
    --query_size  10 \
    --epochs      50 \
    --channels    204 \
    --se_attention \
    --reduction_ratio 8
```

### 6-way classification (exclude Rye and WH5, for novel class generalisation)

```bash
python train.py \
    --train_dir        /path/to/organised_data/fsl_train \
    --save_dir         /path/to/checkpoints/6way \
    --k_shot           5 \
    --query_size       10 \
    --epochs           50 \
    --channels         204 \
    --se_attention \
    --reduction_ratio  8 \
    --exclude_classes  Rye_Midsummer Wheat_H5
```

### Without SE attention (ablation)

```bash
python train.py \
    --train_dir  /path/to/organised_data/fsl_train \
    --save_dir   /path/to/checkpoints/8way_no_attn \
    --k_shot 5 --query_size 10 --epochs 50 --channels 204
```

### With spectral binning (102 channels, ablation)

```bash
python train.py \
    --train_dir      /path/to/organised_data/fsl_train \
    --save_dir       /path/to/checkpoints/8way_102ch \
    --k_shot 5 --query_size 10 --epochs 50 \
    --channels       204 \
    --binning_factor 2
```

After training completes, `save_dir` will contain:
- `best_model.pth` — best model by training loss
- `model_epoch_NNN.pth` — checkpoint at each epoch
- `episode_prototypes/` — per-episode prototypes for CCP computation
- `classes.txt` — ordered class list

---

## Step 4 — Compute Collective Class Prototypes (CCP)

```bash
python compute_ccp.py \
    --save_dir /path/to/checkpoints/8way \
    --output   /path/to/checkpoints/8way/ccp.pth
```

This averages all per-episode prototypes from `episode_prototypes/` into one
CCP tensor of shape `(n_way, feature_dim)`.

---

## Step 5 — Evaluate

### 8-way evaluation with CCP (recommended, paper Table V)

```bash
python evaluate.py \
    --test_dir     /path/to/organised_data/fsl_test \
    --model_path   /path/to/checkpoints/8way/best_model.pth \
    --ccp_path     /path/to/checkpoints/8way/ccp.pth \
    --classes_file /path/to/checkpoints/8way/classes.txt \
    --k_shot 5 --query_size 10 \
    --channels 204 --se_attention --reduction_ratio 8
```


### 8-way evaluation with support sets (paper Table VII, for CCP comparison)

```bash
python evaluate.py \
    --test_dir     /path/to/organised_data/fsl_test \
    --model_path   /path/to/checkpoints/8way/best_model.pth \
    --classes_file /path/to/checkpoints/8way/classes.txt \
    --k_shot 5 --query_size 10 \
    --channels 204 --se_attention
```


### Novel class evaluation (6-way model, paper Table XI)

```bash
python evaluate_novel.py \
    --test_dir         /path/to/organised_data/fsl_test \
    --model_path       /path/to/checkpoints/6way/best_model.pth \
    --classes_file     /path/to/checkpoints/6way/classes.txt \
    --excluded_classes Rye_Midsummer Wheat_H5 \
    --strategy         both \
    --k_shot 5 --query_size 10 \
    --channels 204 --se_attention --reduction_ratio 8
```

Expected:
- Strategy 1 (support = excluded classes only): ~98.33%
- Strategy 2 (support = all 8 classes): ~83.89%

---

## Step 6 — Visualise

```bash
python visualize.py \
    --test_dir     /path/to/organised_data/fsl_test \
    --model_path   /path/to/checkpoints/8way/best_model.pth \
    --ccp_path     /path/to/checkpoints/8way/ccp.pth \
    --proto_dir    /path/to/checkpoints/8way/episode_prototypes \
    --classes_file /path/to/checkpoints/8way/classes.txt \
    --output_dir   figures/ \
    --channels 204 --se_attention \
    --k_shot 5 --query_size 10
```

Produces in `figures/`:
- `tsne_prototypes.png` — episode prototypes (circles) and CCPs (stars) in 2D
- `attention_heatmap.png` — SE attention weights across spectral channels per class
- `confusion_matrix.png` — classification result as percentage confusion matrix

---

## File structure

```
reproducible_code/
  data_access/          scripts and CSV for downloading the database
  dataset.py            CDF data loader and preprocessing (HSIDataLoader, load_cdf)
  models.py             SEBlock, ResNet18Backbone, ResNet18BackboneWithSE, PrototypicalNetwork
  train.py              training script
  compute_ccp.py        CCP computation from episode prototypes
  evaluate.py           evaluation with CCP or support sets
  evaluate_novel.py     novel class evaluation (Strategy 1 and 2)
  visualize.py          t-SNE, attention heatmap, confusion matrix
  data_preparation.py   organise downloaded files into train/test directories
  requirements.txt
  README.md
```

---

## Key hyperparameters

| Parameter | Paper value | Argument |
|---|---|---|
| Support set size | 5 | `--k_shot` |
| Query set size | 10 | `--query_size` |
| Training epochs | 50 | `--epochs` |
| Spectral channels | 204 | `--channels` |
| SE reduction ratio | 8 | `--reduction_ratio` |
| Learning rate | 0.001 | `--lr` |
| Classes for 8-way | all 8 | (default) |
| Classes excluded for 6-way | Rye_Midsummer, Wheat_H5 | `--exclude_classes` |

---

## Citation

If you use this code, please cite:

```
Priyabrata Karmakar, Manzur Murshed, Shyh Wei Teng,
"Hyperspectral Imaging-Based Grain Quality Assessment with limited labelled data",
IEEE Transactions on Emerging Topics in Computing, 2025.
```

Database citation:

```
E. S. Dreier, K. M. Sorensen, T. Lund-Hansen, B. M. Jespersen, and K. S. Pedersen,
"Hyperspectral imaging for classification of bulk grain samples with deep convolutional
neural networks," Journal of Near Infrared Spectroscopy, vol. 30, no. 3, pp. 107-121, 2022.
```
