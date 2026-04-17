# Computer Vision HW2

This repository contains the implementation for Computer Vision Homework 2 (Spring 2026), including Bag-of-Words Scene Recognition (Part 1) and CNN Image Classification with Semi-supervised learning (Part 2).

## 1. Environment Setup

We strictly use `micromamba` (or `conda`) to manage our dependencies for reproducibility.

```bash
# Create environment
micromamba create -n cv_hw2 python=3.8 -y
micromamba activate cv_hw2

# Install dependencies (from root directory)
pip install -r requirements_py38.txt
micromamba install -c conda-forge cyvlfeat -y
```

## 2. Part 1: Bag-of-Words Scene Recognition

In this section, we extract SIFT features to construct a Visual Vocabulary with K-Means and calculate Bag-of-Words histograms, followed by a Custom K-Nearest Neighbor classifier.

To train the Bag-of-Words and Tiny Image algorithms, simply run:
```bash
cd p1
bash p1_run.sh
```

## 3. Part 2: CNN Image Classification & Semi-Supervised Learning

We implemented a simple Convolution network `MyNet` from scratch, and also adapted the `ResNet18` model for 32x32 image dimensions to push past the Strong Baseline. To further boost accuracy, we appended an experimental Pseudo-Labeling strategy using the `unlabel` dataset inside our training phase.

### A. Download Pretrained Checkpoints (For TA Verification)
To verify the performance on the `test` or `val` split without training from scratch, you can use `gdown` to download our best `.pth` models directly from Google Drive into the `p2/checkpoint/` repository.

1. Create the checkpoint directory:
```bash
cd p2
mkdir -p checkpoint
```

2. Download your checkpoints via `gdown` (**Note: Replace the Google Drive `<FILE_ID>` with the actual ID after you upload them!**):
```bash
# For ResNet18 model
gdown --id 1S7Lyt2JdPAoai_x_tHt-RVRyHaq2A2R0 -O checkpoint/resnet18_best.pth

# For MyNet model (If necessary)
gdown --id 10S4dcz1VDqIBM0w6PZGQ6XA6HYna7Xgo -O checkpoint/mynet_best.pth
```

### B. Testing / Inference
After the `.pth` weights are successfully seated inside the `checkpoint/` directory, evaluate the code by running:
```bash
# This script internally calls p2_inference.py and p2_eval.py
bash p2_run_test.sh
```
This automatically produces a `pred.csv` and outputs the result via the TA's evaluation script.

### C. Training from scratch
To observe the entire training loop (including our semi-supervised pseudo-labeling augmentation technique), run:
```bash
bash p2_run_train.sh
```
