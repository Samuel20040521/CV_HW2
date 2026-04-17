#!/bin/bash

# 建立 checkpoint 資料夾確保路徑存在
mkdir -p checkpoint

echo "Downloading ResNet18 best weights..."
gdown --id 1S7Lyt2JdPAoai_x_tHt-RVRyHaq2A2R0 -O checkpoint/resnet18_best.pth

echo "Downloading MyNet best weights..."
gdown --id 10S4dcz1VDqIBM0w6PZGQ6XA6HYna7Xgo -O checkpoint/mynet_best.pth

echo "Download completed. Best models are in the checkpoint/ directory."
