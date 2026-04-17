# Computer Vision HW2 Report

- Student ID: B11202015
- Name: 鄧恩陞

## Part 1. (10%)
- Plot confusion matrix of two settings. (i.e. Bag of sift and tiny image) (5%)
Ans:
Tiny Image Confusion Matrix:
![Tiny Image Confusion Matrix](p1/tiny_image.png)

Bag of SIFT Confusion Matrix:
![Bag of SIFT Confusion Matrix](p1/bag_of_sift.png)

- Compare the results/accuracy of both settings and explain the result. (5%)
Ans:
**Accuracy**: 
- Tiny image: 24.9%
- Bag of SIFT: 60.8%

**Explanation**: 
The **Tiny image** feature achieves low accuracy because it merely downsizes the image to 16x16 and flattens it. This brutally destroys spatial structures, local textures, and fine details, creating a feature that is highly sensitive to shifts, rotations, and lighting changes.

Conversely, the **Bag of SIFT** approach achieves a robust 60.8% accuracy. It extracts dense local descriptors (SIFT) that effectively encode edge orientations in a robust manner. By employing K-Means clustering (size=400), we project these SIFT descriptors into "Visual Words". Representing the image as a histogram of these visual words captures the abstract, global semantic configuration of the scene (e.g., repeating leaf textures for forests or strong linear edges for cityscapes), allowing the K-Nearest Neighbor classifier to make much better comparisons.

## Part 2. (25%)
- Report accuracy of both models on the validation set. (2%)
Ans:
|                 |     A (MyNet)    |     B (ResNet18)    |
|-----------------|----------|----------|
|     accuracy    |    86.18%  |    92.98%  |

- Print the network architecture & number of parameters of both models. What is the main difference between ResNet and other CNN architectures? (5%)
Ans:
**Number of parameters**:
- **MyNet**: 3,231,626
- **ResNet18**: 11,173,962

**Network Architecture**:
**MyNet**: Modified into a VGG-like deep architecture. It consists of multiple 3x3 convolutional blocks (`Conv2d(3x3) -> BatchNorm2d -> ReLU`), progressively scaling channels from 32 to 256. `MaxPool2d(2x2)` layers are used for downsampling. The features are then flattened and passed into a fully-connected layer with `BatchNorm1d` and `Dropout(0.5)` for regularization before outputting 10 classes.

**ResNet18 (Modified)** retains the deep structure characteristic of the original ResNet architecture. However, the large 7x7 initial convolutional filter with stride 2 and the consecutive MaxPool layer were downscaled significantly into a single `Conv2d(3, 64, 3x3, stride=1, padding=1)`. This minimizes the abrupt initial 4x downsampling which would otherwise crush our small 32x32 feature maps.

**Main Difference**:
ResNet fundamentally introduces **Residual (Skip) Connections**. Deep CNNs usually suffer from the vanishing gradient problem. ResNet allows features (or identity maps) to bypass multiple weight layers ($H(x) + x$), meaning gradients during backpropagation flow effortlessly directly to the early layers, enabling training on significantly deeper structures without degradation.

- Plot four learning curves (loss & accuracy) of the training process (train/validation) for both models. Total 8 plots. (8%)
Ans: 
**ResNet18**:
![ResNet18 Accuracy](p2/experiment/resnet18_2026_04_17_15_36_11_default/log/accuracy.png)
![ResNet18 Loss](p2/experiment/resnet18_2026_04_17_15_36_11_default/log/loss.png)

**MyNet**:
![MyNet Accuracy](p2/experiment/mynet_2026_04_17_15_51_20_default/log/accuracy.png)
![MyNet Loss](p2/experiment/mynet_2026_04_17_15_51_20_default/log/loss.png)

- Briefly describe what method do you apply on your best model? (e.g. data augmentation, model architecture, loss function, etc) (10%)
Ans:
Our highest-performing model is the modified **ResNet18**, and it leverages the following techniques:
1. **Model Architecture Resizing & Initialization**: The original pretrained ResNet18 is built for 224x224 ImageNet inputs. To map this properly on our 32x32 dataset context, we replaced `resnet.conv1` to a 3x3 kernel (stride=1, padding=1), stripped away `resnet.maxpool`, and applied Kaiming Normalization to stabilize the initial gradients.
2. **Data Augmentation**: For the training images, a PyTorch transformation pipeline `transforms.RandomCrop(32, padding=4)` coupled with `transforms.RandomHorizontalFlip()` introduces strong rotational/translational invariances suppressing extreme overfitting on our 20k samples.
3. **Training & Regularization**: We completely turned off the uncalibrated Semi-supervised Learning (Pseudo-labeling) at early epochs since the noise was severely holding back accuracy. We switched to an SGD optimizer with high momentum (0.9) and strong L2 weight decay (`1e-4`~`5e-4`), alongside a significantly larger batch size (128) for stabler Batch Normalization statistics and robust convergence.
