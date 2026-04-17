# ============================================================================
# File: config.py
# Date: 2026-03-27
# Author: TA
# Description: Experiment configurations.
# ============================================================================

################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name = 'default'  # name of experiment

# Model Options
model_type = 'resnet18'  # 'mynet' or 'resnet18'

# Learning Options
epochs = 50                # train how many epochs
batch_size = 128           # increased batch size for better BatchNorm stability
use_adam = False           # Adam or SGD optimizer
lr = 0.01                  # learning rate for SGD
milestones = [25, 40]      # reduce learning rate at 'milestones' epochs