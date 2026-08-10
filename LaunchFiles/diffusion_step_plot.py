#!/usr/bin/env python
"""
Diffusion Step Plotting Launcher.

This script executes data preprocessing and diffusion model sample generation 
visualization across diffusion steps for cultural bias analysis.

Author: Enzo Ubaldo Petrocco
"""

import sys
import os
import gc
import random
from math import floor
from datetime import datetime
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Insert root directory path for imports
sys.path.insert(1, "../")
from GradCam.gradCam import GradCAM
from Utils.FileManager.FileManager import FileManagerClass
from Processing.processing import ProcessingClass

# ==============================================================================
# SEED & HARDWARE SETUP
# ==============================================================================
random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Assign target GPU ID

# Memory limit setting for TensorFlow GPU memory allocation
memory_limit = 7000
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [
                tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=memory_limit
                )
            ],
        )
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(f"[GPU] Physical: {len(gpus)}, Logical: {len(logical_gpus)}")
    except RuntimeError as e:
        print(f"[GPU Warning] {e}")
else:
    print("[GPU Warning] No GPUs detected. Running on CPU.")

# ==============================================================================
# PIPELINE CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
percent = 0.05            # 5% minority culture data ratio
standard = 1              # Standard control model
c = 0                     # Target majority culture ID
verbose_param = 1         # Verbose output flag
n = 1000                  # Sample cap per class/culture
bs = 2                    # Batch size for diffusion sampling
learning_rate = 5e-4      # Learning rate
val_split = 0.2           # Validation split fraction
test_split = 0.1          # Test split fraction
epochs = 15               # Training epoch count

g_gaugs = np.logspace(-4, 0, 6)
g_aug = g_gaugs[0]
eps = np.logspace(-6, -1, 5)
test_eps = [0.0005, 0.001, 0.005]
mult = 0.25
cs = [0, 1, 2]            # Culture IDs list
ks = [1]
imb = 0                   # Imbalance flag
basePath = "./Diff_step_plot/"  # Directory path to save output diffusion step plots
lamp = 0                  # 0 = Carpet dataset, 1 = Lamp dataset

# ==============================================================================
# PIPELINE INITIALIZATION & EXECUTION
# ==============================================================================
procObj = ProcessingClass(
    shallow=0,
    lamp=lamp,
    gpu=False,
    memory_limit=memory_limit,
    basePath=basePath
)

print("[Diffusion Step Plot] Starting data processing with diffusion model enabled...")
procObj.process(
    standard=standard,
    type="DL",
    verbose_param=verbose_param,
    culture=c,
    percent=percent,
    val_split=val_split,
    test_split=test_split,
    n=n,
    augment=1,
    diffusion=1
)

# ==============================================================================
# CLEANUP
# ==============================================================================
gc.collect()
tf.keras.backend.clear_session()
print("[Diffusion Step Plot] Execution completed and sessions cleared.")

