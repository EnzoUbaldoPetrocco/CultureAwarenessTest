#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

import cv2

sys.path.insert(1, "../")
from GradCam.gradCam import GradCAM
from Utils.FileManager.FileManager import FileManagerClass
from Processing.processing import ProcessingClass
from math import floor
import tensorflow as tf
import os
import gc
import random
from datetime import datetime
import numpy as np
from tensorflow.keras import mixed_precision

random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# tf.config.set_soft_device_placement(True)


memory_limit = 7000
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
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
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
else:
    print("no gpus")


percent = 0.05
standard = 1
c = 0
# lamp = 1

verbose_param = 1
n = 1000
bs = 2
learning_rate = 5e-4
val_split = 0.2
test_split = 0.1
epochs = 15

g_gaugs = np.logspace(-4, 0, 6)
g_aug = g_gaugs[0]
eps = np.logspace(-6, -1, 5)
test_eps = [0.0005, 0.001, 0.005]
mult = 0.25
cs = [0, 1, 2]
ks = [1]
imb = 0
basePath = "./Diff_step_plot/"
lamp = 1


procObj = ProcessingClass(
    shallow=0, lamp=lamp, gpu=False, memory_limit=memory_limit, basePath=basePath
)
procObj.process(
    standard=standard,
    type="DL",
    verbose_param=verbose_param,
    culture=c,
    percent=percent,
    val_split=val_split,
    test_split=test_split,
    n=n,
    augment=k,
    diffusion = 1
)
# NoAUg
gc.collect()
tf.keras.backend.clear_session()
