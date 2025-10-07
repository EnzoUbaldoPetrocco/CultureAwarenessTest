#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

import cv2

sys.path.insert(1, "../")
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
from Utils.Data.deep_paths import DeepStrings
from Utils.Data.Data import DataClass
from copy import deepcopy
from Utils.Data.Data import PreprocessingClass
from matplotlib import pyplot as plt
import keras
from keras import layers

random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# tf.config.set_soft_device_placement(True)


memory_limit = 1000
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


percent = 0.9
standard = 1
culture = 0
verbose_param = 1
n = 1000

ks = [1]
imb = 0
basePath = "./"
lamps = [0,1]
for lamp in lamps:
    strObj = DeepStrings()
    if lamp:
        paths = strObj.lamp_paths
    else:
        paths = strObj.carpet_paths_str
    if paths:
        dataobj = DataClass(paths)
    else:
        raise Exception("Carpet Problem has not been tackled in shallow learning")


    dataobj.prepare(
        standard=standard,
        culture=culture,
        percent=percent,
        adversarial=0,
    )
    images = deepcopy(dataobj.Xt) 
    images = np.asarray(images)[:,0] / 255.0
    shape = np.shape(images[0])
    del dataobj
    prepObj = PreprocessingClass()
    for c in range(3):
        fig, ax = plt.subplots(nrows=2, ncols=3)
        ax[0,0].imshow(images[c])
        ax[0,0].get_yaxis().set_visible(False)
        ax[0,0].get_xaxis().set_visible(False)
        print(images)
        for i, g in enumerate(np.logspace(-4, 0, 5)):
            row = 0 if i < 2 else 1
            col = ((i % 3)+1) % 3
            print(f"row is {row}, col is {col}")
            data_augmentation = keras.Sequential(
                    [
                        layers.RandomFlip("horizontal"),
                        layers.RandomRotation(0.01),
                        layers.GaussianNoise(g),
                        #tf.keras.layers.RandomBrightness(0.01),
                        layers.RandomZoom(g, g),
                        layers.Resizing(shape[0], shape[1]),
                    ]
                )
            X_augmented = data_augmentation(tf.expand_dims(images[c], axis=0), training = True)
            ax[row,col].get_yaxis().set_visible(False)
            ax[row,col].get_xaxis().set_visible(False)
            ax[row, col].imshow(X_augmented[0])
        pt = basePath+f"/standard_augmentation/lamp={lamp}/culture={c}/img.pdf"
        fObj = FileManagerClass(pt)
        del fObj
        plt.savefig(pt)
        #plt.show()
            
    # NoAUg
    gc.collect()
    tf.keras.backend.clear_session()
