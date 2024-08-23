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

random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_asyn"

# tf.config.set_soft_device_placement(True)

memory_limit = 8000
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


percents = [0.05]
standard = 1
# lamp = 1

verbose_param = 1
n = 1000
bs = 2
learning_rate = 5e-4
val_split = 0.2
test_split = 0.1
epochs = 15
class_divisions = [1,0]
imbalances = [1, 0]

g_gaugs = np.logspace(-4, 0, 6)
test_g_augs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
eps = np.logspace(-4, 0, 6)
test_eps = [0.0005, 0.001, 0.005]
mult = 0.25
cs = [2,1,0]
ks = [0]

basePath = "./PROVA/"
# with tf.device("/CPU:0"):
for i in range(4):
 for percent in percents:
    for lamp in [0,1]:
        procObj = ProcessingClass(
            shallow=0,
            lamp=lamp,
            gpu=False,
            memory_limit=memory_limit,
            basePath=basePath,
        )
        for imb in imbalances:
         for c in cs:
            for k in ks:
                for ep in eps:
                    for cl_div in class_divisions:
                        if k%2==1:
                            for g_aug in g_gaugs:
                                model = None
                                print(f"Training->aug={k%2};adv={floor(k/2)}")
                                procObj.process(
                                    standard=standard,
                                    type="DL",
                                    verbose_param=verbose_param,
                                    learning_rate=learning_rate,
                                    epochs=epochs,
                                    batch_size=bs,
                                    lambda_index=0,
                                    culture=c,
                                    percent=percent,
                                    val_split=val_split,
                                    test_split=test_split,
                                    n=n,
                                    augment=k % 2,
                                    gaug=g_aug,
                                    adversary=1,
                                    eps=ep,
                                    mult=mult,
                                    imbalanced=imb,  
                                    class_division= cl_div
                                )
                                # NoAUg
                                print(f"Testing->aug={0};adv={0}")
                                procObj.test(
                                    standard=standard,
                                    culture=c,
                                    augment=0,
                                    gaug=0,
                                    adversary=0,
                                    eps=test_eps,
                                )
                                procObj.partial_clear(basePath)
                    
                        else:
                            model = None
                            print(f"Training->aug={k%2};adv={floor(k/2)}")
                            procObj.process(
                                standard=standard,
                                type="DL",
                                verbose_param=verbose_param,
                                learning_rate=learning_rate,
                                epochs=epochs,
                                batch_size=bs,
                                lambda_index=0,
                                culture=c,
                                percent=percent,
                                val_split=val_split,
                                test_split=test_split,
                                n=n,
                                augment=k % 2,
                                gaug=0,
                                adversary=1,
                                eps=ep,
                                mult=mult,
                                imbalanced=imb,
                                class_division= cl_div
                            )
                            # NoAUg
                            print(f"Testing->aug={0};adv={0}")
                            procObj.test(
                                standard=standard,
                                culture=c,
                                augment=0,
                                gaug=0,
                                adversary=0,
                                eps=test_eps,
                            )
                            procObj.partial_clear(basePath)
                    