#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

import cv2

sys.path.insert(1, "../")
from Utils.FileManager.FileManager import FileManagerClass
from Processing.processing import ProcessingClass
from math import floor
import tensorflow as tf

tf.config.set_soft_device_placement(True)

percent = 0.05
standards = [0, 1]
lamps = [0, 1]

verbose_param = 1
n = 1000
bs = 2
learning_rate = 5e-4
val_split = 0.2
test_split = 0.1
epochs = 15

g_aug = 0.1
eps = 0.03
mult = 0.25
memory_limit = 2700
cs = [2, 1, 0]

for lamp in lamps:
    procObj = ProcessingClass(shallow=0, lamp=lamp, gpu=True, memory_limit=memory_limit)
    with tf.device("/CPU:0"):
        #only european cultures
        if lamp:
            c = 1 #french
        else: 
            c = 2 #scandinavian
        print(f"Training...")
        procObj.process(
            standard=1,
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
            augment=0,
            g_rot=g_aug,
            g_noise=g_aug,
            g_bright=g_aug,
            adversary=1,
            eps=eps,
            mult=mult,
            complete = 1
        )
        procObj.save_model()
        procObj.partial_clear()
