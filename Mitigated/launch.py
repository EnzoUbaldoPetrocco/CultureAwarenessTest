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

tf.config.set_soft_device_placement(True)

percents = [0.05]
standard = 1
#lamp = 1

verbose_param = 0
n = 1000
bs = 2
learning_rate = 5e-4
val_split = 0.2
test_split = 0.1
epochs = 15

g_gaugs = [0.0005, 0.001, 0.05, 0.1, 0.5, 0.75]
test_g_augs = [0.01, 0.05, 0.1]
eps = 0.03
test_eps = [0.0005, 0.001, 0.005]
mult = 0.25
memory_limit = 3000
cs = [2,1,0]
ks = [1]

basePath = './PREDICT/'


with tf.device("/CPU:0"):
    for lamp in [1,0]:
        procObj = ProcessingClass(shallow=0, lamp=lamp, gpu=True, memory_limit=memory_limit, basePath=basePath)
        for percent in percents:
            for c in cs:
                for k in ks:
                        for g_aug in g_gaugs:
                            model = None
                            for i in range(5):
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
                                    adversary=floor(k / 2),
                                    eps=eps,
                                    mult=mult,
                                )
                                # NoAUg
                                print(f"Testing->aug={0};adv={0}")
                                procObj.test(
                                    standard=standard,
                                    culture=c,
                                    augment=0,
                                    gaug=g_aug,
                                    adversary=0,
                                    eps=test_eps,
                                )
                                procObj.partial_clear(basePath)
