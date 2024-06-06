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

memory_limit = 3000

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
        print(
            len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
        )

    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
else:
    print("no gpus")

percents = [0.05]
standard = 1
lamp = 0

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
memory_limit = 5000
cs = [2,1,0]
ks = [0,1]

basePath = './PREDICT/'


procObj = ProcessingClass(shallow=0, lamp=lamp, gpu=True, memory_limit=memory_limit, basePath=basePath)
with tf.device("/CPU:0"):
        for g_aug in g_gaugs:
            for percent in percents:
                for c in cs:
                    for k in ks:
                        model = None
                        for i in range(2):
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
                                augment=1,#k % 2,
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
                            """print(f"Testing->aug={1};adv={0}")
                            for t_g_aug in test_g_augs:
                                procObj.test(
                                        standard=standard,
                                        culture=c,
                                        augment=1,
                                        g_rot=t_g_aug,
                                        g_noise=t_g_aug,
                                        g_bright=t_g_aug,
                                        adversary=0,
                                        eps=None)
                            print(f"Testing->aug={0};adv={1}")
                            for test_ep in test_eps:
                                procObj.test(
                                            standard=standard,
                                            culture=c,
                                            augment=0,
                                            g_rot=None,
                                            g_noise=None,
                                            g_bright=None,
                                            adversary=1,
                                            eps=test_ep)
                            print(f"Testing->aug={1};adv={1}")
                            for t, t_g_aug in enumerate(test_g_augs):
                                for test_ep in test_eps:  
                                    procObj.test(
                                        standard=standard,
                                        culture=c,
                                        augment=1,
                                        g_rot=t_g_aug,
                                        g_noise=t_g_aug,
                                        g_bright=t_g_aug,
                                        adversary=1,
                                        eps=test_ep)
                            model = procObj.model.model   
                            path = procObj.basePath + "out.jpg"""
                            procObj.partial_clear(basePath)
