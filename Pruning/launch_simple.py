#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys


sys.path.insert(1, "../")
from Processing.processing import ProcessingClass
from math import floor
import tensorflow as tf
import random
from datetime import datetime
import numpy as np
from Pruning.pruning import Pruning
import os

random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# tf.config.set_soft_device_placement(True)

memory_limit = 2000
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


# lamp = 1

verbose_param = 1
n = 1000
cs = [0, 1, 2]
lamps = [1, 0]

standard = 1
percent = 0.05

# Pruning values
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.CategoricalCrossentropy()
metrics = ["accuracy"]
batch_size = 32
epochs = 2


basePath = "./"
for lamp in lamps:
    for c in cs:
        for i in range(5):
            procObj = ProcessingClass(
                shallow=0,
                lamp=lamp,
                gpu=False,
                memory_limit=memory_limit,
                basePath=basePath,
            )
            print(f"Training->aug={0};adv={0}")
            procObj.process(
                standard=standard,
                type="DL",
                verbose_param=verbose_param,
                culture=c,
                percent=percent,
                n=n,
            )
            print(f"Testing->aug={0};adv={0}")
            # start with the original model
            model = tf.keras.models.clone_model(procObj.model)
            procObj.test(
                standard=standard,
                culture=c,
            )
            # Test Model with pruning
            pruningObj = Pruning(
                model=model,
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
            pruningObj.fine_tune_pruned_model(
                x_train=procObj.dataobj.Xt,
                y_train=procObj.dataobj.yt,
                x_val=procObj.dataobj.Xv,
                y_val=procObj.dataobj.yv,
                batch_size=batch_size,
                epochs=epochs,
            )
            final_model = pruningObj.strip_pruning()
            procObj.model = final_model
            procObj.basePath = procObj.basePath + "pruned/"
            if not os.path.exists(procObj.basePath):
                os.makedirs(procObj.basePath)
            procObj.test(
                standard=standard,
                culture=c,
            )
            procObj.partial_clear(basePath)
