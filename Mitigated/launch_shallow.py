#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys


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
import cv2

random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())


# tf.config.set_soft_device_placement(True)



percents = [0.05, 0.1, 0.2, 0.5]
standard = 1
tps = ["SVC", "RFC"]
kernels = ["linear", "rbf"]
points = 10
# lamp = 1

verbose_param = 1
n = 1000
cs = [0, 1, 2]
lamps = [1]


basePath = "./first_comparing_with_teacher/"
for i in range(5):
   for percent in percents:
      for lamp in lamps:
        for c in cs:
          for tp in tps:
            for kernel in kernels:
                if tp == "DL":
                   shallow=0
                else:
                   shallow=1
                procObj = ProcessingClass(
                    shallow=shallow,
                    lamp=lamp,
                    gpu=False,
                    memory_limit=0,
                    basePath=basePath,
                )
                model = None
                print(f"Training->Model={tp};kernel={kernel}")
                procObj.process(
                    standard=standard,
                    type=tp,
                    kernel=kernel,
                    points = points,
                    verbose_param=verbose_param,
                    culture=c,
                    percent=percent,
                    n=n,
                    mitigation_type=0
                )
                # NoAUg
                print(f"Testing->aug={0};adv={0}")
                procObj.test(
                    standard=standard,
                    culture=c,
                    augment=0,
                    gaug=0,
                    adversary=0,
                )
                procObj.partial_clear(basePath)
                    
                        