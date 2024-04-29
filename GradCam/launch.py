#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

sys.path.insert(1, "../")
from Processing.processing import ProcessingClass
from math import floor
import tensorflow as tf
from GradCam.gradCam import GradCAM
import cv2
from Utils.FileManager.FileManager import FileManagerClass

tf.config.set_soft_device_placement(True)

percents = [0.05, 0.1]
standards = [1, 0]

verbose_param = 1
n = 1000
bs = 2
learning_rate = 5e-4
val_split = 0.2
test_split = 0.1
epochs = 15

g_aug = 0.1
test_g_augs = [0.01, 0.05, 0.1]
eps = 0.03
test_eps = [0.0005, 0.001, 0.005]
mult = 0.25

nt = 10

memory_limit = 3000


for lamp in [1,0]:
    procObj = ProcessingClass(shallow=0, lamp=lamp, gpu=True)
    with tf.device("/CPU:0"):
        for standard in standards:
            if standard: 
                lim = 0
            else:
                lim = 13
            for j in range(-1, lim):
                for percent in percents:
                    for c in range(3):
                        for k in range(4):
                            print(f"Training->aug={k%2};adv={floor(k/2)}")
                            procObj.process(
                                standard=standard,
                                type="DL",
                                verbose_param=verbose_param,
                                learning_rate=learning_rate,
                                epochs=epochs,
                                batch_size=bs,
                                lambda_index=j,
                                culture=c,
                                percent=percent,
                                val_split=val_split,
                                test_split=test_split,
                                n=n,
                                augment=k % 2,
                                g_rot=g_aug,
                                g_noise=g_aug,
                                g_bright=g_aug,
                                adversary=floor(k / 2),
                                eps=eps,
                                mult=mult,
                                gradcam=0,
                            )

                            grdC = GradCAM(procObj.model.model, 0, "conv5_block3_out")
                            
                            if standard:
                                path = procObj.basePath + "out.jpg"
                                heatmap = grdC.compute_heatmap(procObj.dataobj.Xv[0:nt], path=path )
                                print(f"saved heatmap in file {path}")
                            else:
                                for out in range(3):
                                    path = procObj.basePath + "out" + out + ".jpg"
                                    heatmap = grdC.compute_heatmap(
                                        procObj.dataobj.Xv[0:nt], out=out, path = path
                                    )
                                    print(f"saved heatmap in file {path}")
                            procObj.partial_clear()
