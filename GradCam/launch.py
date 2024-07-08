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
from sklearn.metrics import confusion_matrix
import numpy as np

tf.config.set_soft_device_placement(True)


def mkdirs(path, nt):
    for i in range(nt):
        fObj = FileManagerClass(path + f"TP/{i}/")
        del fObj
        fObj = FileManagerClass(path + f"TN/{i}/")
        del fObj
        fObj = FileManagerClass(path + f"FP/{i}/")
        del fObj
        fObj = FileManagerClass(path + f"FN/{i}/")
        del fObj


def get_cm_samples(procObj: ProcessingClass, Xt, yt, out, n=1, standard=0):
    if not standard:
        print(f"Shape of yt is {np.shape(yt)}")
        yt = list(np.asarray(yt, dtype=object)[:,1])
    yP = procObj.model.test(Xt, out)  # Prediction
    cm = confusion_matrix(yt, yP)
    if cm[0][0] == 0:  # tn
        tn = []
    else:
        tns = [Xt[i] for i in range(len(Xt)) if yP[i] == yt[i] and yP[i] == 0  ]
        tn = tns[: min(n, len(tns) - 1)]
        del tns
    if cm[1][0] == 0:  # fn
        fn = []
    else:
        fns = [Xt[i] for i in range(len(Xt)) if yP[i] != yt[i] and yP[i] == 0  ]
        fn = fns[: min(n, len(fns) - 1)]


        del fns
    if cm[0][1] == 0:  # fp dim
        fp = []
    else:
        fps = [Xt[i] for i in range(len(Xt)) if yP[i] != yt[i] and yP[i] == 1  ]
        fp = fps[: min(n, len(fps) - 1)]

        del fps
    if cm[0][0] == 0:  # tp dim
        tp = []
    else:
        tps = [Xt[i] for i in range(len(Xt)) if yP[i] == yt[i] and yP[i] == 1  ]
        tp = tps[: min(n, len(tps) - 1)]

        del tps

    cms= [[tn, fp], [fn, tp]]
    return cms


def cmp_and_save_heatmap(pt, standard, grdC: GradCAM, Xt, yt, procObj: ProcessingClass):
    for culture in range(3):
        path = pt + f"CULTURE{culture}/"
        mkdirs(path, nt)
        if standard:
            out = -1
        else:
            out = culture
        cms = get_cm_samples(procObj, Xt[culture], yt[culture], out, n=nt)
        
        heatmap = grdC.compute_heatmap(
            cms[0][0],
            out=out,
            path=path + "TN/",
        )
        heatmap = grdC.compute_heatmap(
            cms[1][0],
            out=out,
            path=path + "FN/",
        )
        heatmap = grdC.compute_heatmap(
            cms[0][1],
            out=out,
            path=path + "FP/",
        )
        heatmap = grdC.compute_heatmap(
            cms[1][1],
            out=out,
            path=path + "TP/",
        )


percent = 0.05
standard = 1

verbose_param = 0
n = 1000
bs = 2
learning_rate = 5e-4
val_split = 0.2
test_split = 0.1
epochs = 10

g_augs = [0.001, 0.05, 0.1, 0.5]
test_g_augs = [0.0005, 0.005]
eps = 0.03
test_eps = [0.0005, 0.005]
mult = 0.25

nt = 10

memory_limit = 5000


for lamp in [0, 1]:
    procObj = ProcessingClass(shallow=0, lamp=lamp, gpu=True, memory_limit=memory_limit)
    with tf.device("/CPU:0"):
                for c in range(3):
                    for k in range(1):
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
                            augment=0,
                            gaug=0,
                            adversary=0,
                            eps=0,
                            mult=0,
                            gradcam=0,
                            complete=1,
                        )

                        grdC = GradCAM(procObj.model.model, 0, "conv5_block3_out")
                        # NoAUg
                        print(f"Testing->aug={0};adv={0}")
                        pt = procObj.basePath + f"TNOAUG/"
                        Xt = procObj.dataobj.Xt
                        yt = procObj.dataobj.yt
                        cmp_and_save_heatmap(pt, standard, grdC, Xt, yt, procObj)


                        