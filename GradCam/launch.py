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


percents = [0.05, 0.1]
standards = [0]

verbose_param = 0
n = 1000
bs = 2
learning_rate = 5e-4
val_split = 0.2
test_split = 0.1
epochs = 1

g_aug = 0.05
test_g_augs = [0.0005, 0.005]
eps = 0.03
test_eps = [0.0005, 0.005]
mult = 0.25

nt = 1

memory_limit = 3000


for lamp in [0, 1]:
    procObj = ProcessingClass(shallow=0, lamp=lamp, gpu=True, memory_limit=memory_limit)
    with tf.device("/CPU:0"):
        for standard in standards:
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
                            lambda_index=0,
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
                            complete=1,
                        )

                        grdC = GradCAM(procObj.model.model, 0, "conv5_block3_out")
                        # NoAUg
                        print(f"Testing->aug={0};adv={0}")
                        pt = procObj.basePath + f"TNOAUG/"
                        Xt = procObj.dataobj.Xt
                        yt = procObj.dataobj.yt
                        cmp_and_save_heatmap(pt, standard, grdC, Xt, yt, procObj)

                        print(f"Testing->aug={1};adv={0}")
                        for t_g_aug in test_g_augs:
                            procObj.prepare_test(
                                augment=1,
                                g_rot=t_g_aug,
                                g_noise=t_g_aug,
                                g_bright=t_g_aug,
                                adversary=0,
                                eps=None,
                            )
                            pt = procObj.basePath + f"TSTDAUG/G_AUG={t_g_aug}/"
                            Xt = procObj.Xt_aug
                            cmp_and_save_heatmap(pt, standard, grdC, Xt, yt, procObj)

                        print(f"Testing->aug={0};adv={1}")
                        for test_ep in test_eps:
                            procObj.prepare_test(
                                augment=0,
                                g_rot=None,
                                g_noise=None,
                                g_bright=None,
                                adversary=1,
                                eps=test_ep,
                            )
                            pt = procObj.basePath + f"TAVD/EPS={test_ep}/"
                            Xt = procObj.Xt_adv
                            cmp_and_save_heatmap(pt, standard, grdC, Xt, yt, procObj)

                        print(f"Testing->aug={1};adv={1}")
                        for t, t_g_aug in enumerate(test_g_augs):
                            for test_ep in test_eps:
                                procObj.prepare_test(
                                    augment=1,
                                    g_rot=t_g_aug,
                                    g_noise=t_g_aug,
                                    g_bright=t_g_aug,
                                    adversary=1,
                                    eps=test_ep,
                                )
                                pt = (
                                    procObj.basePath
                                    + f"TTOTAUG//G_AUG={t_g_aug}/EPS={test_ep}/"
                                )
                                Xt = procObj.Xt_totaug
                                cmp_and_save_heatmap(pt, standard, grdC, Xt, yt, procObj)

                        procObj.partial_clear()
