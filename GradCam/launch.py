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

verbose_param = 0
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

nt = 5

memory_limit = 3000


for lamp in [1,0]:
    procObj = ProcessingClass(shallow=0, lamp=lamp, gpu=True, memory_limit=memory_limit)
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
                            # NoAUg
                            print(f"Testing->aug={0};adv={0}")
                            procObj.prepare_test(
                                augment=0,
                                g_rot=g_aug,
                                g_noise=g_aug,
                                g_bright=g_aug,
                                adversary=0,
                                eps=test_eps,
                                nt=nt
                            )
                            for culture in range(3):
                                if standard:
                                    path = procObj.basePath + f"TNOAUG/CULTURE{culture}/out/"
                                    fObj = FileManagerClass(path)
                                    del fObj
                                    heatmap = grdC.compute_heatmap(procObj.dataobj.Xt[culture], path=path )
                                    print(f"saved heatmap in file {path}")
                                else:
                                    for out in range(3):
                                        path = procObj.basePath + f"TNOAUG/CULTURE{culture}/out" + out + "/"
                                        fObj = FileManagerClass(path)
                                        del fObj
                                        heatmap = grdC.compute_heatmap(
                                            procObj.dataobj.Xt[culture], out=out, path = path
                                        )
                                        print(f"saved heatmap in file {path}")


                            print(f"Testing->aug={1};adv={0}")
                            for t_g_aug in test_g_augs:
                                procObj.prepare_test(
                                        augment=1,
                                        g_rot=t_g_aug,
                                        g_noise=t_g_aug,
                                        g_bright=t_g_aug,
                                        adversary=0,
                                        eps=None,
                                        nt=nt)
                                for culture in range(3):
                                    if standard:
                                        path = procObj.basePath + f"TSTDAUG/CULTURE{culture}/G_AUG={t_g_aug}/out/"
                                        fObj = FileManagerClass(path)
                                        del fObj
                                        heatmap = grdC.compute_heatmap(procObj.Xt_aug[culture], path=path )
                                        print(f"saved heatmap in file {path}")
                                    else:
                                        for out in range(3):
                                            path = procObj.basePath + f"TSTDAUG/CULTURE{culture}/G_AUG={t_g_aug}/out" + out + "/"
                                            fObj = FileManagerClass(path)
                                            del fObj
                                            heatmap = grdC.compute_heatmap(
                                                procObj.Xt_aug[culture], out=out, path = path
                                            )
                                            print(f"saved heatmap in file {path}")


                            print(f"Testing->aug={0};adv={1}")
                            for test_ep in test_eps:
                                procObj.prepare_test(
                                            augment=0,
                                            g_rot=None,
                                            g_noise=None,
                                            g_bright=None,
                                            adversary=1,
                                            eps=test_ep,
                                            nt=nt)
                                for culture in range(3):
                                    if standard:
                                        path = procObj.basePath + f"TAVD/CULTURE{culture}/EPS={eps}/out/"
                                        fObj = FileManagerClass(path)
                                        del fObj
                                        heatmap = grdC.compute_heatmap(procObj.Xt_adv[culture], path=path )
                                        print(f"saved heatmap in file {path}")
                                    else:
                                        for out in range(3):
                                            path = procObj.basePath + f"TAVD/CULTURE{culture}/EPS={eps}/out" + out + "/"
                                            fObj = FileManagerClass(path)
                                            del fObj
                                            heatmap = grdC.compute_heatmap(
                                                procObj.Xt_adv[culture], out=out, path = path
                                            )
                                            print(f"saved heatmap in file {path}")


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
                                        nt=nt)
                                    for culture in range(3):
                                        if standard:
                                            path = procObj.basePath + f"TTOTAUG/CULTURE{culture}/G_AUG={t_g_aug}/EPS={eps}/out/"
                                            fObj = FileManagerClass(path)
                                            del fObj
                                            heatmap = grdC.compute_heatmap(procObj.Xt_totaug[culture], path=path )
                                            print(f"saved heatmap in file {path}")
                                        else:
                                            for out in range(3):
                                                path = procObj.basePath + f"TTOTAUG/CULTURE{culture}/G_AUG={t_g_aug}/EPS={eps}/out" + out + "/"
                                                fObj = FileManagerClass(path)
                                                del fObj
                                                heatmap = grdC.compute_heatmap(
                                                    procObj.Xt_totaug[culture], out=out, path = path
                                                )
                                                print(f"saved heatmap in file {path}")
                                        
                            procObj.partial_clear()

                            
