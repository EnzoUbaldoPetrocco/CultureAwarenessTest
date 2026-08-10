#!/usr/bin/env python
"""
GradCAM Visual Interpretability Execution Script.

This script trains model instances and computes Gradient-weighted Class Activation Maps (GradCAM)
heatmaps on key convolutional layers (e.g., 'conv5_block3_out' of ResNet) to visually inspect 
model attention maps across target cultures and confusion matrix buckets (TP, TN, FP, FN).

Author: Enzo Ubaldo Petrocco
"""

import sys
import os
from math import floor
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# Insert parent directory for module imports
sys.path.insert(1, "../")
from Processing.processing import ProcessingClass
from GradCam.gradCam import GradCAM
from Utils.FileManager.FileManager import FileManagerClass

# Force soft device placement for GPU/CPU execution flexibility
tf.config.set_soft_device_placement(True)


def mkdirs(path: str, nt: int):
    """
    Create result directory subtrees for confusion matrix sample buckets.
    
    Args:
        path (str): Root directory path for heatmaps.
        nt (int): Target number of samples per bucket.
    """
    for i in range(nt):
        fObj = FileManagerClass(path + f"TP/{i}/")
        del fObj
        fObj = FileManagerClass(path + f"TN/{i}/")
        del fObj
        fObj = FileManagerClass(path + f"FP/{i}/")
        del fObj
        fObj = FileManagerClass(path + f"FN/{i}/")
        del fObj


def get_cm_samples(procObj: ProcessingClass, Xt, yt, out: int, n=1, standard=0):
    """
    Categorize dataset test samples into True Positive, True Negative, False Positive, 
    and False Negative buckets based on model predictions.
    
    Args:
        procObj (ProcessingClass): Main processing orchestrator object.
        Xt (list/array): Test input images.
        yt (list/array): Ground truth labels.
        out (int): Output head index (culture output head or standard head).
        n (int): Max samples per bucket.
        standard (int): Standard control (1) vs Mitigated (0) model switch.
        
    Returns:
        list: 2x2 matrix containing lists of sampled images for [[TN, FP], [FN, TP]].
    """
    if not standard:
        yt = list(np.asarray(yt, dtype=object)[:, 1])
    
    yP = procObj.model.test(Xt, out)  # Model predictions
    cm = confusion_matrix(yt, yP)
    
    # True Negatives (y_true=0, y_pred=0)
    if cm[0][0] == 0:
        tn = []
    else:
        tns = [Xt[i] for i in range(len(Xt)) if yP[i] == yt[i] and yP[i] == 0]
        tn = tns[: min(n, len(tns) - 1)] if tns else []
        del tns

    # False Negatives (y_true=1, y_pred=0)
    if cm[1][0] == 0:
        fn = []
    else:
        fns = [Xt[i] for i in range(len(Xt)) if yP[i] != yt[i] and yP[i] == 0]
        fn = fns[: min(n, len(fns) - 1)] if fns else []
        del fns

    # False Positives (y_true=0, y_pred=1)
    if cm[0][1] == 0:
        fp = []
    else:
        fps = [Xt[i] for i in range(len(Xt)) if yP[i] != yt[i] and yP[i] == 1]
        fp = fps[: min(n, len(fps) - 1)] if fps else []
        del fps

    # True Positives (y_true=1, y_pred=1)
    if cm[0][0] == 0:
        tp = []
    else:
        tps = [Xt[i] for i in range(len(Xt)) if yP[i] == yt[i] and yP[i] == 1]
        tp = tps[: min(n, len(tps) - 1)] if tps else []
        del tps

    return [[tn, fp], [fn, tp]]


def cmp_and_save_heatmap(pt: str, standard: int, grdC: GradCAM, Xt, yt, procObj: ProcessingClass, images_test_path: str):
    """
    Compute and save GradCAM activation heatmaps for all culture subsets across CM buckets.
    
    Args:
        pt (str): Base destination directory for heatmaps.
        standard (int): Standard (1) vs Mitigated (0) model toggle.
        grdC (GradCAM): Initialized GradCAM explainer object.
        Xt (list): Per-culture test input images.
        yt (list): Per-culture ground truth labels.
        procObj (ProcessingClass): Active pipeline processor instance.
        images_test_path (str): Test images root directory.
    """
    for culture in range(3):
        path = pt + f"CULTURE{culture}/"
        mkdirs(path, nt)
        out = -1 if standard else culture

        # Retrieve confusion matrix sample buckets for the target culture
        cms = get_cm_samples(procObj, Xt[culture], yt[culture], out, n=nt, standard=standard)
        
        # Generate and write heatmaps per bucket
        _ = grdC.compute_heatmap(cms[0][0], out=out, path=path + "TN/")
        _ = grdC.compute_heatmap(cms[1][0], out=out, path=path + "FN/")
        _ = grdC.compute_heatmap(cms[0][1], out=out, path=path + "FP/")
        _ = grdC.compute_heatmap(cms[1][1], out=out, path=path + "TP/")


# ==============================================================================
# PIPELINE HYPERPARAMETERS & EXECUTION LOOP
# ==============================================================================
percent = 0.05
standard = 1
verbose_param = 0
n = 1000
bs = 2
learning_rate = 5e-4
val_split = 0.2
test_split = 0.1
epochs = 10
nt = 5                # Number of sample heatmaps per CM category
memory_limit = 5000

basePath = "./"
images_test_path = './STD/Dl/BAL/'

# Execute across Carpets (0) and Lamps (1) datasets
for lamp in [0, 1]:
    print(f"\n========================================================")
    print(f"[GradCAM Launch] Starting visual explanation for Lamp={lamp}")
    print(f"========================================================")
    
    procObj = ProcessingClass(shallow=0, lamp=lamp, gpu=True, memory_limit=memory_limit, basePath=basePath)
    
    with tf.device("/CPU:0"):
        for c in range(3):  # For each target majority culture
            for k in range(1):
                print(f"Training GradCAM Model -> Culture: {c}, Augment: {k%2}, Adversarial: {floor(k/2)}")
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
                    gradcam=1,      # Instantiate model with GradCAM feature map hooks
                    complete=1,
                )

                # Initialize GradCAM visualizer targeting final ResNet conv layer
                grdC = GradCAM(procObj.model.model, 0, "conv5_block3_out")
                
                print(f"Generating and saving heatmaps for Culture {c}...")
                pt = procObj.basePath + f"TNOAUG/"
                Xt = procObj.dataobj.Xt
                yt = procObj.dataobj.yt
                
                cmp_and_save_heatmap(pt, standard, grdC, Xt, yt, procObj, images_test_path)
                procObj.partial_clear(basePath)

print("\n[GradCAM Launch] Visual heatmaps generated successfully.")



                        