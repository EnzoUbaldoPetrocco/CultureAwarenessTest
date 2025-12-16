#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

import cv2

sys.path.insert(1, "../")
from Utils.FileManager.FileManager import FileManagerClass
from Processing.processing import ProcessingClass
from math import floor
import random
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
import numpy as np


lamps = [0,1]
basePath = "./KMeans/"
lamp = 0
procObj = ProcessingClass(
                    shallow=0,
                    lamp=lamp,
                    basePath=basePath,
                )
procObj.dataobj.prepare(
            standard=0,
            culture=0,
            percent=1.0,
            shallow=0,
            val_split=0.1,
            test_split=0.2,
            n=1000,
            adversarial=0,
            imbalanced=0,
        )
X = np.asarray(procObj.dataobj.X).reshape(np.shape(procObj.dataobj.X)[0], -1)
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
print(f"n_clusters={n_clusters}")
print("Labels:")
print(kmeans.labels_)
print("Cluster Centers:")
print(kmeans.cluster_centers_)
print("\n")

for i in range(3):
    Xt = np.asarray(procObj.dataobj.Xt[i]).reshape(np.shape(procObj.dataobj.Xt[i])[0], -1)
    yt = np.asarray(procObj.dataobj.yt[i])
    labels = kmeans.predict(Xt)
    print(f"Culture Set {i}:")
    print("Labels Predicted:")
    print(labels)
    print("True Labels:")
    print(yt)