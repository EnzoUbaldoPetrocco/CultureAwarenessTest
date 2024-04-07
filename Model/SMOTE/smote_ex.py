#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

sys.path.insert(1, "../../")
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

from Processing.processing import ProcessingClass
from matplotlib import pyplot as plt
from Utils.Data.Data import DataClass
from Utils.FileManager.FileManager import FileManagerClass
from Utils.Results.Results import ResultsClass
from Utils.Data.deep_paths import DeepStrings
from Utils.Data.shallow_paths import ShallowStrings
from Model.SMOTE.SMOTEGen import SMOTEGen
import numpy as np
import random
import time

# Example of SMOTE
X, y = make_classification(n_classes=2, class_sep=2,
weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
print('Original dataset shape %s' % Counter(y))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

def get_culture_set(X, y, culture):
    Xc = []
    yc = []
    shape_ = np.shape(X[0])
    for i in range(len(X)):
        if y[i][0]==culture:
            #X_i = X[i]
            Xc.append(list(np.reshape(X[i], shape_[0]*shape_[1]*shape_[2])))
            yc.append(y[i][0])
    return Xc, yc

# Example of SMOTE with our LAMP dataset in RGB

strObj = DeepStrings("../../../")
paths = strObj.lamp_paths
dataobj = DataClass(paths)
dataobj.prepare(standard=0, culture=0, percent=0.1, shallow=False, val_split=0.2, test_split=0.2)


#Example of SMOTE using my custom class
print("------------------")
print("Example using custom classes such as SMOTEGen")
dataobj.clear()
dataobj.prepare(standard=0, culture=0, percent=0.1, shallow=False, val_split=0.2, test_split=0.2)
TS = (dataobj.X, dataobj.y)
print('Original dataset shape %s' % Counter(np.asarray(TS[1])[:, 0]))
start_time = time.time()
smoteGen = SMOTEGen()
X_res, y_res = smoteGen(TS)
print("--- %s seconds ---" % (time.time() - start_time))
print('Resampled dataset shape %s' % Counter(np.asarray(y_res)[:,0]))