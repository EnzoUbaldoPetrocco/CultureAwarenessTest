#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
from typing import Any
from imblearn.over_sampling import SMOTE
import random
import time
import numpy as np
import gc


class SMOTEGen:
    """
    Class implementing Generation of samples using SMOTE
    """
    def separate_classes(self, TS, n_classes=2):
        """
        This function separates the classes from training set
        :param TS: training set
        :param n_classes: number of classes
        :return list of training sets separated by classes
        """
        TSs = []
        X, y = TS[0], TS[1]
        for i in range(n_classes):
            tempX = []
            tempy = []
            for j in range(len(X)):
                if i == y[j][1]:
                    X[j] = X[j].flatten()
                    tempX.append(X[j])
                    tempy.append(y[j][0])
            TSs.append([tempX, tempy])
            del tempX
            del tempy
        return TSs

    def __call__(self, TS, n_classes=2):
        """
        Implementation of call function, so it uses SMOTE for culturally balancing the training samples
        :param TS: training set
        :param n_classes: number of classes
        :return training set with augmented data 
        """
        TSs = self.separate_classes(TS, n_classes=n_classes)
        X_res, y_res = [], []
        for i, ts in enumerate(TSs):
            x_res, y_res = self.call_smote(ts, i)
            X_res.extend(x_res)
            y_res.extend(y_res)

        return X_res, y_res

    def call_smote(self, TS, out=0):
        """
        This function implements SMOTE
        :param TS: training samples
        :param out: output to use for model prediction
        :return training samples balanced
        """
        rnd_state = random.seed(time.time())
        sm = SMOTE(random_state=rnd_state)
        X, y = TS[0], TS[1]

        X_res, y_res = sm.fit_resample(X, y)
        for i in range(len(X_res)):
            X_res[i] = np.reshape(
                X_res[i],
                (int(np.sqrt(len(X_res[i]) / 3)), int(np.sqrt(len(X_res[i]) / 3)), 3),
            )
            y_res[i] = [y_res[i], out]

        del rnd_state
        del sm
        gc.collect()
        return X_res, y_res
