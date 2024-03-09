from typing import Any
from imblearn.over_sampling import SMOTE
import random
import time
import numpy as np
import gc


class SMOTEGen:

    def separate_classes(self, TS, n_classes=2):
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

        TSs = self.separate_classes(TS, n_classes=n_classes)
        X_res, y_res = [], []
        for i, ts in enumerate(TSs):
            x_res, y_res = self.call_smote(ts, i)
            X_res.extend(x_res)
            y_res.extend(y_res)

        return X_res, y_res

    def call_smote(self, TS, out=0):
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
