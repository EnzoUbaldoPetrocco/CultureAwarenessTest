from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot as plt


class GeneralModelClass:

    def __init__(self) -> None:
        self.model = 0

    def __call__(self, X):
        if self.model != None:
            with tf.device("/gpu:0"):
                if tf.is_tensor(X):
                    yP = self.model(X)
                    if tf.shape(yP)[0] > 1:
                        return tf.gather(yP, indices=[[0, 0], [1, 0], [2, 0]])
                    else:
                        return yP[0][0]
                else:
                    yP = np.asarray(self.model(X))
                    if type(self.model) == keras.engine.functional.Functional:
                        if tf.shape(yP)[0] > 1:
                            # print(np.shape(yP))
                            return yP[:, 0]
                        else:
                            # print(np.shape(yP))
                            return yP[0]
                    else:
                        return yP
        else:
            print("Try fitting the model before")
            return None

    def quantize(self, yF):
        values = []
        for y in yF:
            if y > 0.5:
                values.append(1)
            else:
                values.append(0)
        return values

    def test(self, Xt, out=-1):
        if self.model:
            yF = []
            for xt in Xt:
                if out < 0:
                    yF.append(np.asarray(self(xt[None, ...])))
                else:
                    # print(np.shape(self.model(xt[None, ...])))
                    pL = np.asarray(self(xt[None, ...]))[out]
                    yF.append(pL)
                    #print(f"predicted label is {pL}")
                    #plt.imshow(xt)
                    #plt.show()
            yFq = self.quantize(yF)
            return yFq
        else:
            print("Try fitting the model before")
            return None

    def get_model_stats(self, Xt, yT, out=-1):
        yFq = self.test(Xt, out)
        if len(np.shape(yT)) > 1:
            if type(yT) == np.ndarray:
                yT = yT[:, 1]
            elif type(yT) == list:
                yT = np.asarray(yT)[:, 1]
        
        if yFq:
            # yT = list([c_i, y_i])
            cm = confusion_matrix(y_true=yT, y_pred=yFq)
            return cm
